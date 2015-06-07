// This file is a part of MPDN Extensions.
// https://github.com/zachsaw/MPDN_Extensions
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
// 
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public
// License along with this library.

using System;
using System.Collections.Generic;
using Mpdn.Extensions.Framework.RenderChain;
using Mpdn.Extensions.RenderScripts.Mpdn.OclNNedi3;
using Mpdn.Extensions.RenderScripts.Mpdn.ScriptChain;
using Mpdn.Extensions.RenderScripts.Mpdn.ScriptGroup;
using Mpdn.RenderScript;
using Mpdn.RenderScript.Scaler;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Shiandow.SuperRes
    {
        public class SuperResPreset : Preset
        {
            public int Passes { get; set; }
            public float Strength { get; set; }
            public float Sharpness { get; set; }
            public float AntiAliasing { get; set; }
            public float AntiRinging { get; set; }
            public float Softness { get; set; }
        }

        public class SuperRes : ScriptGroup
        {
            #region Settings

            public bool NoIntermediates { get; set; }

            public bool FirstPassOnly;

            #endregion

            public override bool AllowRegrouping { get { return false; } }

            public Func<TextureSize> TargetSize; // Not saved
            private readonly IScaler m_Downscaler;
            private readonly IScaler m_Upscaler;

            public SuperRes()
            {
                TargetSize = () => Renderer.TargetSize;

                Options = new List<Preset>
                {
                    new SuperResPreset
                    {
                        Name = "Default",
                        Passes = 3,
                        Strength = 0.75f,
                        Sharpness = 0.3f,
                        AntiAliasing = 0.3f,
                        AntiRinging = 0.5f,
                        Softness = 0.1f,
                        Script = new ScriptChainScript()
                    },
                    new SuperResPreset
                    {
                        Name = "NEDI",
                        Passes = 2,
                        Strength = 0.65f,
                        Sharpness = 0.25f,
                        AntiAliasing = 0.25f,
                        AntiRinging = 0.0f,
                        Softness = 0.25f,
                        Script = new Nedi.NediScaler {Settings = new Nedi.Nedi {ForceCentered = true}}
                    },
                    new SuperResPreset
                    {
                        Name = "NNEDI3",
                        Passes = 2,
                        Strength = 0.35f,
                        Sharpness = 0.25f,
                        AntiAliasing = 0.25f,
                        AntiRinging = 0.0f,
                        Softness = 0.25f,
                        Script = new NNedi3.NNedi3Scaler {Settings = new NNedi3.NNedi3 {ForceCentered = true}}
                    },
                    new SuperResPreset
                    {
                        Name = "OpenCL NNEDI3",
                        Passes = 2,
                        Strength = 0.35f,
                        Sharpness = 0.25f,
                        AntiAliasing = 0.25f,
                        AntiRinging = 0.0f,
                        Softness = 0.25f,
                        Script = new OclNNedi3Scaler {Settings = new OclNNedi3 {ForceCentered = true}}
                    }
                };

                SelectedIndex = 0;

                NoIntermediates = false;
                FirstPassOnly = false;
                m_Upscaler = new Jinc(ScalerTaps.Four, false);
                m_Downscaler = new Bilinear();
            }

            public override IFilter CreateFilter(IFilter input)
            {
                return CreateFilter(input, input + SelectedOption);
            }

            public IFilter CreateFilter(IFilter original, IFilter initial)
            {
                IFilter lab;
                IFilter result = initial;

                // Load Settings
                var settings = (SuperResPreset)SelectedOption;
                var passes = settings.Passes;
                var strength = settings.Strength;
                var sharpness = settings.Sharpness;
                var AntiAliasing = settings.AntiAliasing;
                var AntiRinging = settings.AntiRinging;
                var Softness = settings.Softness;

                // Calculate Sizes
                var inputSize = original.OutputSize;
                var currentSize = original.OutputSize;
                var targetSize = TargetSize();

                // Compile Shaders
                var Diff = CompileShader("Diff.hlsl").Configure(format: TextureFormat.Float16);
                var SuperRes = CompileShader("SuperRes.hlsl", macroDefinitions:
                        (AntiRinging  == 0 ? "SkipAntiRinging = 1;" : "") +
                        (AntiAliasing == 0 ? "SkipAntiAliasing = 1;" : "") +
                        (sharpness == 0 ? "SkipSharpening = 1;" : "") +
                        (Softness  == 0 ? "SkipSoftening = 1;" : "")
                    ).Configure(arguments: new[] { strength, sharpness, AntiAliasing, AntiRinging, Softness });

                var GammaToLab = CompileShader("../Common/GammaToLab.hlsl");
                var LabToGamma = CompileShader("../Common/LabToGamma.hlsl");
                var LinearToGamma = CompileShader("../Common/LinearToGamma.hlsl");
                var GammaToLinear = CompileShader("../Common/GammaToLinear.hlsl");
                var LabToLinear = CompileShader("../Common/LabToLinear.hlsl");
                var LinearToLab = CompileShader("../Common/LinearToLab.hlsl");

                // Skip if downscaling
                if (targetSize.Width <= inputSize.Width && targetSize.Height <= inputSize.Height)
                    return original;

                // Initial scaling
                currentSize = CalculateSize(currentSize, targetSize, 1, passes);
                if (initial != original)
                {
                    original = new ShaderFilter(GammaToLab, original);
                    lab = new ShaderFilter(GammaToLab, initial.SetSize(currentSize));
                }
                else
                {
                    original = new ShaderFilter(GammaToLab, original);
                    lab = new ResizeFilter(original, currentSize);
                }

                for (int i = 1; i <= passes; i++)
                {
                    IFilter res, diff, linear;
                    bool useBilinear = (m_Upscaler is Bilinear) || (FirstPassOnly && i != 1);

                    if (i != 1)
                    {
                        // Calculate size
                        if (i == passes || NoIntermediates) currentSize = targetSize;
                        else currentSize = CalculateSize(currentSize, targetSize, i, passes);

                        // Resize
                        lab = new ResizeFilter(lab, currentSize, m_Upscaler, m_Downscaler);
                    }

                    // Downscale and Subtract
                    linear = new ShaderFilter(LabToLinear, lab);
                    res = new ResizeFilter(linear, inputSize, m_Upscaler, m_Downscaler); // Downscale result
                    diff = new ShaderFilter(Diff, res, original);                    // Compare with original

                    // Scale difference back
                    if (!useBilinear)
                        diff = new ResizeFilter(diff, currentSize, m_Upscaler, m_Downscaler);
                    
                    // Update result
                    lab = new ShaderFilter(
                        SuperRes.Configure(perTextureLinearSampling: new[] { false, useBilinear, false }),
                        lab, diff, original);
                    result = new ShaderFilter(LabToGamma, lab);
                }

                return result;
            }

            private TextureSize CalculateSize(TextureSize sizeA, TextureSize sizeB, int k, int passes)
            {            
                double w, h;
                var maxScale = 2.25;
                var minScale = Math.Sqrt(maxScale);
                
                int minW = sizeA.Width; int minH = sizeA.Height;
                int maxW = sizeB.Width; int maxH = sizeB.Height;

                int maxSteps = (int)Math.Floor  (Math.Log(maxH * maxW / (double)(minH * minW)) / (2 * Math.Log(minScale)));
                int minSteps = (int)Math.Ceiling(Math.Log(maxH * maxW / (double)(minH * minW)) / (2 * Math.Log(maxScale)));
                int steps = Math.Max(Math.Max(1,minSteps), Math.Min(maxSteps, passes - (k - 1)));
                
                w = minW * Math.Pow(maxW / (double)minW, Math.Min(k, steps) / (double)steps);
                h = minW * Math.Pow(maxH / (double)minH, Math.Min(k, steps) / (double)steps);

                return new TextureSize(Math.Max(minW, Math.Min(maxW, (int)Math.Round(w))),
                                       Math.Max(minH, Math.Min(maxH, (int)Math.Round(h))));
            }
        }

        public class SuperResUi : RenderChainUi<SuperRes, SuperResConfigDialog>
        {
            protected override string ConfigFileName
            {
                get { return "Shiandow.SuperRes"; }
            }

            public override string Category
            {
                get { return "Upscaling"; }
            }

            public override ExtensionUiDescriptor Descriptor
            {
                get
                {
                    return new ExtensionUiDescriptor
                    {
                        Guid = new Guid("3E7C670C-EFFB-41EB-AC19-207E650DEBD0"),
                        Name = "SuperRes",
                        Description = "SuperRes image scaling",
                        Copyright = "SuperRes by Shiandow",
                    };
                }
            }
        }
    }
}
