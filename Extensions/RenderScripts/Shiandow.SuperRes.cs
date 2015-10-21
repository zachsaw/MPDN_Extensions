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
using System.Linq;
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.Framework.Chain;
using Mpdn.Extensions.Framework.RenderChain;
using Mpdn.Extensions.RenderScripts.Hylian.SuperXbr;
using Mpdn.Extensions.RenderScripts.Mpdn.OclNNedi3;
using Mpdn.Extensions.RenderScripts.Mpdn.ScriptGroup;
using Mpdn.Extensions.RenderScripts.Shiandow.Nedi;
using Mpdn.Extensions.RenderScripts.Shiandow.NNedi3;
using Mpdn.RenderScript;
using Mpdn.RenderScript.Scaler;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Shiandow.SuperRes
    {
        public class SuperRes : RenderChain
        {
            #region Settings

            public ScriptGroup PrescalerGroup { get; set; }

            public int Passes { get; set; }
            public float Strength { get; set; }
            public float Softness { get; set; }

            public bool HQdownscaling { get; set; }

            #endregion

            public Func<TextureSize> TargetSize; // Not saved
            private readonly IScaler m_Downscaler;
            private readonly IScaler m_Upscaler;

            public SuperRes()
            {
                TargetSize = () => Renderer.TargetSize;

                Passes = 2;
                Strength = 1.0f;
                Softness = 0.0f;

                HQdownscaling = true;

                PrescalerGroup = new ScriptGroup
                {
                    Options = new List<IRenderChainUi>
                    {
                        new SuperXbrUi(),
                        new NediScaler(),
                        new NNedi3Scaler(),
                        new OclNNedi3Scaler()
                    }
                        .Select(x => x.ToPreset())
                        .ToList()
                };

                PrescalerGroup.SelectedIndex = 0;

                m_Upscaler = new Jinc(ScalerTaps.Four, false); // Deprecated
                m_Downscaler = HQdownscaling ? (IScaler) new Bicubic(0.75f, false) : new Bilinear();
            }

            protected override IFilter CreateFilter(IFilter input)
            {
                var option = PrescalerGroup.SelectedOption;
                return option == null ? input : CreateFilter(input, input + option);
            }

            private bool IsIntegral(double x)
            {
                return Math.Abs(x - Math.Truncate(x)) < 0.005;
            }

            public IFilter CreateFilter(IFilter original, IFilter initial)
            {
                IFilter result;

                // Calculate Sizes
                var inputSize = original.OutputSize;
                var targetSize = TargetSize();

                string macroDefinitions = "";
                if (IsIntegral(Strength))
                    macroDefinitions += string.Format("strength = {0};", Strength);
                if (IsIntegral(Softness))
                    macroDefinitions += string.Format("softness = {0};", Softness);

                // Compile Shaders
                var Diff = CompileShader("Diff.hlsl")
                    .Configure(format: TextureFormat.Float16);

                var SuperRes = CompileShader("SuperResEx.hlsl", macroDefinitions: macroDefinitions)
                    .Configure(
                        arguments: new[] { Strength, Softness }
                    );

                var FinalSuperRes = CompileShader("SuperResEx.hlsl", macroDefinitions: macroDefinitions + "FinalPass = 1;")
                    .Configure(
                        arguments: new[] { Strength, Softness }
                    );

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
                if (initial != original)
                {
                    // Always correct offset (if any)
                    var filter = initial as ResizeFilter;
                    if (filter != null)
                        filter.ForceOffsetCorrection();

                    result = new ShaderFilter(GammaToLinear, initial.SetSize(targetSize));
                }
                else
                {
                    result = new ResizeFilter(new ShaderFilter(GammaToLinear, original), targetSize);
                }

                for (int i = 1; i <= Passes; i++)
                {
                    // Downscale and Subtract
                    var loRes = new ResizeFilter(result, inputSize, m_Upscaler, m_Downscaler);
                    var diff = new ShaderFilter(Diff, loRes, original);

                    // Update result
                    result = new ShaderFilter(i != Passes ? SuperRes : FinalSuperRes, result, diff);
                }

                return result;
            }

            private TextureSize CalculateSize(TextureSize sizeA, TextureSize sizeB, int k, int passes) // Deprecated
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
