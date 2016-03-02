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
using Mpdn.Extensions.Framework.RenderChain;
using Mpdn.RenderScript;
using Mpdn.RenderScript.Scaler;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Shiandow.SuperRes
    {
        public class SuperRes : RenderChain
        {
            #region Settings

            public RenderScriptGroup PrescalerGroup { get; set; }

            public int Passes { get; set; }
            public float Strength { get; set; }
            public float Softness { get; set; }

            public bool LegacyDownscaling { get; set; }

            #endregion

            public Func<TextureSize> TargetSize; // Not saved

            public SuperRes()
            {
                TargetSize = () => Renderer.TargetSize;

                Passes = 2;
                Strength = 1.0f;
                Softness = 0.0f;

                LegacyDownscaling = false;

                var fastSuperXbrUi = new Hylian.SuperXbr.SuperXbrUi
                {
                    Settings = new Hylian.SuperXbr.SuperXbr
                    {
                        FastMethod = true,
                        ThirdPass = false
                    }
                }.ToPreset("Super-xBR (Fast)");

                PrescalerGroup = new RenderScriptGroup
                {
                    Options = (new[] {fastSuperXbrUi})
                        .Concat(
                            new List<IRenderChainUi>
                            {
                                new Hylian.SuperXbr.SuperXbrUi(),
                                new Nedi.NediScaler(),
                                new NNedi3.NNedi3Scaler(),
                                new Mpdn.OclNNedi3.OclNNedi3Scaler()
                            }.Select(x => x.ToPreset()))
                        .ToList(),
                    SelectedIndex = 0
                };
            }

            protected override ITextureFilter CreateFilter(ITextureFilter input)
            {
                var option = PrescalerGroup.SelectedOption;
                return option == null ? input : CreateFilter(input, input + option);
            }

            private ITextureFilter DownscaleAndDiff(ITextureFilter input, ITextureFilter original, TextureSize targetSize)
            {
                var HDownscaler = CompileShader("../SSimDownscaler/Scalers/Downscaler.hlsl", macroDefinitions: "axis = 0;")
                    .Configure(transform: s => new TextureSize(targetSize.Width, s.Height));
                var VDonwscaleAndDiff = CompileShader("./DownscaleAndDiff.hlsl", macroDefinitions: "axis = 1;")
                    .Configure(transform: s => new TextureSize(s.Width, targetSize.Height))
                    .Configure(format: TextureFormat.Float16);

                var hMean = HDownscaler.ApplyTo(input);
                var diff = VDonwscaleAndDiff.ApplyTo(hMean, original);

                return diff;
            }

            public ITextureFilter CreateFilter(ITextureFilter original, ITextureFilter initial)
            {
                ITextureFilter result;
                var HQDownscaler = (IScaler)new Bicubic(0.75f, false);

                // Calculate Sizes
                var inputSize = original.Output.Size;
                var targetSize = TargetSize();

                string macroDefinitions = "";
                if (Softness == 0.0f)
                    macroDefinitions += "SkipSoftening = 1;";
                if (Strength == 0.0f)
                    return initial;

                // Compile Shaders
                var Diff = CompileShader("Diff.hlsl")
                    .Configure(format: TextureFormat.Float16);

                var SuperRes = CompileShader("SuperResEx.hlsl", macroDefinitions: macroDefinitions)
                    .Configure(arguments: new[] { Strength, Softness });
                var FinalSuperRes = CompileShader("SuperResEx.hlsl", macroDefinitions: macroDefinitions + "FinalPass = 1;")
                    .Configure(arguments: new[] { Strength });

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
                    var filter = initial as IOffsetFilter;
                    if (filter != null)
                        filter.ForceOffsetCorrection();

                    result = initial.SetSize(targetSize).Apply(GammaToLinear);
                }
                else
                    result = original.Apply(GammaToLinear).Resize(targetSize);

                for (int i = 1; i <= Passes; i++)
                {
                    // Downscale and Subtract
                    ITextureFilter diff;
                    if (LegacyDownscaling)
                    {
                        var loRes = result.Resize(inputSize, downscaler: HQDownscaler);
                        diff = Diff.ApplyTo(loRes, original);
                    }
                    else
                    {   diff = DownscaleAndDiff(result, original, inputSize); }

                    /*SuperRes = (i != Passes ? SuperRes : FinalSuperRes).Configure(
                        arguments: new[] { Strength, (float)(Softness * Math.Pow(0.5, i-1))}
                    );*/

                    // Update result
                    result = (i != Passes ? SuperRes : FinalSuperRes).ApplyTo(result, diff);
                }

                return result;
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
