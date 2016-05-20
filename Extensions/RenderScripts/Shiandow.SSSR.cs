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
using System.IO;
using System.Linq;
using Mpdn.Extensions.CustomLinearScalers;
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.Framework.RenderChain;
using Mpdn.Extensions.RenderScripts.Mpdn.EwaScaler;
using Mpdn.RenderScript;
using Mpdn.RenderScript.Scaler;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Shiandow.SuperRes
    {
        public class SSSR : RenderChain
        {
            #region Settings

            public RenderScriptGroup PrescalerGroup { get; set; }

            public int Passes { get; set; }
            public float Strength { get; set; }
            public float Softness { get; set; }

            public bool LegacyDownscaling { get; set; }

            #endregion

            public Func<TextureSize> TargetSize; // Not saved

            public SSSR()
            {
                TargetSize = () => Renderer.TargetSize;

                Passes = 2;
                Strength = 1.0f;
                Softness = 0.0f;

                LegacyDownscaling = false;

                var EWASincJinc = new EwaScalerScaler
                {
                    Settings = new EwaScaler
                    {
                        Scaler = new SincJinc(),
                        TapCount = ScalerTaps.Six,
                        AntiRingingEnabled = false // Not needed
                    }
                }.ToPreset("EWA Sinc-Jinc");

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
                    Options = (new[] { EWASincJinc, fastSuperXbrUi})
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

                PrescalerGroup.Name = "SSSR Prescaler";
            }

            protected override ITextureFilter CreateFilter(ITextureFilter input)
            {
                var option = PrescalerGroup.SelectedOption;
                return option == null ? input : CreateFilter(input, input + option);
            }

            private ITextureFilter Downscale(ITextureFilter input, ITextureFilter original, TextureSize targetSize)
            {
                var HDownscaler = CompileShader("./Downscale.hlsl", macroDefinitions: "axis = 0;")
                    .Configure(transform: s => new TextureSize(targetSize.Width, s.Height));
                var VDonwscale = CompileShader("./DownscaleII.hlsl", macroDefinitions: "axis = 1;")
                    .Configure(transform: s => new TextureSize(s.Width, targetSize.Height))
                    .Configure(format: TextureFormat.Float16);

                var hMean = HDownscaler.ApplyTo(input);
                var output = VDonwscale.ApplyTo(hMean, original);

                return output;
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

                var SuperRes = CompileShader("SuperRes.hlsl", macroDefinitions: macroDefinitions)
                    .Configure(arguments: new[] { Strength, Softness });
                var FinalSuperRes = CompileShader("SuperRes.hlsl", macroDefinitions: macroDefinitions + "FinalPass = 1;")
                    .Configure(arguments: new[] { Strength });
                var GammaToLinear = CompileShader("./GammaToLinear.hlsl");

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

                    original = original.Apply(GammaToLinear);
                    result = initial.SetSize(targetSize).Apply(GammaToLinear);
                }
                else
                {
                    original = original.Apply(GammaToLinear);
                    result = original.Resize(targetSize);
                }

                for (int i = 1; i <= Passes; i++)
                {
                    // Downscale and Subtract
                    var loRes = Downscale(result, original, inputSize);
                    var diff = Diff.ApplyTo(loRes, original);

                    // Update result
                    result = (i != Passes ? SuperRes : FinalSuperRes).ApplyTo(result, diff, loRes);
                }

                return result;
            }
        }

        public class SSSRUi : RenderChainUi<SSSR, SSSRConfigDialog>
        {
            protected override string ConfigFileName
            {
                get { return "Shiandow.SSSR"; }
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
                        Guid = new Guid("162AF4FC-FDD4-11E5-9504-697BFDC8A409"),
                        Name = "SSSR",
                        Description = "SSSR image scaling",
                        Copyright = "SSSR by Shiandow",
                    };
                }
            }
        }
    }
}
