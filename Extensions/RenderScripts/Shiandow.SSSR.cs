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
using System.ComponentModel;
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
        public enum SSSRMode
        {
            [Description("Sharp")]
            Sharp = 0,
            [Description("Soft")]
            Soft = 1,
            [Description("Hybrid")]
            Hybrid = 2
        }

        public class SSSR : RenderChain
        {
            #region Settings

            public RenderScriptGroup PrescalerGroup { get; set; }

            public int Passes { get; set; }
            public bool LinearLight { get; set; }
            public float OverSharp { get; set; }
            public float Locality { get; set; }

            public SSSRMode Mode { get; set; }

            #endregion

            public Func<TextureSize> TargetSize; // Not saved

            public SSSR()
            {
                TargetSize = () => Renderer.TargetSize;

                Passes = 2;
                LinearLight = false;
                OverSharp = 0.0f;
                Locality = 4.0f;

                Mode = SSSRMode.Hybrid;

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
                    .Configure(
                        transform: s => new TextureSize(targetSize.Width, s.Height),
                        format: TextureFormat.Float16);
                var VDownscaler = CompileShader("./DownscaleII.hlsl", macroDefinitions: "axis = 1;")
                    .Configure(
                        transform: s => new TextureSize(s.Width, targetSize.Height),
                        format: TextureFormat.Float16);

                var hMean = HDownscaler.ApplyTo(input);
                var output = VDownscaler.ApplyTo(hMean, original);

                return output;
            }

            public ITextureFilter CreateFilter(ITextureFilter original, ITextureFilter initial)
            {
                ITextureFilter result;

                // Calculate Sizes
                var inputSize = original.Output.Size;
                var targetSize = TargetSize();

                // Compile Shaders
                var SharpDiff = CompileShader("Diff.hlsl", macroDefinitions: "MODE = 0;")
                    .Configure(format: TextureFormat.Float16);
                var Diff = CompileShader("Diff.hlsl", macroDefinitions: String.Format("MODE = {0};", Mode == SSSRMode.Sharp ? 0 : 1))
                    .Configure(format: TextureFormat.Float16);
                var SuperRes = CompileShader("SuperRes.hlsl");
                var FinalSuperRes = CompileShader("SuperRes.hlsl", macroDefinitions: "FinalPass = 1;" + (LinearLight ? "LinearLight = 1;" : "" ));
                var GammaToLinear = CompileShader("GammaToLinear.hlsl");

                SharpDiff["spread"] = 1 / Locality;
                SharpDiff["oversharp"] = OverSharp;

                Diff["spread"] = 1/Locality;
                Diff["oversharp"] = OverSharp;

                // Skip if downscaling
                if (targetSize.Width <= inputSize.Width || targetSize.Height <= inputSize.Height)
                    return original;

                // Initial scaling
                if (initial != original)
                {
                    // Always correct offset (if any)
                    var filter = initial as IOffsetFilter;
                    if (filter != null)
                        filter.ForceOffsetCorrection();

                    result = initial.SetSize(targetSize);
                    if (LinearLight)
                    {
                        original = original.Apply(GammaToLinear);
                        result = result.Apply(GammaToLinear);
                    }
                }
                else
                {
                    if (LinearLight)
                        original = original.Apply(GammaToLinear);
                    result = original.Resize(targetSize);
                }

                for (int i = 1; i <= Passes; i++)
                {
                    ITextureFilter diff;

                    // Downscale and Subtract
                    var loRes = Downscale(result, original, inputSize);
                    
                    // Calculate difference   
                    if (Mode == SSSRMode.Hybrid && i == 1)
                        diff = SharpDiff.ApplyTo(loRes, original);
                    else
                        diff = Diff.ApplyTo(loRes, original);

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
