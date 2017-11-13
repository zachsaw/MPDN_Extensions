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
using Mpdn.Extensions.Framework.RenderChain;
using Mpdn.RenderScript;
using SharpDX;
using Mpdn.Extensions.RenderScripts.Mpdn.EwaScaler;
using Mpdn.Extensions.CustomLinearScalers;
using Mpdn.Extensions.Framework;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Shiandow.SuperRes
    {
        public class ProximalPrimalDual : RenderChain
        {
            #region Settings

            public int Passes { get; set; }
            public float Strength { get; set; }
            public float Softness { get; set; }

            #endregion

            public ProximalPrimalDual()
            {
                Passes = 4;
                Strength = 20.0f;
                Softness = 10.0f;
            }

            private ITextureFilter DownscaleAndDiff(ITextureFilter input, ITextureFilter previousInput, ITextureFilter original, ref ITextureFilter previousDiff)
            {
                var targetSize = original.Size();

                var HDownscaler = new Shader(FromFile("Downscale.hlsl", compilerOptions: "axis = 0;"))
                    {  Transform = s => new TextureSize(targetSize.Width, s.Height) };

                var VDownscaleAndDiff = new Shader(FromFile("DownscaleAndDiff.hlsl", compilerOptions: "axis = 1;" + (previousDiff != null ? "" : "FIRST_PASS = 1;")))
                {
                    Transform = s => new TextureSize(s.Width, targetSize.Height),
                    Format = TextureFormat.Float16
                };

                var LanczosIteration = new Shader(FromFile("LanczosIteration.hlsl"))//, macroDefinitions: previousDiff != null ? "" : "FIRST_PASS = 1;"))
                    { Format = TextureFormat.Float16 };

                LanczosIteration["sizeResult"] = new Vector4(input.Size().Width, input.Size().Height, 1 / (float)input.Size().Width, 1 / (float)input.Size().Height);

                var hMean = HDownscaler.ApplyTo(input, previousInput);

                var diff = VDownscaleAndDiff.ApplyTo(hMean, original, previousDiff ?? original);

                previousDiff = diff;

                return diff;// LanczosIteration.ApplyTo(diff);
            }

            protected override ITextureFilter CreateFilter(ITextureFilter input)
            {
                // Calculate Sizes
                var inputSize = input.Size();
                var targetSize = Renderer.TargetSize;

                // Skip if downscaling
                if ((targetSize <= inputSize).Any)
                    return input;

                Passes = 6;

                var ProximalPrimalDual = new Shader(FromFile("ProximalPrimalDual.hlsl"))
                {
                    Arguments = new[] { Strength, Softness }
                };

                var UpdateDelta = new Shader(FromFile("UpdateDelta.hlsl"))
                {
                    Format = TextureFormat.Float16
                };

                var InitDelta = new Shader(UpdateDelta)
                {
                    Definition = FromFile("UpdateDelta.hlsl", compilerOptions: "FIRST_PASS = 1;")
                };

                var GammaToLinear = new Shader(FromFile("GammaToLinear.hlsl"));

                var EWASincJinc = new EwaScalerScaler
                {
                    Settings = new EwaScaler
                    {
                        Scaler = new SincJinc(),
                        TapCount = ScalerTaps.Six,
                        AntiRingingEnabled = true,
                        AntiRingingStrength = 1.0f
                    }
                }.ToPreset("EWA Sinc-Jinc");

                var original = GammaToLinear.ApplyTo(input);
                ITextureFilter result = original + EWASincJinc; //original.Resize(targetSize, upscaler: new Bicubic(0.66f, true), tagged: true);//
                ITextureFilter previousResult = result;
                ITextureFilter previousDiff = null;
                ITextureFilter delta = null;

                // min |Phi(result)| where Downscale(result) = input
                for (int i = 1; i <= Passes; i++)
                {
                    // Downscale and Subtract
                    var diff = DownscaleAndDiff(result, previousResult, original, ref previousDiff);

                    if (delta != null)
                        delta = UpdateDelta.ApplyTo(result, previousResult, delta); 
                    else
                        delta = InitDelta.ApplyTo(result, previousResult);
                    
                    // Update result
                    ProximalPrimalDual["iteration"] = i;

                    previousResult = result;
                    result = ProximalPrimalDual.ApplyTo(result, diff, delta, original);
                }

                return result;
            }
        }

        public class ProximalPrimalDualUi : RenderChainUi<ProximalPrimalDual>
        {
            protected override string ConfigFileName
            {
                get { return "Shiandow.ProximalPrimalDual"; }
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
                        Guid = new Guid("a1ba045f-c181-40f6-b521-2b992db6232c"),
                        Name = "Proximal Primal Dual Scaling",
                        Description = "Proximal Primal Dual",
                        Copyright = "Shiandow (2017)",
                    };
                }
            }
        }
    }
}
