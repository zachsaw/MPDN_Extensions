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

// See also: Perceptually Based Downscaling of Images, by Oztireli, A. Cengiz and Gross, Markus, 10.1145/2766891, https://graphics.ethz.ch/~cengizo/Files/Sig15PerceptualDownscaling.pdf

using System;
using Mpdn.Extensions.Framework.RenderChain;
using Mpdn.RenderScript;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Shiandow.SSimDownscaling
    {
        public class SSimDownscaler : RenderChain
        {
            #region Settings

            public bool SoftDownscaling { get; set; }
            public float Strength { get; set; }

            public SSimDownscaler()
            {
                SoftDownscaling = false;
                Strength = 0.5f;
            }

            #endregion

            private void DownscaleAndCalcVar(ITextureFilter input, TextureSize targetSize, out ITextureFilter mean, out ITextureFilter var)
            {
                var HDownscaler = CompileShader(SoftDownscaling ? "SoftDownscaler.hlsl" : "./Scalers/Downscaler.hlsl", macroDefinitions: "axis = 0;")
                    .Configure(transform: s => new TextureSize(targetSize.Width, s.Height), format: input.Output.Format);
                var VDownscaler = CompileShader(SoftDownscaling ? "SoftDownscaler.hlsl" : "./Scalers/Downscaler.hlsl", macroDefinitions: "axis = 1;")
                    .Configure(transform: s => new TextureSize(s.Width, targetSize.Height), format: input.Output.Format);
                var HVar = CompileShader("DownscaledVarI.hlsl", macroDefinitions: "axis = 0;")
                    .Configure(transform: s => new TextureSize(targetSize.Width, s.Height), format: input.Output.Format);
                var VVar = CompileShader("DownscaledVarII.hlsl", macroDefinitions: "axis = 1;")
                    .Configure(transform: s => new TextureSize(s.Width, targetSize.Height), format: input.Output.Format);

                var hMean = HDownscaler.ApplyTo(input);
                mean = VDownscaler.ApplyTo(hMean);

                var hVariance = HVar.ApplyTo(input, hMean);
                var = VVar.ApplyTo(hVariance, hMean, mean);
            }

            private void ConvolveAndCalcR(ITextureFilter input, ITextureFilter sh, out ITextureFilter mean, out ITextureFilter r)
            {
                var Convolver = CompileShader("SinglePassConvolver.hlsl").Configure(format: input.Output.Format);
                var CalcR = CompileShader("CalcR.hlsl").Configure(format: TextureFormat.Float16);

                mean = input.Apply(Convolver);
                r = CalcR.ApplyTo(input, mean, sh);
            }

            protected override ITextureFilter CreateFilter(ITextureFilter input)
            {
                ITextureFilter H = input, Sh, L, M, R;
                var targetSize = Renderer.TargetSize;

                if ((Renderer.TargetSize >= input.Size()).Any)
                    return input;

                var Calc = CompileShader("calc.hlsl");
                Calc["strength"] = Strength;

                DownscaleAndCalcVar(H, targetSize, out L, out Sh);
                ConvolveAndCalcR(L, Sh, out M, out R);

                return Calc.ApplyTo(L, M, R);
            }
        }
        
        public class SSimDownscalerUi : RenderChainUi<SSimDownscaler, SSimDownscalingConfigDialog>
        {
            protected override string ConfigFileName
            {
                get { return "SSIM.Downscaler"; }
            }

            public override string Category
            {
                get { return "Downscaling"; }
            }

            public override ExtensionUiDescriptor Descriptor
            {
                get
                {
                    return new ExtensionUiDescriptor
                    {
                        Guid = new Guid("ED1BD188-BA46-11E5-BADB-8BEA19563991"),
                        Name = "SSIM downscaler",
                        Description = "Structural Similarity based Downscaling",
                        Copyright = "Shiandow",
                    };
                }
            }
        }
    }
}
