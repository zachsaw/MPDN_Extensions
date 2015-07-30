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
using Mpdn.Extensions.RenderScripts.Shiandow.NNedi3.Filters;
using Mpdn.RenderScript;
using SharpDX;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Shiandow.NNedi3
    {
        public class NNedi3 : RenderChain
        {
            #region Settings

            public NNedi3()
            {
                Neurons1 = NNedi3Neurons.Neurons16;
                Neurons2 = NNedi3Neurons.Neurons16;
                CodePath = NNedi3Path.ScalarMad;
            }

            public NNedi3Neurons Neurons1 { get; set; }
            public NNedi3Neurons Neurons2 { get; set; }
            public NNedi3Path CodePath { get; set; }

            #endregion

            private static readonly int[] s_NeuronCount = {16, 32, 64, 128, 256};
            private static readonly string[] s_CodePath = {"A", "B", "C", "D", "E"};

            public override IFilter CreateFilter(IFilter input)
            {
                if (!Renderer.IsDx11Avail)
                {
                    Renderer.FallbackOccurred = true; // Warn user via player stats OSD
                    return input; // DX11 is not available; fallback
                }

                Func<TextureSize, TextureSize> Transformation = s => new TextureSize(2 * s.Height, s.Width);

                var NNEDI3p1 = LoadShader11(string.Format("NNEDI3_{0}_{1}.cso", s_NeuronCount[(int)Neurons1], s_CodePath[(int)CodePath]));
                var NNEDI3p2 = LoadShader11(string.Format("NNEDI3_{0}_{1}.cso", s_NeuronCount[(int)Neurons2], s_CodePath[(int)CodePath]));
                var Interleave = CompileShader("Interleave.hlsl").Configure(transform: Transformation);
                var Combine = CompileShader("Combine.hlsl").Configure(transform: Transformation);

                var sourceSize = input.OutputSize;
                if (!IsUpscalingFrom(sourceSize))
                    return input;

                var yuv = input.ConvertToYuv();

                var chroma = new ResizeFilter(yuv, new TextureSize(sourceSize.Width*2, sourceSize.Height*2),
                    TextureChannels.ChromaOnly, new Vector2(-0.25f, -0.25f), Renderer.ChromaUpscaler, Renderer.ChromaDownscaler);
                
                IFilter resultY, result;

                var pass1 = NNedi3Helpers.CreateFilter(NNEDI3p1, yuv, Neurons1, Neurons1 != Neurons2);
                resultY = new ShaderFilter(Interleave, yuv, pass1);
                var pass2 = NNedi3Helpers.CreateFilter(NNEDI3p2, resultY, Neurons2, Neurons1 != Neurons2);
                result = new ShaderFilter(Combine, resultY, pass2, chroma);

                return new ResizeFilter(result.ConvertToRgb(), result.OutputSize, new Vector2(0.5f, 0.5f),
                    Renderer.LumaUpscaler, Renderer.LumaDownscaler);
            }
        }

        public class NNedi3Scaler : RenderChainUi<NNedi3, NNedi3ConfigDialog>
        {
            protected override string ConfigFileName
            {
                get { return "Shiandow.NNedi3"; }
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
                        Guid = new Guid("B210A4E6-E3F9-4FEE-9840-5D6EDB0BFE05"),
                        Name = "NNEDI3",
                        Description = "NNEDI3 image doubler",
                        Copyright = "Adapted by Shiandow for MPDN"
                    };
                }
            }
        }
    }
}
