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
// 
using System;
using Mpdn.RenderScript.Shiandow.NNedi3Filters;
using SharpDX;

namespace Mpdn.RenderScript
{
    namespace Shiandow.NNedi3
    {
        public class NNedi3 : RenderChain
        {
            #region Settings

            public NNedi3()
            {
                Neurons = NNedi3Neurons.Neurons16;
            }

            public NNedi3Neurons Neurons { get; set; }

            #endregion

            private static readonly int[] s_NeuronCount = { 16, 32, 64, 128, 256 };

            public override IFilter CreateFilter(IResizeableFilter sourceFilter)
            {
                if (!Renderer.IsDx11Avail)
                    return sourceFilter; // DX11 is not available; fallback

                Func<TextureSize, TextureSize> Transformation = s => new TextureSize(2 * s.Height, s.Width);

                var NNEDI3 = LoadShader11(string.Format("NNEDI3_{0}.cso", s_NeuronCount[(int) Neurons]));
                var Interleave = CompileShader("Interleave.hlsl").Configure(transform: Transformation);
                var Combine = CompileShader("Combine.hlsl");

                var sourceSize = sourceFilter.OutputSize;
                if (!IsUpscalingFrom(sourceSize))
                    return sourceFilter;

                IFilter input = sourceFilter.ConvertToYuv();

                // Note: This is not optimal as it scales all 3 channels when we only need 2
                // TODO: RenderScript.Scale() to have a new argument (Channels.Luma, Channels.Chroma, Channels.All)
                var chroma = new ResizeFilter(input, new TextureSize(sourceSize.Width*2, sourceSize.Height*2),
                    new Vector2(-0.5f, -0.5f), Renderer.ChromaUpscaler);

                IFilter resultY;

                var pass1 = NNedi3Helpers.CreateFilter(NNEDI3, input, Neurons);
                resultY = new ShaderFilter(Interleave, input, pass1);
                var pass2 = NNedi3Helpers.CreateFilter(NNEDI3, resultY, Neurons);
                resultY = new ShaderFilter(Interleave, resultY, pass2);

                var result = new ShaderFilter(Combine, resultY, chroma);
                return result.ConvertToRgb();
            }
        }

        public class NNedi3ScalerTest : RenderChainUi<NNedi3>
        {
            protected override string ConfigFileName
            {
                get { return "Shiandow.NNedi3"; }
            }

            public override ExtensionUiDescriptor Descriptor
            {
                get
                {
                    return new ExtensionUiDescriptor
                    {
                        Guid = new Guid("B210A4E6-E3F9-4FEE-9840-5D6EDB0BFE05"),
                        Name = "NNedi3",
                        Description = "Shader adaptation of the NNedi3 algorithm",
                        Copyright = "<Various>",
                    };
                }
            }
        }
    }
}
