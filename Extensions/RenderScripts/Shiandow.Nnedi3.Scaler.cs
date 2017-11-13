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
using Mpdn.Extensions.Framework.RenderChain;
using Mpdn.Extensions.Framework.RenderChain.Filters;
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
                Structured = false;
                ChromaScalerGuid = Guid.Empty;
                ChromaScalers = new List<ChromaScalerPreset>();
            }

            public NNedi3Neurons Neurons1 { get; set; }
            public NNedi3Neurons Neurons2 { get; set; }
            public NNedi3Path CodePath { get; set; }
            public bool Structured { get; set; }
            public List<ChromaScalerPreset> ChromaScalers { get; set; }
            public Guid ChromaScalerGuid { get; set; }

            #endregion

            private static readonly int[] s_NeuronCount = {16, 32, 64, 128, 256};
            private static readonly string[] s_CodePath = {"A", "B", "C", "D", "E"};

            private IChromaScaler ChromaScaler
            {
                get { return ChromaScalers.FirstOrDefault(s => s.Script.Descriptor.Guid == ChromaScalerGuid) ?? (IChromaScaler)new DefaultChromaScaler(); }
            }

            public override string Description
            {
                get
                {
                    return string.Format("{0} {1}/{2}", base.Description, s_NeuronCount[(int) Neurons1],
                        s_NeuronCount[(int) Neurons2]);
                }
            }

            protected override ITextureFilter CreateFilter(ITextureFilter input)
            {
                if (!Renderer.IsDx11Avail)
                {
                    Renderer.FallbackOccurred = true; // Warn user via player stats OSD
                    return input; // DX11 is not available; fallback
                }

                Func<TextureSize, TextureSize> transform = s => new TextureSize(2 * s.Height, s.Width);

                var shaderPass1 = GetShader(Neurons1);
                var shaderPass2 = GetShader(Neurons2);
                var interleave = new Shader(FromFile("Interleave.hlsl")) { Transform = transform };

                if ((Renderer.TargetSize <= input.Size()).Any)
                    return input;

                var yuv = input.ConvertToYuv();

                var resultY = interleave.ApplyTo(yuv    , shaderPass1.ApplyTo(yuv));
                var luma    = interleave.ApplyTo(resultY, shaderPass2.ApplyTo(resultY));

                var result = ChromaScaler.ScaleChroma(
                    new CompositionFilter(luma, yuv, targetSize: luma.Size(), chromaOffset: new Vector2(-0.25f, -0.25f)));

                return result.Convolve(null, offset: new Vector2(0.5f, 0.5f));
            }

            private NNedi3Shader GetShader(NNedi3Neurons neurons)
            {
                var filename = string.Format("NNEDI3_{0}_{1}{2}.cso", 
                    s_NeuronCount[(int) neurons], 
                    s_CodePath[(int) CodePath],
                    Structured ? "_S" : string.Empty);
                return new NNedi3Shader(FromByteCode(filename))
                {
                    Neurons = neurons,
                    Structured = Structured
                };
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
                        Copyright = "Adapted by Shiandow and Zachs for MPDN"
                    };
                }
            }
        }
    }
}
