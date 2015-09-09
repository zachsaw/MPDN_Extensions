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
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.Framework.RenderChain;
using Mpdn.Extensions.RenderScripts.Shiandow.NNedi3.Filters;
using Mpdn.RenderScript;
using SharpDX;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Shiandow.NNedi3.Chroma
    {
        public class NNedi3Chroma : RenderChain, IChromaScaler
        {
            #region Settings

            public NNedi3Chroma()
            {
                Neurons1 = NNedi3Neurons.Neurons16;
                Neurons2 = NNedi3Neurons.Neurons16;
                CodePath = NNedi3Path.ScalarMad;
                Structured = false;
            }

            public NNedi3Neurons Neurons1 { get; set; }
            public NNedi3Neurons Neurons2 { get; set; }
            public NNedi3Path CodePath { get; set; }
            public bool Structured { get; set; }

            #endregion

            private static readonly int[] s_NeuronCount = {16, 32, 64, 128, 256};
            private static readonly string[] s_CodePath = {"A", "B", "C", "D", "E"};

            private Nnedi3Filter m_UFilter1;
            private Nnedi3Filter m_UFilter2;
            private Nnedi3Filter m_VFilter1;
            private Nnedi3Filter m_VFilter2;

            public override void Reset()
            {
                base.Reset();
                Cleanup();
            }

            private void Cleanup()
            {
                DisposeHelper.Dispose(ref m_UFilter1);
                DisposeHelper.Dispose(ref m_UFilter2);
                DisposeHelper.Dispose(ref m_VFilter1);
                DisposeHelper.Dispose(ref m_VFilter2);
            }

            protected override string ShaderPath
            {
                get { return typeof (NNedi3).Name; }
            }

            public override IFilter CreateFilter(IFilter input)
            {
                var chromaFilter = input as ChromaFilter;
                if (chromaFilter != null)
                    return chromaFilter.MakeNew(this);

                return input;
            }

            private string GetShaderFileName(NNedi3Neurons neurons, bool u)
            {
                return string.Format("NNEDI3_{3}_{0}_{1}{2}.cso", s_NeuronCount[(int) neurons], s_CodePath[(int) CodePath],
                    Structured ? "_S" : string.Empty, u ? "u" : "v");
            }

            public IFilter CreateChromaFilter(IFilter lumaInput, IFilter chromaInput, TextureSize targetSize, Vector2 chromaOffset)
            {
                Cleanup();

                if (!Renderer.IsDx11Avail)
                {
                    Renderer.FallbackOccurred = true; // Warn user via player stats OSD
                    return new ChromaFilter(lumaInput, chromaInput, null, targetSize, chromaOffset); // DX11 is not available; fallback
                }

                var lumaSize = lumaInput.OutputSize;
                var chromaSize = chromaInput.OutputSize;
                
                if (lumaSize.Width != 2*chromaSize.Width || lumaSize.Height != 2*chromaSize.Height)
                    return new ChromaFilter(lumaInput, chromaInput, null, targetSize, chromaOffset); // Chroma shouldn't be doubled; fallback

                Func<TextureSize, TextureSize> transform = s => new TextureSize(2 * s.Height, s.Width);

                var shaderUPass1 = LoadShader11(GetShaderFileName(Neurons1, true));
                var shaderUPass2 = LoadShader11(GetShaderFileName(Neurons2, true));
                var shaderVPass1 = LoadShader11(GetShaderFileName(Neurons1, false));
                var shaderVPass2 = LoadShader11(GetShaderFileName(Neurons2, false));
                var interleaveU = CompileShader("Interleave.hlsl", macroDefinitions: "CHROMA_U=1").Configure(transform: transform);
                var interleaveV = CompileShader("Interleave.hlsl", macroDefinitions: "CHROMA_V=1").Configure(transform: transform);

                m_UFilter1 = NNedi3Helpers.CreateFilter(shaderUPass1, chromaInput, Neurons1, Structured);
                var resultU = new ShaderFilter(interleaveU, chromaInput, m_UFilter1);
                m_UFilter2 = NNedi3Helpers.CreateFilter(shaderUPass2, resultU, Neurons2, Structured);
                var u = new ShaderFilter(interleaveU, resultU, m_UFilter2);

                m_VFilter1 = NNedi3Helpers.CreateFilter(shaderVPass1, chromaInput, Neurons1, Structured);
                var resultV = new ShaderFilter(interleaveV, chromaInput, m_VFilter1);
                m_VFilter2 = NNedi3Helpers.CreateFilter(shaderVPass2, resultV, Neurons2, Structured);
                var v = new ShaderFilter(interleaveV, resultV, m_VFilter2);

                return new MergeFilter(lumaInput, u, v).ConvertToRgb();
            }
        }

        public class NNedi3ChromaScaler : RenderChainUi<NNedi3Chroma, NNedi3ChromaConfigDialog>
        {
            protected override string ConfigFileName
            {
                get { return "Shiandow.NNedi3Chroma"; }
            }

            public override string Category
            {
                get { return "Chroma Scaling"; }
            }

            public override ExtensionUiDescriptor Descriptor
            {
                get
                {
                    return new ExtensionUiDescriptor
                    {
                        Guid = new Guid("994C176F-AB9F-47E0-81FE-DC20609A40C2"),
                        Name = "NNEDI3 Chroma Doubler",
                        Description = "NNEDI3 chroma doubler",
                        Copyright = "Adapted by Shiandow and Zachs for MPDN"
                    };
                }
            }
        }
    }
}
