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
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.Framework.RenderChain;
using Mpdn.OpenCl;
using Mpdn.RenderScript;
using SharpDX;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Mpdn.OclNNedi3.Chroma
    {
        public class OclNNedi3Chroma : ChromaChain
        {
            #region Settings

            public OclNNedi3Chroma()
            {
                Neurons1 = OclNNedi3Neurons.Neurons16;
                Neurons2 = OclNNedi3Neurons.Neurons16;
            }

            public OclNNedi3Neurons Neurons1 { get; set; }
            public OclNNedi3Neurons Neurons2 { get; set; }

            #endregion

            private IDisposable m_Buffer1;
            private IDisposable m_Buffer2;

            private static readonly uint[][] s_Weights =
            {
                Weights.Weights16Neurons,
                Weights.Weights32Neurons,
                Weights.Weights64Neurons,
                Weights.Weights128Neurons,
                Weights.Weights256Neurons
            };
            private static readonly int[] s_NeuronCount = { 16, 32, 64, 128, 256 };

            public override string Description
            {
                get
                {
                    return string.Format("{0} {1}/{2}", base.Description, s_NeuronCount[(int) Neurons1],
                        s_NeuronCount[(int) Neurons2]);
                }
            }

            protected override string ShaderPath
            {
                get { return "OCL_NNEDI3"; }
            }

            protected NNedi3Kernel CompileKernel(bool u)
            {
                var localWorkSizes = new[] { 8, 8 };
                return new NNedi3Kernel(
                    FromString("nnedi3ocl.cl",
                        entryPoint: "nnedi3",
                        compilerOptions: string.Format("-cl-fast-relaxed-math -D {0}", u ? "CHROMA_U=1" : "CHROMA_V=1")),
                    localWorkSizes);
            }

            public override ITextureFilter ScaleChroma(ICompositionFilter composition)
            {
                DisposeHelper.Dispose(ref m_Buffer1);
                DisposeHelper.Dispose(ref m_Buffer2);

                if (!Renderer.IsOpenClAvail || Renderer.RenderQuality.PerformanceMode())
                {
                    Renderer.FallbackOccurred = true; // Warn user via player stats OSD
                    return composition; // OpenCL is not available; fallback
                }

                var lumaSize = composition.Luma.Size();
                var chromaSize = composition.Chroma.Size();

                if (lumaSize.Width != 2 * chromaSize.Width || lumaSize.Height != 2 * chromaSize.Height)
                    return composition; // Chroma shouldn't be doubled; fallback

                Func<TextureSize, TextureSize> transformWidth = s => new TextureSize(2 * s.Width, s.Height);
                Func<TextureSize, TextureSize> transformHeight = s => new TextureSize(s.Width, 2 * s.Height);

                var neuronCount1 = s_NeuronCount[(int)Neurons1];
                var neuronCount2 = s_NeuronCount[(int)Neurons2];

                var weights1 = s_Weights[(int)Neurons1];
                m_Buffer1 = Renderer.CreateClBuffer(weights1);
                var differentWeights = neuronCount1 != neuronCount2;
                if (differentWeights)
                {
                    var weights2 = s_Weights[(int)Neurons2];
                    m_Buffer2 = Renderer.CreateClBuffer(weights2);
                }

                var kernelU = CompileKernel(true);  // Note: compiled shader is shared between filters
                var kernelV = CompileKernel(false); // Note: compiled shader is shared between filters

                var shaderUh = new NNedi3Kernel(kernelU) { Horizontal = true , Buffer = m_Buffer1, NeuronCount = neuronCount1, Transform = transformWidth };
                var shaderUv = new NNedi3Kernel(kernelU) { Horizontal = false, Buffer = m_Buffer2, ReloadWeights = differentWeights, NeuronCount = neuronCount2, Transform = transformHeight};
                var shaderVh = new NNedi3Kernel(kernelV) { Horizontal = true , Buffer = m_Buffer1, NeuronCount = neuronCount1, Transform = transformWidth };
                var shaderVv = new NNedi3Kernel(kernelV) { Horizontal = false, Buffer = m_Buffer2, ReloadWeights = differentWeights, NeuronCount = neuronCount2, Transform = transformHeight };

                var nnedi3Uh = composition.Chroma.Apply(shaderUh);
                var nnedi3Uv = nnedi3Uh.Apply(shaderUv);

                var nnedi3Vh = composition.Chroma.Apply(shaderVh);
                var nnedi3Vv = nnedi3Vh.Apply(shaderVv);

                return composition.Luma.MergeWith(nnedi3Uv, nnedi3Vv).ConvertToRgb();
            }
        }

        public class OclNNedi3ChromaScaler : RenderChainUi<OclNNedi3Chroma, OclNNedi3ChromaConfigDialog>
        {
            protected override string ConfigFileName
            {
                get { return "Mpdn.OclNNedi3Chroma"; }
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
                        Guid = new Guid("2B4780B6-7AA7-478B-BE30-36EB5A92D296"),
                        Name = "OpenCL NNEDI3 Chroma Doubler",
                        Description = "OpenCL NNEDI3 chroma doubler",
                        Copyright = "Adapted by Zachs for MPDN"
                    };
                }
            }
        }
    }
}
