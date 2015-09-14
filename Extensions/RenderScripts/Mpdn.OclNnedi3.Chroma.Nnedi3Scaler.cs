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
        public class OclNNedi3Chroma : RenderChain, IChromaScaler
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

            public override string Active()
            {
                return string.Format("{0} {1}/{2}", base.Active(), s_NeuronCount[(int)Neurons1],
                    s_NeuronCount[(int)Neurons2]);
            }

            public override void Reset()
            {
                DisposeHelper.Dispose(ref m_Buffer1);
                DisposeHelper.Dispose(ref m_Buffer2);

                base.Reset();
            }

            protected override string ShaderPath
            {
                get { return "OCL_NNEDI3"; }
            }

            protected IKernel CompileKernel(bool u)
            {
                return CompileClKernel("nnedi3ocl.cl", "nnedi3",
                    string.Format("-cl-fast-relaxed-math -D {0}", u ? "CHROMA_U=1" : "CHROMA_V=1"));
            }

            protected override IFilter CreateFilter(IFilter input)
            {
                return this.MakeChromaFilter(input);
            }

            public IFilter CreateChromaFilter(IFilter lumaInput, IFilter chromaInput, TextureSize targetSize, Vector2 chromaOffset)
            {
                DisposeHelper.Dispose(ref m_Buffer1);
                DisposeHelper.Dispose(ref m_Buffer2);

                if (!Renderer.IsOpenClAvail || Renderer.RenderQuality.PerformanceMode())
                {
                    Renderer.FallbackOccurred = true; // Warn user via player stats OSD
                    return null; // OpenCL is not available; fallback
                }

                var lumaSize = lumaInput.OutputSize;
                var chromaSize = chromaInput.OutputSize;

                if (lumaSize.Width != 2*chromaSize.Width || lumaSize.Height != 2*chromaSize.Height)
                    return null; // Chroma shouldn't be doubled; fallback

                Func<TextureSize, TextureSize> transformWidth = s => new TextureSize(2 * s.Width, s.Height);
                Func<TextureSize, TextureSize> transformHeight = s => new TextureSize(s.Width, 2 * s.Height);

                var kernelU = CompileKernel(true);
                var kernelV = CompileKernel(false);
                var shaderUh = kernelU.Configure(transform: transformWidth);
                var shaderUv = kernelU.Configure(transform: transformHeight);
                var shaderVh = kernelV.Configure(transform: transformWidth);
                var shaderVv = kernelV.Configure(transform: transformHeight);

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

                var localWorkSizes = new[] { 8, 8 };
                var nnedi3Uh = new NNedi3HKernelFilter(shaderUh, m_Buffer1, neuronCount1,
                    new TextureSize(chromaInput.OutputSize.Width, chromaInput.OutputSize.Height),
                    localWorkSizes, chromaInput);
                var nnedi3Uv = new NNedi3VKernelFilter(shaderUv, m_Buffer2, neuronCount2, differentWeights,
                    new TextureSize(nnedi3Uh.OutputSize.Width, nnedi3Uh.OutputSize.Height),
                    localWorkSizes, nnedi3Uh);

                var nnedi3Vh = new NNedi3HKernelFilter(shaderVh, m_Buffer1, neuronCount1,
                    new TextureSize(chromaInput.OutputSize.Width, chromaInput.OutputSize.Height),
                    localWorkSizes, chromaInput);
                var nnedi3Vv = new NNedi3VKernelFilter(shaderVv, m_Buffer2, neuronCount2, differentWeights,
                    new TextureSize(nnedi3Vh.OutputSize.Width, nnedi3Vh.OutputSize.Height),
                    localWorkSizes, nnedi3Vh);

                Status = this.ChromaScalerStatus(lumaInput);

                return new MergeFilter(lumaInput, nnedi3Uv, nnedi3Vv).ConvertToRgb();
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
