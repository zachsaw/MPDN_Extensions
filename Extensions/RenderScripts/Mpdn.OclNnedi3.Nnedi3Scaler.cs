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
using System.Collections.Generic;
using System.Linq;
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.Framework.RenderChain;
using Mpdn.OpenCl;
using Mpdn.RenderScript;
using SharpDX;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Mpdn.OclNNedi3
    {
        public enum OclNNedi3Neurons
        {
            Neurons16 = 0,
            Neurons32 = 1,
            Neurons64 = 2,
            Neurons128 = 3,
            Neurons256 = 4
        }

        public static class GlobalWorkSizesHelper
        {
            public static int[] Get(int width, int height, int[] localWorkSizes)
            {
                var baseWidth = (float) localWorkSizes[0];
                var baseHeight = (float) localWorkSizes[1];

                var widthGroupCount = Math.Ceiling(width/(baseWidth*baseWidth));
                var heightGroupCount = Math.Ceiling(height/baseHeight);
                return new[] {(int) (baseWidth*widthGroupCount), (int) (baseHeight*heightGroupCount)};
            }
        }

        public class NNedi3HKernelFilter : ClKernelFilter
        {
            private readonly IDisposable m_Buffer;
            private readonly int m_NeuronCount;
            private readonly TextureSize m_TextureSize;

            public NNedi3HKernelFilter(ShaderFilterSettings<IKernel> settings, IDisposable buffer, int neuronCount, TextureSize textureSize, int[] localWorkSizes,
                IFilter<IBaseTexture> inputFilter)
                : base(settings, GlobalWorkSizesHelper.Get(textureSize.Height, textureSize.Width, localWorkSizes), localWorkSizes, inputFilter)
            {
                m_Buffer = buffer;
                m_NeuronCount = neuronCount;
                m_TextureSize = textureSize;
            }

            protected override void LoadInputs(IList<IBaseTexture> inputs)
            {
                Shader.SetInputTextureArg(0, (ITexture2D)inputs[0]); // srcImg
                                                                     // First pass doesn't require result to be copied back to Direct3D - so we use a 'temp' texture
                Shader.SetTempTextureArg(1, OutputTarget); // dstImg
                Shader.SetBufferArg(2, m_Buffer); // weights
                Shader.SetArg(3, m_NeuronCount); // nnst
                Shader.SetArg(4, m_TextureSize.Height); // SrcWidth
                Shader.SetArg(5, m_TextureSize.Width); // SrcHeight
                Shader.SetArg(6, 1); // SwapXy
            }
        }

        public class NNedi3VKernelFilter : ClKernelFilter
        {
            private readonly bool m_ReloadWeights;
            private readonly IDisposable m_Buffer;
            private readonly int m_NeuronCount;
            private readonly TextureSize m_TextureSize;

            public NNedi3VKernelFilter(ShaderFilterSettings<IKernel> settings, IDisposable buffer, int neuronCount, bool reloadWeights, TextureSize textureSize, int[] localWorkSizes,
                IFilter<IBaseTexture> inputFilter)
                : base(settings, GlobalWorkSizesHelper.Get(textureSize.Width, textureSize.Height, localWorkSizes), localWorkSizes, inputFilter)
            {
                m_Buffer = buffer;
                m_NeuronCount = neuronCount;
                m_ReloadWeights = reloadWeights;
                m_TextureSize = textureSize;
            }

            protected override void LoadInputs(IList<IBaseTexture> inputs)
            {
                // Use the 'temp' texture from first pass as input
                Shader.SetTempTextureArg(0, (ITexture2D)inputs[0]); // srcImg
                Shader.SetOutputTextureArg(1, OutputTarget); // dstImg
                if (m_ReloadWeights)
                {
                    Shader.SetBufferArg(2, m_Buffer); // weights
                    Shader.SetArg(3, m_NeuronCount); // nnst
                }
                Shader.SetArg(4, m_TextureSize.Width); // SrcWidth
                Shader.SetArg(5, m_TextureSize.Height); // SrcHeight
                Shader.SetArg(6, 0); // SwapXy
            }
        }

        public class OclNNedi3 : RenderChain
        {
            #region Settings

            public OclNNedi3()
            {
                Neurons1 = OclNNedi3Neurons.Neurons16;
                Neurons2 = OclNNedi3Neurons.Neurons16;
                ChromaScalers = new List<ChromaScalerPreset>();
                ChromaScalerGuid = Guid.Empty;
            }

            public OclNNedi3Neurons Neurons1 { get; set; }
            public OclNNedi3Neurons Neurons2 { get; set; }
            public List<ChromaScalerPreset> ChromaScalers { get; set; }
            public Guid ChromaScalerGuid { get; set; }

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

            protected IKernel CompileKernel()
            {
                return CompileClKernel("nnedi3ocl.cl", "nnedi3", "-cl-fast-relaxed-math");
            }

            private IChromaScaler ChromaScaler
            {
                get
                {
                    return ChromaScalers.FirstOrDefault(s => s.Script.Descriptor.Guid == ChromaScalerGuid) ??
                           (IChromaScaler)new DefaultChromaScaler();
                }
            }

            public override IFilter CreateFilter(IFilter input)
            {
                DisposeHelper.Dispose(ref m_Buffer1);
                DisposeHelper.Dispose(ref m_Buffer2);

                if (!Renderer.IsOpenClAvail || Renderer.RenderQuality.PerformanceMode())
                {
                    Renderer.FallbackOccurred = true; // Warn user via player stats OSD
                    return input; // OpenCL is not available, or UNORM8 textures used (not supported); fallback
                }

                Func<TextureSize, TextureSize> transformWidth = s => new TextureSize(2*s.Width, s.Height);
                Func<TextureSize, TextureSize> transformHeight = s => new TextureSize(s.Width, 2*s.Height);

                var kernel = CompileKernel();
                var shaderH = kernel.Configure(transform: transformWidth);
                var shaderV = kernel.Configure(transform: transformHeight);

                var neuronCount1 = s_NeuronCount[(int) Neurons1];
                var neuronCount2 = s_NeuronCount[(int) Neurons2];
                var weights1 = s_Weights[(int) Neurons1];
                m_Buffer1 = Renderer.CreateClBuffer(weights1);
                var differentWeights = neuronCount1 != neuronCount2;
                if (differentWeights)
                {
                    var weights2 = s_Weights[(int) Neurons2];
                    m_Buffer2 = Renderer.CreateClBuffer(weights2);
                }

                var sourceSize = input.OutputSize;
                if (!IsUpscalingFrom(sourceSize))
                    return input;

                var yuv = input.ConvertToYuv();

                var localWorkSizes = new[] {8, 8};
                var nnedi3H = new NNedi3HKernelFilter(shaderH, m_Buffer1, neuronCount1,
                    new TextureSize(yuv.OutputSize.Width, yuv.OutputSize.Height), 
                    localWorkSizes, yuv);
                var nnedi3V = new NNedi3VKernelFilter(shaderV, m_Buffer2, neuronCount2, differentWeights,
                    new TextureSize(nnedi3H.OutputSize.Width, nnedi3H.OutputSize.Height), 
                    localWorkSizes, nnedi3H);

                var result = ChromaScaler.CreateChromaFilter(nnedi3V, yuv, new Vector2(-0.25f, -0.25f));

                return new ResizeFilter(result, result.OutputSize, new Vector2(0.5f, 0.5f),
                    Renderer.LumaUpscaler, Renderer.LumaDownscaler);
            }
        }

        public class OclNNedi3Scaler : RenderChainUi<OclNNedi3, OclNNedi3ConfigDialog>
        {
            protected override string ConfigFileName
            {
                get { return "Mpdn.OclNNedi3"; }
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
                        Guid = new Guid("600C4E21-8C25-41E5-AD15-E58B86E2BA3B"),
                        Name = "OpenCL NNEDI3",
                        Description = "OpenCL NNEDI3 image doubler",
                        Copyright = "Adapted by Zachs for MPDN"
                    };
                }
            }
        }
    }
}
