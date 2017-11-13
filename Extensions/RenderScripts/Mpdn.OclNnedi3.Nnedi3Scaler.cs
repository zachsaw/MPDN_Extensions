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
using Mpdn.Extensions.Framework.RenderChain.Filters;
using Mpdn.Extensions.Framework.RenderChain.Shaders;
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
                var baseWidth = (float)localWorkSizes[0];
                var baseHeight = (float)localWorkSizes[1];

                var widthGroupCount = Math.Ceiling(width / (baseWidth * baseWidth));
                var heightGroupCount = Math.Ceiling(height / baseHeight);
                return new[] { (int)(baseWidth * widthGroupCount), (int)(baseHeight * heightGroupCount) };
            }
        }

        public class NNedi3Kernel : ClKernel
        {
            public int NeuronCount { get; set; }
            public bool ReloadWeights { get; set; }
            public bool Horizontal { get; set; }

            public IDisposable Buffer { get; set; }

            public bool ReuseHandle { get; set; }

            public NNedi3Kernel(IShaderDefinition<IKernel> definition, int[] localWorkSizes)
                : base(new CachedDefinition<IKernel>(definition), null, localWorkSizes)
            {
                NeuronCount = 16;
                ReloadWeights = true;
                Buffer = null;
                Horizontal = true;
            }
            
            public NNedi3Kernel(NNedi3Kernel config)
                : base(config)
            {
                NeuronCount = config.NeuronCount;
                ReloadWeights = config.ReloadWeights;
                Buffer = config.Buffer;
            }

            public override IShaderHandle GetHandle()
            {
                return new NNediKernelHandle(this, Definition, Horizontal);
            }
        }

        public class NNediKernelHandle : ClKernelHandle
        {
            private readonly bool m_ReloadWeights;
            private readonly int m_NeuronCount;
            private readonly bool m_Horizontal;

            private readonly IDisposable m_Buffer;

            public NNediKernelHandle(NNedi3Kernel parameters, IShaderDefinition<IKernel> definition, bool horizontal)
                : base(parameters, definition)
            {
                m_NeuronCount = parameters.NeuronCount;
                m_ReloadWeights = parameters.ReloadWeights;
                m_Horizontal = horizontal;

                m_Buffer = parameters.Buffer;
            }

            protected override void LoadArguments(IList<IBaseTexture> inputs, ITargetTexture output)
            {
                var inputTexture = (ITexture2D)inputs[0];

                GlobalWorkSizes = GlobalWorkSizesHelper.Get(inputTexture.Width, inputTexture.Height, LocalWorkSizes);

                // Use the 'temp' texture from first pass as input
                Shader.SetTempTextureArg(0, inputTexture); // srcImg
                Shader.SetOutputTextureArg(1, output); // dstImg
                if (m_ReloadWeights)
                {
                    Shader.SetBufferArg(2, m_Buffer); // weights
                    Shader.SetArg(3, m_NeuronCount); // nnst
                }
                Shader.SetArg(4, m_Horizontal ? inputTexture.Height : inputTexture.Width ); // SrcWidth
                Shader.SetArg(5, m_Horizontal ? inputTexture.Width  : inputTexture.Height); // SrcHeight
                Shader.SetArg(6, m_Horizontal ? 1 : 0); // SwapXy
            }

            protected override void Dispose(bool disposing)
            {
                base.Dispose(disposing);
                DisposeHelper.Dispose(m_Buffer);
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

            protected NNedi3Kernel GetKernel()
            {
                var localWorkSizes = new[] { 8, 8 };
                return new NNedi3Kernel(FromFile("nnedi3ocl.cl", entryPoint: "nnedi3", compilerOptions: "-cl-fast-relaxed-math"), localWorkSizes);
            }

            private IChromaScaler ChromaScaler
            {
                get
                {
                    return ChromaScalers.FirstOrDefault(s => s.Script.Descriptor.Guid == ChromaScalerGuid) ??
                           (IChromaScaler) new DefaultChromaScaler();
                }
            }

            protected override ITextureFilter CreateFilter(ITextureFilter input)
            {
                if (!Renderer.IsOpenClAvail || Renderer.RenderQuality.PerformanceMode())
                {
                    Renderer.FallbackOccurred = true; // Warn user via player stats OSD
                    return input; // OpenCL is not available, or UNORM8 textures used (not supported); fallback
                }

                Func<TextureSize, TextureSize> transformWidth = s => new TextureSize(2*s.Width, s.Height);
                Func<TextureSize, TextureSize> transformHeight = s => new TextureSize(s.Width, 2*s.Height);

                var neuronCount1 = s_NeuronCount[(int) Neurons1];
                var neuronCount2 = s_NeuronCount[(int) Neurons2];
                var weights1 = s_Weights[(int) Neurons1];
                var buffer1 = Renderer.CreateClBuffer(weights1);
                var buffer2 = buffer1;

                var differentWeights = neuronCount1 != neuronCount2;
                if (differentWeights)
                {
                    var weights2 = s_Weights[(int) Neurons2];
                    buffer2 = Renderer.CreateClBuffer(weights2);
                }

                var kernel = GetKernel(); // Note Kernel is reused between different filters
                var shaderH = new NNedi3Kernel(kernel) { Horizontal = true, Buffer = buffer1, NeuronCount = neuronCount1, Transform = transformWidth };
                var shaderV = new NNedi3Kernel(kernel) { Horizontal = false, Buffer = buffer2, NeuronCount = neuronCount2, Transform = transformHeight, ReloadWeights = differentWeights };

                var sourceSize = input.Size();
                if ((Renderer.TargetSize <= sourceSize).Any)
                    return input;

                var yuv = input.ConvertToYuv();

                var nnedi3H = shaderH.ApplyTo(yuv);
                var nnedi3V = shaderV.ApplyTo(nnedi3H);

                var result = ChromaScaler.ScaleChroma(
                    new CompositionFilter(nnedi3V, yuv, targetSize: nnedi3V.Size(), chromaOffset: new Vector2(-0.25f, -0.25f)));

                return result.Convolve(null, offset: new Vector2(0.5f, 0.5f));
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
