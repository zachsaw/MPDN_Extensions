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
using Mpdn.RenderScript;
using TransformFunc = System.Func<System.Drawing.Size, System.Drawing.Size>;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Mpdn.Jinc2D
    {
        public class Jinc2D : RenderChain
        {
            private ISourceTexture[] m_Weights;

            #region Settings

            public ScalerTaps TapCount { get; set; }

            #endregion

            protected override string ShaderPath
            {
                get { return "Jinc2D"; }
            }

            public override IFilter CreateFilter(IFilter input)
            {
                var sourceSize = input.OutputSize;
                if (!IsUpscalingFrom(sourceSize))
                    return input;

                CreateWeights();

                var shader = CompileShader("Jinc2D.hlsl")
                    .Configure(linearSampling: false, transform: size => Renderer.TargetSize);

                return new ShaderFilter(shader, input,
                    new TextureSourceFilter<ISourceTexture>(m_Weights[0]),
                    new TextureSourceFilter<ISourceTexture>(m_Weights[1]),
                    new TextureSourceFilter<ISourceTexture>(m_Weights[2]),
                    new TextureSourceFilter<ISourceTexture>(m_Weights[3]));
            }

            public override void Reset()
            {
                DiscardTextures();

                base.Reset();
            }

            private void DiscardTextures()
            {
                if (m_Weights == null)
                    return;

                foreach (var w in m_Weights)
                {
                    DisposeHelper.Dispose(w);
                }
                m_Weights = null;
            }

            private static float GetWeight(float x)
            {
                const double jinc2WindowSinc = 0.44;
                const double jinc2Sinc = 0.82;

                const double wa = jinc2WindowSinc*Math.PI;
                const double wb = jinc2Sinc*Math.PI;

                // Approximation of 2 lobed windowed jinc from libretro
                // TODO: Replace with actual windowed jinc function
                return x < 1e-8 ? (float) (wa*wb) : (float) (Math.Sin(x*wa)*Math.Sin(x*wb)/(x*x));
            }

            private static float CalculateLength(float point1, float point2)
            {
                return (float) Math.Sqrt(point1*point1 + point2*point2);
            }

            private void CreateWeights()
            {
                if (m_Weights != null)
                    return;

                const int dataPoints = 20; // ~2.5% error in high contrast edges
                // Note: 
                //    Increase by steps of 4 if you want better quality (less error vs mathematical model)
                //    At 32 data points, you'll reduce the error to ~2% but it's probably not worth the GPU load overhead

                // For 2 lobed Jinc, we need 4 2D LUTs
                const int tapCount = 4;
                m_Weights = new ISourceTexture[tapCount];
                var data = new float[dataPoints, dataPoints*4];
                for (int z = 0; z < tapCount; z++)
                {
                    for (int y = 0; y < dataPoints; y++)
                    {
                        for (int x = 0; x < dataPoints; x++)
                        {
                            var offsetX = x/(float) dataPoints;
                            var offsetY = y/(float) dataPoints;
                            const int topLeft = -tapCount/2 + 1;
                            data[y, x*4 + 0] = GetWeight(CalculateLength(topLeft + 0 - offsetX, topLeft + z - offsetY));
                            data[y, x*4 + 1] = GetWeight(CalculateLength(topLeft + 1 - offsetX, topLeft + z - offsetY));
                            data[y, x*4 + 2] = GetWeight(CalculateLength(topLeft + 2 - offsetX, topLeft + z - offsetY));
                            data[y, x*4 + 3] = GetWeight(CalculateLength(topLeft + 3 - offsetX, topLeft + z - offsetY));
                        }
                    }
                    m_Weights[z] = Renderer.CreateTexture(dataPoints, dataPoints, TextureFormat.Float32);
                    Renderer.UpdateTexture(m_Weights[z], data);
                }
            }
        }

        public class ChromaScaler : RenderChainUi<Jinc2D, Jinc2DConfigDialog>
        {
            protected override string ConfigFileName
            {
                get { return "Mpdn.Jinc2D"; }
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
                        Guid = new Guid("317F3D2F-8A0C-4FA3-AF71-41BD4DDC6FC8"),
                        Name = "Jinc2D",
                        Description = "Jinc (cylindrical) image upscaler"
                    };
                }
            }
        }
    }
}
