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
using Mpdn.RenderScript;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Example
    {
        public class Lut3D : RenderChain
        {
            private ISourceTexture3D m_Texture3D;

            protected override string ShaderPath
            {
                get { return "Examples"; }
            }

            public override IFilter CreateFilter(IFilter sourceFilter)
            {
                var shader = CompileShader("Lut3D.hlsl").Configure(linearSampling: true);
                return new ShaderFilter(shader, sourceFilter, new TextureSourceFilter<ISourceTexture3D>(m_Texture3D));
            }

            public override void Initialize()
            {
                Create3DTexture();

                base.Initialize();
            }

            public override void Reset()
            {
                DiscardTextures();

                base.Reset();
            }

            private void DiscardTextures()
            {
                DisposeHelper.Dispose(ref m_Texture3D);
            }

            private void Create3DTexture()
            {
                if (m_Texture3D != null)
                    return;

                const int cubeSize = 256;

                const int width = cubeSize;
                const int height = cubeSize;
                const int depth = cubeSize;
                m_Texture3D = Renderer.CreateTexture3D(width, height, depth, TextureFormat.Unorm16);
                Renderer.UpdateTexture3D(m_Texture3D, Create3DLut(width, height, depth));
            }

            private static ushort[,,] Create3DLut(int width, int height, int depth)
            {
                // Create a color-swap 3D LUT (r ==> b, g ==> r, b ==> g)
                // Note: This method is very slow (it's called once on init only though),
                //       but in real-life scenario, you'd be loading it from a 3dlut file 
                //       instead of generating it on the fly
                var lut = new ushort[depth, height, width*4];
                for (int b = 0; b < depth; b++)
                {
                    for (int g = 0; g < height; g++)
                    {
                        for (int r = 0; r < width; r++)
                        {
                            lut[b, g, r*4 + 0] = (ushort) ((b/(float) (width -1)) * ushort.MaxValue); // R channel, swap it with B
                            lut[b, g, r*4 + 1] = (ushort) ((r/(float) (height-1)) * ushort.MaxValue); // G channel, swap it with R
                            lut[b, g, r*4 + 2] = (ushort) ((g/(float) (depth -1)) * ushort.MaxValue); // B channel, swap it with G
                            lut[b, g, r*4 + 3] = ushort.MaxValue; // Alpha
                        }
                    }
                }
                return lut;
            }
        }

        public class Lut3DExample : RenderChainUi<Lut3D>
        {
            public override ExtensionUiDescriptor Descriptor
            {
                get
                {
                    return new ExtensionUiDescriptor
                    {
                        Name = "3DLut Example",
                        Description = "(Example) A color swap 3D LUT",
                        Guid = new Guid("0C44DF3F-5FAA-43B0-A530-9D55E6CE9D1C"),
                        Copyright = "" // Optional field
                    };
                }
            }

            public override string Category
            {
                get { return "Example"; }
            }
        }
    }
}
