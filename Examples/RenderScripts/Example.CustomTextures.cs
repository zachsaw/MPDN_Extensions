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
using System.Drawing;
using SharpDX;

namespace Mpdn.RenderScript
{
    namespace Example
    {
        public class CustomTextures : RenderChain
        {
            private ITexture m_Texture1;
            private ITexture m_Texture2;

            protected override string ShaderPath
            {
                get { return "Examples"; }
            }

            public override IFilter CreateFilter(IFilter sourceFilter)
            {
                CreateTextures();
                var shader = CompileShader("CustomTextures.hlsl");
                return new ShaderFilter(shader, sourceFilter, new TextureSourceFilter(m_Texture1),
                    new TextureSourceFilter(m_Texture2));
            }

            public override void RenderScriptDisposed()
            {
                DiscardTextures();

                base.RenderScriptDisposed();
            }

            private void DiscardTextures()
            {
                DisposeHelper.Dispose(ref m_Texture1);
                DisposeHelper.Dispose(ref m_Texture2);
            }

            private void CreateTextures()
            {
                CreateTexture1();
                CreateTexture2();
            }

            private void CreateTexture1()
            {
                if (m_Texture1 != null)
                    return;

                const int width = 10;
                const int height = 10;
                m_Texture1 = CreateTexture(width, height);
            }

            private void CreateTexture2()
            {
                if (m_Texture2 != null)
                    return;

                const int width = 40;
                const int height = 40;
                m_Texture2 = CreateTexture(width, height);
            }

            private static ITexture CreateTexture(int width, int height)
            {
                int pitch = width*4;
                var result = Renderer.CreateTexture(new Size(width, height));
                var tex = GenerateChequeredPattern(pitch, height);
                Renderer.UpdateTexture(result, tex);
                return result;
            }

            private static Half[,] GenerateChequeredPattern(int pitch, int height)
            {
                var tex = new Half[height, pitch]; // 16-bit per channel, 4 channels per pixel
                var color0 = new Half[] { 0, 0, 0, 0 };
                var color1 = new Half[] { 0.1f, 0.1f, 0.1f, 0 };
                for (int j = 0; j < height; j++)
                {
                    for (int i = 0; i < pitch; i += 4)
                    {
                        // Fill texture with chequered pattern
                        var c = (i / 4 + j) % 2 == 0 ? color0 : color1;
                        tex[j, (i + 0)] = c[0]; // r
                        tex[j, (i + 1)] = c[1]; // g
                        tex[j, (i + 2)] = c[2]; // b
                        tex[j, (i + 3)] = c[3]; // a
                    }
                }
                return tex;
            }
        }

        public class CustomTexturesExample : RenderChainUi<CustomTextures>
        {
            public override ExtensionUiDescriptor Descriptor
            {
                get
                {
                    return new ExtensionUiDescriptor
                    {
                        Name = "Custom Textures Example",
                        Description = "(Example) Use custom textures as overlay",
                        Guid = new Guid("8BE548DE-F426-4249-95CB-879236866A07"),
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
