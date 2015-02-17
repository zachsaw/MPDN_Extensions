using System;
using Mpdn.RenderScript;
using SharpDX;

namespace Mpdn.RenderScripts
{
    namespace Example
    {
        public class Lut3D : RenderChain
        {
            private ITexture3D m_Texture3D;

            protected override string ShaderPath
            {
                get { return "Examples"; }
            }

            public override IFilter CreateFilter(IResizeableFilter sourceFilter)
            {
                Create3DTexture();
                var shader = CompileShader("Lut3D.hlsl");
                return new ShaderFilter(shader, true, sourceFilter, new Texture3DSourceFilter(m_Texture3D));
            }

            public override void RenderScriptDisposed()
            {
                DiscardTextures();

                base.RenderScriptDisposed();
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
                m_Texture3D = Renderer.CreateTexture3D(width, height, depth);
                Renderer.UpdateTexture3D(m_Texture3D, Create3DLut(width, height, depth));
            }

            private static Half[,,] Create3DLut(int width, int height, int depth)
            {
                // Create a color-swap 3D LUT (r ==> b, g ==> r, b ==> g)
                // Note: This method is very slow (it's called once on init only though),
                //       but in real-life scenario, you'd be loading it from a 3dlut file 
                //       instead of generating it on the fly
                var lut = new Half[depth, height, width*4];
                for (int b = 0; b < depth; b++)
                {
                    for (int g = 0; g < height; g++)
                    {
                        for (int r = 0; r < width; r++)
                        {
                            lut[b, g, r*4 + 0] = b/(float) (width-1); // R channel, swap it with B
                            lut[b, g, r*4 + 1] = r/(float) (height-1); // G channel, swap it with R
                            lut[b, g, r*4 + 2] = g/(float) (depth-1); // B channel, swap it with G
                            lut[b, g, r*4 + 3] = 1; // Alpha
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
        }
    }
}
