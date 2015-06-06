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
using System.Drawing;
using Mpdn.RenderScript;

namespace Mpdn.Extensions.Framework.RenderChain
{
    public class RenderChainScript : IRenderScript, IDisposable
    {
        private SourceFilter m_SourceFilter;
        private IFilter<ITexture2D> m_Filter;

        protected readonly RenderChain Chain;

        public RenderChainScript(RenderChain chain)
        {
            Chain = chain;
            Chain.Initialize();
        }

        ~RenderChainScript()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            Chain.Reset();
        }

        public ScriptInterfaceDescriptor Descriptor
        {
            get
            {
                if (m_SourceFilter == null)
                    return null;

                return new ScriptInterfaceDescriptor
                {
                    WantYuv = true,
                    Prescale = (m_SourceFilter.LastDependentIndex > 0),
                    PrescaleSize = (Size)m_SourceFilter.OutputSize
                };
            }
        }

        public void Update()
        {
            m_SourceFilter = new SourceFilter();
            var rgbInput = m_SourceFilter.Transform(x => new RgbFilter(x));
            m_Filter = Chain
                .CreateSafeFilter(rgbInput)
                .SetSize(Renderer.TargetSize)
                .Compile();
            m_Filter.Initialize();
        }

        public void Render()
        {
            TexturePool.PutTempTexture(Renderer.OutputRenderTarget);
            m_Filter.Render();
            if (Renderer.OutputRenderTarget != m_Filter.OutputTexture)
            {
                Scale(Renderer.OutputRenderTarget, m_Filter.OutputTexture);
            }
            m_Filter.Reset();
            TexturePool.FlushTextures();
        }

        private static void Scale(ITargetTexture output, ITexture2D input)
        {
            Renderer.Scale(output, input, Renderer.LumaUpscaler, Renderer.LumaDownscaler);
        }
    }
}
