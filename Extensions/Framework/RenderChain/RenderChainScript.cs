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
using System.Diagnostics;
using System.Linq;
using Mpdn.RenderScript;

namespace Mpdn.Extensions.Framework.RenderChain
{
    public class RenderChainScript : IRenderScript, IDisposable
    {
        private SourceFilter m_SourceFilter;
        private IFilter<ITexture2D> m_Filter;
        private FilterTag m_Tag;

        protected readonly RenderChain Chain;

        public RenderChainScript(RenderChain chain)
        {
            Chain = chain;
            Chain.Initialize();
            Status = string.Empty;
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
            get { return m_SourceFilter == null ? null : m_SourceFilter.Descriptor; }
        }

        public string Status { get; private set; }

        public void Update()
        {
            var initialFilter = MakeInitialFilter();
            initialFilter.MakeTagged();

            m_Filter = CreateSafeFilter(Chain, initialFilter)
                .SetSize(Renderer.TargetSize)
                .GetTag(out m_Tag)
                .Compile()
                .InitializeFilter();

            UpdateStatus();
        }

        private void UpdateStatus()
        {
            Status = m_Tag.CreateString();
        }

        public bool Execute()
        {
            if (Renderer.InputRenderTarget != Renderer.OutputRenderTarget)
                TexturePool.PutTempTexture(Renderer.OutputRenderTarget);

            m_Filter.Render();

            if (Renderer.OutputRenderTarget != m_Filter.OutputTexture)
                Scale(Renderer.OutputRenderTarget, m_Filter.OutputTexture);

            m_Filter.Reset();
            TexturePool.FlushTextures();

            return true;
        }

        private IResizeableFilter MakeInitialFilter()
        {
            m_SourceFilter = new SourceFilter();

            if (Renderer.InputFormat.IsRgb())
                return m_SourceFilter;

            if (Renderer.ChromaSize.Width < Renderer.LumaSize.Width || Renderer.ChromaSize.Height < Renderer.LumaSize.Height)
                return new ChromaFilter(new YSourceFilter(), new ChromaSourceFilter(), chromaScaler: new InternalChromaScaler(m_SourceFilter));

            return m_SourceFilter;
        }

        private static void Scale(ITargetTexture output, ITexture2D input)
        {
            Renderer.Scale(output, input, Renderer.LumaUpscaler, Renderer.LumaDownscaler);
        }

        #region Error Handling

        private TextFilter m_TextFilter;

        public IFilter CreateSafeFilter(RenderChain chain, IFilter input)
        {
            DisposeHelper.Dispose(ref m_TextFilter);
            try
            {
                return Chain.MakeFilter(input);
            }
            catch (Exception ex)
            {
                return DisplayError(ex);
            }
        }

        private IFilter DisplayError(Exception e)
        {
            var message = ErrorMessage(e);
            Trace.WriteLine(message);
            return m_TextFilter = new TextFilter(message);
        }

        protected static Exception InnerMostException(Exception e)
        {
            while (e.InnerException != null)
            {
                e = e.InnerException;
            }

            return e;
        }

        private string ErrorMessage(Exception e)
        {
            var ex = InnerMostException(e);
            return string.Format("Error in {0}:\r\n\r\n{1}\r\n\r\n~\r\nStack Trace:\r\n{2}",
                    GetType().Name, ex.Message, ex.StackTrace);
        }

        #endregion

    }
}
