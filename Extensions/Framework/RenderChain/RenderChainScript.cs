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
using Mpdn.Extensions.Framework.Chain;
using Mpdn.Extensions.Framework.RenderChain.TextureFilter;
using Mpdn.RenderScript;

namespace Mpdn.Extensions.Framework.RenderChain
{
    public class RenderChainScript : FilterChainScript<ITextureFilter, ITextureOutput<ITexture2D>>, IRenderScript
    {
        private VideoSourceFilter m_SourceFilter;

        public RenderChainScript(Chain<ITextureFilter> chain)
            : base(chain) 
        { }

        public ScriptInterfaceDescriptor Descriptor
        {
            get { return m_SourceFilter == null ? DefaultDescriptor() : m_SourceFilter.Descriptor; }
        }

        protected ScriptInterfaceDescriptor DefaultDescriptor()
        {
            return new ScriptInterfaceDescriptor
            {
                PrescaleSize = Renderer.VideoSize,
                Prescale = true,
                WantYuv = false
            };
        }

        protected override void OutputResult(ITextureOutput<ITexture2D> result)
        {
            if (Renderer.OutputRenderTarget != result.Texture)
                Scale(Renderer.OutputRenderTarget, result.Texture);
        }

        public override bool Execute()
        {
            try
            {
                if (Renderer.InputRenderTarget != Renderer.OutputRenderTarget)
                    TexturePool.PutTempTexture(Renderer.OutputRenderTarget);
                return base.Execute();
            }
            finally
            {
                TexturePool.FlushTextures();
            }
        }

        protected override ITextureFilter MakeInitialFilter()
        {
            m_SourceFilter = new VideoSourceFilter(this);

            if (Renderer.InputFormat.IsYuv() 
                && (Renderer.ChromaSize.Width < Renderer.LumaSize.Width || Renderer.ChromaSize.Height < Renderer.LumaSize.Height))
                return new InternalChromaScaler(m_SourceFilter).MakeChromaFilter(new YSourceFilter(), new ChromaSourceFilter());

            return m_SourceFilter;
        }

        protected override ITextureFilter FinalizeOutput(ITextureFilter output)
        {
            return output.SetSize(Renderer.TargetSize, tagged: true);
        }

        protected override ITextureFilter HandleError(Exception e)
        {
            var message = ErrorMessage(e);
            Trace.WriteLine(message);
            return new TextFilter(message);
        }

        private static void Scale(ITargetTexture output, ITexture2D input)
        {
            Renderer.Scale(output, input, Renderer.LumaUpscaler, Renderer.LumaDownscaler);
        }

        #region Resource Management

        protected override void Dispose(bool disposing)
        {
            base.Dispose(disposing);
            DisposeHelper.Dispose(ref m_SourceFilter);
        }

        #endregion
    }
}
