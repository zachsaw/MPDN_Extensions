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
using System.Collections.Generic;
using Mpdn.Extensions.Framework.Filter;
using Mpdn.RenderScript;
using Size = System.Drawing.Size;

namespace Mpdn.Extensions.Framework.RenderChain.TextureFilter
{
    public class SourceTextureOutput<TTexture> : FilterOutput, ITextureOutput<TTexture>
        where TTexture : class, IBaseTexture
    {
        private readonly bool m_ManageTexture;

        public virtual TTexture Texture { get; private set; }

        public SourceTextureOutput(TTexture texture, bool manageTexture)
            : this(manageTexture)
        {
            if (texture == null)
                throw new ArgumentNullException("texture");

            Texture = texture;
        }

        protected SourceTextureOutput(bool manageTexture)
        {
            m_ManageTexture = manageTexture;
        }

        public virtual TextureSize Size
        {
            get { return Texture.GetSize(); }
        }

        public virtual TextureFormat Format
        {
            get { return Texture.Format; }
        }

        public override void Allocate() { }
        public override void Deallocate() { }

        protected override void Dispose(bool disposing)
        {
            base.Dispose(disposing);
            if (m_ManageTexture)
                DisposeHelper.Dispose(Texture);
        }
    };

    public class DeferredTextureOutput<TTexture> : SourceTextureOutput<TTexture>
        where TTexture : class, IBaseTexture
    {
        private readonly Func<TTexture> m_TextureFunc;
        private readonly Func<TextureSize> m_Size;
        private readonly Func<TextureFormat> m_Format;

        public DeferredTextureOutput(Func<TTexture> textureFunc, TextureSize size)
            : this(textureFunc, () => size)
        { }

        public DeferredTextureOutput(Func<TTexture> textureFunc, TextureSize size, TextureFormat format)
            : this(textureFunc, () => size, () => format)
        { }

        public DeferredTextureOutput(Func<TTexture> textureFunc, Func<TextureSize> size = null, Func<TextureFormat> format = null)
            : base(false)
        {
            m_TextureFunc = textureFunc;
            m_Size = size ?? (() => new TextureSize(0,0));
            m_Format = format ?? (() => Renderer.RenderQuality.GetTextureFormat());
        }

        public override TTexture Texture { get { return m_TextureFunc(); } }

        public override TextureSize Size
        {
            get { return Texture == null ? m_Size() : Texture.GetSize(); }
        }

        public override TextureFormat Format
        {
            get { return Texture == null ? m_Format() : Texture.Format; }
        }
    }

    public class ManagedTexture<TTexture> : IManagedTexture<TTexture>
    where TTexture : class, IBaseTexture
    {
        private TTexture m_Texture;
        private int m_Leases;

        public ManagedTexture(TTexture texture)
        {
            if (texture == null)
            {
                throw new ArgumentNullException("texture");
            }
            m_Texture = texture;
        }

        ~ManagedTexture()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        public virtual void Dispose(bool disposing)
        {
            Discard();
        }

        public ITextureOutput<TTexture> GetLease()
        {
            if (!Valid)
                throw new InvalidOperationException("Cannot renew lease on a texture that is no longer valid");

            m_Leases++;
            return new Lease(this);
        }

        public void RevokeLease()
        {
            m_Leases--;
            if (m_Leases <= 0) Discard();
        }

        private void Discard()
        {
            DisposeHelper.Dispose(ref m_Texture);
            m_Leases = 0;
        }

        public bool Valid
        {
            get { return m_Texture != null; }
        }

        public class Lease : SourceTextureOutput<TTexture>
        {
            private ManagedTexture<TTexture> m_Owner;

            public Lease(ManagedTexture<TTexture> owner)
                : base(owner.m_Texture, false)
            {
                m_Owner = owner;
            }

            #region Resource Management

            protected override void Dispose(bool disposing)
            {
                base.Dispose(disposing);

                if (m_Owner == null)
                    return;

                m_Owner.RevokeLease();
                m_Owner = null;
            }

            #endregion Resource Management
        }
    }

    public class TextureSourceFilter<TTexture> : SourceFilter<ITextureOutput<TTexture>>, ITextureFilter<TTexture>
     where TTexture : class, IBaseTexture
    {
        public TextureSourceFilter(TTexture texture, bool manageTexture)
            : this(new SourceTextureOutput<TTexture>(texture, manageTexture))
        { }

        public TextureSourceFilter(ITextureOutput<TTexture> output)
            : base(output)
        {
            /* Don't connect to bottom label */
            Tag.RemoveInput(FilterTag.Bottom);
        }
    }
    
    public sealed class NullFilter : BaseSourceFilter<ITextureOutput<ITexture2D>>, ITextureFilter
    {
        protected override ITextureOutput<ITexture2D> DefineOutput()
        {
            return new DeferredTextureOutput<ITexture2D>(() => Renderer.OutputRenderTarget, Renderer.TargetSize);
        }
    }

    public sealed class YSourceFilter : BaseSourceFilter<ITextureOutput<ITexture2D>>, ITextureFilter
    {
        protected override ITextureOutput<ITexture2D> DefineOutput()
        {
            return new DeferredTextureOutput<ITexture2D>(() => Renderer.TextureY, Renderer.LumaSize);
        }
    }

    public sealed class USourceFilter : BaseSourceFilter<ITextureOutput<ITexture2D>>, ITextureFilter
    {
        protected override ITextureOutput<ITexture2D> DefineOutput()
        {
            return new DeferredTextureOutput<ITexture2D>(() => Renderer.TextureU, Renderer.ChromaSize);
        }
    }

    public sealed class VSourceFilter : BaseSourceFilter<ITextureOutput<ITexture2D>>, ITextureFilter
    {
        protected override ITextureOutput<ITexture2D> DefineOutput()
        {
            return new DeferredTextureOutput<ITexture2D>(() => Renderer.TextureV, Renderer.ChromaSize);
        }
    }

    public sealed class VideoSourceFilter : TextureFilter, IResizeableFilter
    {
        private readonly TextureSize m_OutputSize;
        private readonly bool m_WantYuv;
        private readonly TrueSourceFilter m_TrueSource;

        public ITextureFilter GetYuv()
        {
            return new VideoSourceFilter(m_TrueSource, m_OutputSize, true).Tagged(Tag);
        }

        public VideoSourceFilter(TrueSourceFilter trueSource, TextureSize? outputSize = null, bool? wantYuv = null)
            : base(trueSource)
        {
            m_TrueSource = trueSource;
            m_OutputSize = outputSize ?? trueSource.OutputSize;
            m_WantYuv = wantYuv ?? trueSource.WantYuv;

            if (m_WantYuv) m_TrueSource.WantYuv = true;
            m_TrueSource.OutputSize = m_OutputSize;
        }

        protected override TextureSize OutputSize { get { return m_OutputSize; } }

        protected override IFilter<ITextureOutput<ITexture2D>> Optimize()
        {
            ITextureFilter result = m_TrueSource.Resize(OutputSize);

            if (m_TrueSource.WantYuv && !m_WantYuv)
                result = result.ConvertToRgb();

            if (!m_TrueSource.WantYuv && m_WantYuv)
                result = result.ConvertToYuv();

            return result.Compile();
        }

        protected override void Render(IList<ITextureOutput<IBaseTexture>> inputs)
        {
            throw new NotImplementedException("Uncompiled Filter.");
        }

        public void EnableTag() { }

        public ITextureFilter SetSize(TextureSize outputSize)
        {
            return new VideoSourceFilter(m_TrueSource, outputSize, m_WantYuv).Tagged(Tag);
        }
    }

    public sealed class TrueSourceFilter : BaseSourceFilter<ITextureOutput<ITexture2D>>, ITextureFilter
    {
        private TextureSize? m_OutputSize;
        private bool m_WantYuv;

        // Note: Argument doesn't technically do anything, just prevents the creation of a TrueSourceFilter not coupled to an IRenderScript's Descriptor
        public TrueSourceFilter(IRenderScript script) { }

        public string Description()
        {
            var chromaConvolver = Renderer.ChromaOffset.IsZero ? null : Renderer.ChromaUpscaler;
            var chromastatus = StatusHelpers.ScaleDescription(Renderer.ChromaSize, OutputSize, Renderer.ChromaUpscaler, Renderer.ChromaDownscaler, chromaConvolver)
                .PrependToStatus("Chroma: ");
            var lumastatus = StatusHelpers.ScaleDescription(Renderer.VideoSize, OutputSize, Renderer.LumaUpscaler, Renderer.LumaDownscaler)
                .PrependToStatus("Luma: ");

            return chromastatus.AppendStatus(lumastatus);
        }

        protected override void Initialize()
        {
            Tag.Insert(Description());
        }

        public bool WantYuv
        {
            get { return m_WantYuv && Renderer.InputFormat.IsYuv(); }
            set { m_WantYuv = value; }
        }

        public TextureSize OutputSize // Uses last given value
        {
            get { return m_OutputSize ?? Renderer.VideoSize; }
            set { m_OutputSize = value; }
        }

        public ScriptInterfaceDescriptor Descriptor
        {
            get
            {
                return new ScriptInterfaceDescriptor
                {
                    WantYuv = WantYuv,
                    Prescale = LastDependentIndex > 0,
                    PrescaleSize = (Size)OutputSize
                };
            }
        }

        protected override ITextureOutput<ITexture2D> DefineOutput()
        {
            return new DeferredTextureOutput<ITexture2D>(() => Renderer.InputRenderTarget, () => OutputSize);
        }
    }
}
