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

namespace Mpdn.Extensions.Framework.RenderChain
{
    public abstract class BaseTextureSourceFilter<TTexture> : BaseSourceFilter<ITextureOutput<TTexture>>, ITextureFilter<TTexture>
        where TTexture : class, IBaseTexture
    { }

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

    public class ManagedTexture<TTexture> : IDisposable where TTexture : class, IBaseTexture
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

        public Lease GetLease()
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
            Tag = new EmptyTag();
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
        private readonly TrueSourceFilter m_TrueSource;

        public VideoSourceFilter() 
            : base(new RgbFilter(new TrueSourceFilter()))
        {
            var rgbSource = (RgbFilter) InputFilters[0];
            m_TrueSource = (TrueSourceFilter) rgbSource.InputFilters[0];
        }

        public void SetSize(TextureSize targetSize)
        {
            m_TrueSource.OutputSize = targetSize;
        }

        public void EnableTag()
        {
            m_Tagged = true;
        }

        public ScriptInterfaceDescriptor Descriptor
        {
            get { return m_TrueSource.Descriptor; }
        }

        public string Status()
        {
            var chromaConvolver = Renderer.ChromaOffset.IsZero ? null : Renderer.ChromaUpscaler;
            var chromastatus = StatusHelpers.ScaleDescription(Renderer.ChromaSize, Output.Size, Renderer.ChromaUpscaler,
                Renderer.ChromaDownscaler, chromaConvolver)
                .PrependToStatus("Chroma: ");
            var lumastatus = StatusHelpers.ScaleDescription(Renderer.VideoSize, Output.Size, Renderer.LumaUpscaler,
                Renderer.LumaDownscaler)
                .PrependToStatus("Luma: ");

            return chromastatus.AppendStatus(lumastatus);
        }

        #region IFilter Implementation

        private bool m_Tagged;

        protected override TextureSize OutputSize
        {
            get { return m_TrueSource.OutputSize; }
        }

        protected override void Render(IList<ITextureOutput<IBaseTexture>> textureOutputs)
        {
            throw new NotImplementedException();
        }

        protected override IFilter<ITextureOutput<ITexture2D>> Optimize()
        {
            var result = m_TrueSource.WantYuv ? (IFilter<ITextureOutput<ITexture2D>>) m_TrueSource.RgbSource : m_TrueSource;

            if (m_Tagged)
            {
                result.AddTag(Status());
                m_Tagged = false;
            }

            return result;
        }

        #endregion

        #region Auxilary Classes

        public sealed class TrueSourceFilter : BaseSourceFilter<ITextureOutput<ITexture2D>>, ITextureFilter
        {
            public bool WantYuv { get; private set; }
            public RgbFilter RgbSource { get; private set; }

            public TextureSize OutputSize;

            public ITextureFilter GetYuv()
            {
                WantYuv = true;
                RgbSource = new RgbFilter(this);
                return this;
            }

            public ScriptInterfaceDescriptor Descriptor
            {
                get
                {
                    return new ScriptInterfaceDescriptor
                    {
                        WantYuv = WantYuv,
                        Prescale = LastDependentIndex > 0 || (WantYuv && RgbSource.LastDependentIndex > 0),
                        PrescaleSize = (Size)OutputSize
                    };
                }
            }

            protected override ITextureOutput<ITexture2D> DefineOutput()
            {
                OutputSize = Renderer.VideoSize;
                return new DeferredTextureOutput<ITexture2D>(() => Renderer.InputRenderTarget, () => OutputSize);
            }
        }

        #endregion
    }

    public static class SharedTextureHelpers
    {
        public static ManagedTexture<TTexture> GetManaged<TTexture>(this TTexture texture) where TTexture : class, IBaseTexture
        {
            return new ManagedTexture<TTexture>(texture);
        }

        public static TextureSourceFilter<TTexture> ToFilter<TTexture>(this ManagedTexture<TTexture> texture)
            where TTexture : class, IBaseTexture
        {
            return new TextureSourceFilter<TTexture>(texture.GetLease());
        }
    }
}
