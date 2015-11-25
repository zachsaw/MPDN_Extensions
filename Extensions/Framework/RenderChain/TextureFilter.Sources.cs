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

        public SourceTextureOutput(TTexture texture, bool manageTexture = false)
        {
            m_ManageTexture = manageTexture;
            Texture = texture;
        }

        protected SourceTextureOutput() { }

        public virtual TextureSize Size
        {
            get { return Texture.GetSize(); }
        }

        public virtual TextureFormat Format
        {
            get { return Texture.Format; }
        }

        public override void Allocate()
        { }

        public override void Deallocate()
        { }

        protected override void Dispose(bool disposing)
        {
            base.Dispose(disposing);
            if (m_ManageTexture) // Disabled by default to keep backwards compatibility
                DisposeHelper.Dispose(Texture);
        }
    };

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
                : base(owner.m_Texture)
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
        protected bool ManageTexture;

        public TextureSourceFilter(TTexture texture)
            : this(new SourceTextureOutput<TTexture>(texture))
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
            return new SourceTextureOutput<ITexture2D>(Renderer.OutputRenderTarget);
        }
    }

    public sealed class YSourceFilter : BaseSourceFilter<ITextureOutput<ITexture2D>>, ITextureFilter
    {
        protected override ITextureOutput<ITexture2D> DefineOutput()
        {
            return new SourceTextureOutput<ITexture2D>(Renderer.TextureY);
        }
    }

    public sealed class USourceFilter : BaseSourceFilter<ITextureOutput<ITexture2D>>, ITextureFilter
    {
        protected override ITextureOutput<ITexture2D> DefineOutput()
        {
            return new SourceTextureOutput<ITexture2D>(Renderer.TextureU);
        }
    }

    public sealed class VSourceFilter : BaseSourceFilter<ITextureOutput<ITexture2D>>, ITextureFilter
    {
        protected override ITextureOutput<ITexture2D> DefineOutput()
        {
            return new SourceTextureOutput<ITexture2D>(Renderer.TextureV);
        }
    }

    public sealed class VideoSourceFilter : TextureFilter, IResizeableFilter
    {
        private readonly VideoSourceOutput m_VideoSource = new VideoSourceOutput();
        private YuvSourceFilter m_YuvFilter;

        public void SetSize(TextureSize targetSize)
        {
            m_VideoSource.Size = targetSize;
        }

        public void EnableTag()
        {
            m_Tagged = true;
        }

        public ScriptInterfaceDescriptor Descriptor
        {
            get
            {
                return new ScriptInterfaceDescriptor
                {
                    WantYuv = m_YuvFilter != null,
                    Prescale = LastDependentIndex > 0 || m_YuvFilter != null,
                    PrescaleSize = (Size) OutputSize
                };
            }
        }

        public ITextureFilter<ITexture2D> GetYuv()
        {
            if (m_YuvFilter != null)
                return m_YuvFilter;

            if (Renderer.InputFormat.IsYuv())
            {
                m_YuvFilter = new YuvSourceFilter(this);
                return m_YuvFilter;
            }

            return new YuvFilter(this);
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

        protected override ITextureOutput<ITexture2D> DefineOutput()
        {
            if (m_YuvFilter == null)
                return m_VideoSource;

            return base.DefineOutput();
        }

        protected override TextureSize OutputSize
        {
            get { return Renderer.VideoSize; }
        }

        protected override void Render(IList<ITextureOutput<IBaseTexture>> textureOutputs)
        {
            if (m_YuvFilter == null)
                return;

            Renderer.ConvertToRgb(Target.Texture, Renderer.InputRenderTarget, Renderer.Colorimetric,
                Renderer.OutputLimitedRange, Renderer.LimitChroma);
        }

        protected override IFilter<ITextureOutput<ITexture2D>> Optimize()
        {
            if (m_Tagged)
            {
                AddTag(Status());
                m_Tagged = false;
            }

            return base.Optimize();
        }

        #endregion

        #region Auxilary Classes

        private class YuvSourceFilter : TextureSourceFilter<ITexture2D>, ITextureFilter
        {
            public YuvSourceFilter(VideoSourceFilter rgbSourceFilter)
                : base(rgbSourceFilter.Output)
            {
                Tag.AddInput(rgbSourceFilter);
            }
        }

        private sealed class VideoSourceOutput : FilterOutput, ITextureOutput<ITargetTexture>
        {
            private TextureSize m_OutputSize;

            public VideoSourceOutput()
            {
                Allocate();
            }

            public TextureSize Size
            {
                get { return m_OutputSize.IsEmpty ? Renderer.VideoSize : m_OutputSize; }
                set { m_OutputSize = value; }
            }

            public TextureFormat Format
            {
                get { return Renderer.InputRenderTarget.Format; }
            }

            public ITargetTexture Texture
            {
                get { return Renderer.InputRenderTarget; }
            }

            public override void Allocate() { }
            public override void Deallocate() { }
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
