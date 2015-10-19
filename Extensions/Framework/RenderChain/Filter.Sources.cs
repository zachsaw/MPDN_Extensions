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
using Mpdn.RenderScript;
using Size = System.Drawing.Size;

namespace Mpdn.Extensions.Framework.RenderChain
{
    public abstract class BaseSourceFilter<TTexture> : IFilter<TTexture>
        where TTexture : class, IBaseTexture
    {
        public abstract TTexture OutputTexture { get; }

        public abstract TextureSize OutputSize { get; }

        #region IFilter Implementation

        protected BaseSourceFilter()
        {
            Tag = FilterTag.Bottom;
        } 

        public virtual TextureFormat OutputFormat
        {
            get
            {
                return OutputTexture != null
                    ? OutputTexture.Format
                    : Renderer.RenderQuality.GetTextureFormat();
            }
        }

        public virtual int LastDependentIndex { get; private set; }

        public virtual void Render()
        {
        }

        public virtual void Reset()
        {
        }

        public virtual void Initialize(int time = 1)
        {
            LastDependentIndex = time;
        }

        public virtual IFilter<TTexture> Compile()
        {
            return this;
        }

        public FilterTag Tag { get; protected set; }

        public void AddTag(FilterTag newTag)
        {
            Tag = Tag.Append(newTag);
        }

        #endregion

        #region Resource Management

        ~BaseSourceFilter()
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
        }

        #endregion
    }

    public abstract class BaseSourceFilter : BaseSourceFilter<ITexture2D>, IFilter
    {
    }

    public sealed class SourceFilter : BaseSourceFilter, IResizeableFilter
    {
        private TextureSize m_OutputSize;
        private YuvSourceFilter m_YuvFilter;

        public void SetSize(TextureSize targetSize)
        {
            m_OutputSize = targetSize;
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
                    PrescaleSize = (Size)OutputSize
                };
            }
        }

        public IFilter GetYuv()
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
            var chromastatus = StatusHelpers.ScaleDescription(Renderer.ChromaSize, OutputSize, Renderer.ChromaUpscaler, Renderer.ChromaDownscaler, chromaConvolver)
                .PrependToStatus("Chroma: ");
            var lumastatus = StatusHelpers.ScaleDescription(Renderer.VideoSize, OutputSize, Renderer.LumaUpscaler, Renderer.LumaDownscaler)
                .PrependToStatus("Luma: ");

            return chromastatus.AppendStatus(lumastatus);
        }

        #region IFilter Implementation

        private ITargetTexture OutputTarget { get; set; }

        private bool m_Updated;
        private bool m_Tagged;

        public override ITexture2D OutputTexture
        {
            get
            {
                return OutputTarget ?? Renderer.InputRenderTarget;                    
            }
        }

        public override TextureSize OutputSize
        {
            get { return (m_OutputSize.IsEmpty ? Renderer.VideoSize : m_OutputSize); }
        }

        public override void Reset()
        {
            m_Updated = false;

            if (OutputTarget != null)
            {
                TexturePool.PutTexture(OutputTarget);
            }

            OutputTarget = null;

            TexturePool.PutTempTexture(Renderer.InputRenderTarget);
        }

        public override void Render()
        {
            if (m_YuvFilter == null)
                return;

            if (m_Updated)
                return;

            OutputTarget = TexturePool.GetTexture(OutputSize, OutputFormat);

            Renderer.ConvertToRgb(OutputTarget, Renderer.InputRenderTarget, Renderer.Colorimetric,
                Renderer.OutputLimitedRange, Renderer.LimitChroma);

            m_Updated = true;
        }

        public override void Initialize(int time = 1)
        {
            if (m_Tagged)
            {
                AddTag(Status());
                m_Tagged = false;
            }
            base.Initialize(time);
        }

        #endregion

        private class YuvSourceFilter : BaseSourceFilter
        {
            private readonly SourceFilter m_RgbSourceFilter;

            public YuvSourceFilter(SourceFilter rgbSourceFilter)
            {
                m_RgbSourceFilter = rgbSourceFilter;
                
                Tag = new EmptyTag();
                Tag.AddInput(rgbSourceFilter);
            }

            public override ITexture2D OutputTexture
            {
                get
                {
                    return Renderer.InputRenderTarget;
                }
            }

            public override TextureSize OutputSize
            {
                get { return m_RgbSourceFilter.OutputSize; }
            }

            public override void Reset()
            {
                m_RgbSourceFilter.Reset();
            }

            public override void Initialize(int time = 1)
            {
                m_RgbSourceFilter.Initialize(time);
            }

            public override int LastDependentIndex
            {
                get { return m_RgbSourceFilter.LastDependentIndex; }
            }
        }
    }

    public sealed class YSourceFilter : BaseSourceFilter
    {
        public override ITexture2D OutputTexture
        {
            get { return Renderer.TextureY; }
        }

        public override TextureSize OutputSize
        {
            get { return Renderer.LumaSize; }
        }
    }

    public sealed class USourceFilter : BaseSourceFilter
    {
        public override ITexture2D OutputTexture
        {
            get { return Renderer.TextureU; }
        }

        public override TextureSize OutputSize
        {
            get { return Renderer.ChromaSize; }
        }
    }

    public sealed class VSourceFilter : BaseSourceFilter
    {
        public override ITexture2D OutputTexture
        {
            get { return Renderer.TextureV; }
        }

        public override TextureSize OutputSize
        {
            get { return Renderer.ChromaSize; }
        }
    }

    public sealed class NullFilter : BaseSourceFilter
    {
        public override ITexture2D OutputTexture
        {
            get { return Renderer.OutputRenderTarget; }
        }

        public override TextureSize OutputSize
        {
            get { return Renderer.TargetSize; }
        }
    }

    public class TextureSourceFilter<TTexture> : BaseSourceFilter<TTexture>
        where TTexture : class, IBaseTexture
    {
        protected readonly TTexture Texture;
        protected bool ManageTexture;

        private readonly TextureSize m_OutputSize;

        public TextureSourceFilter(TTexture texture)
        {
            Texture = texture;
            m_OutputSize = Texture.GetSize();

            /* Don't connect to bottom label */
            Tag = new EmptyTag();
        }

        public override TTexture OutputTexture
        {
            get { return Texture; }
        }

        public override TextureSize OutputSize
        {
            get { return m_OutputSize; }
        }

        protected override void Dispose(bool disposing)
        {
            base.Dispose(disposing);
            if (ManageTexture) // Disabled by default to keep backwards compatibility
                DisposeHelper.Dispose(Texture);
        }
    }

    public class ManagedTexture<TTexture> where TTexture : class, IDisposable
    {
        private TTexture m_Texture;
        private int m_Leases;

        public ManagedTexture(TTexture texture)
        {
            m_Texture = texture;
        }

        public TTexture GetLease()
        {
            m_Leases++;
            return m_Texture;
        }

        public void RevokeLease()
        {
            m_Leases--;
            if (m_Leases > 0) return;
            DisposeHelper.Dispose(m_Texture);
            m_Texture = null;
        }

        public bool Valid
        {
            get { return m_Texture != null; }
        }
    }

    public static class SharedTextureHelpers
    {
        public static ManagedTexture<TTexture> GetManaged<TTexture>(this TTexture texture) where TTexture : class, IDisposable
        {
            return new ManagedTexture<TTexture>(texture);
        }

        public static SharedTextureSourceFilter<TTexture> ToFilter<TTexture>(this ManagedTexture<TTexture> texture)
            where TTexture : class, IBaseTexture
        {
            return new SharedTextureSourceFilter<TTexture>(texture);
        }
    }

    public class SharedTextureSourceFilter<TTexture> : BaseSourceFilter<TTexture>
        where TTexture : class, IBaseTexture
    {
        private readonly TextureSize m_OutputSize;
        protected readonly ManagedTexture<TTexture> ManagedTexture;
        protected TTexture Texture;

        public SharedTextureSourceFilter(ManagedTexture<TTexture> managedTexture)
        {
            ManagedTexture = managedTexture;
            Texture = ManagedTexture.GetLease();
            m_OutputSize = Texture.GetSize();

            /* Don't connect to bottom label */
            Tag = new EmptyTag();
        }

        public override TTexture OutputTexture
        {
            get { return Texture; }
        }

        public override TextureSize OutputSize
        {
            get { return m_OutputSize; }
        }

        protected override void Dispose(bool disposing)
        {
            base.Dispose(disposing);

            if (Texture == null)
                return;

            ManagedTexture.RevokeLease();
            Texture = null;
        }
    }
}
