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

using Mpdn.RenderScript;
using IBaseFilter = Mpdn.Extensions.Framework.RenderChain.IFilter<Mpdn.IBaseTexture>;

namespace Mpdn.Extensions.Framework.RenderChain
{
    public abstract class BaseSourceFilter<TTexture> : IFilter<TTexture>
        where TTexture : class, IBaseTexture
    {
        protected BaseSourceFilter(params IBaseFilter[] inputFilters)
        {
            InputFilters = inputFilters;
        }

        public abstract TTexture OutputTexture { get; }

        public abstract TextureSize OutputSize { get; }

        public abstract void Reset();

        #region IFilter Implementation

        public IBaseFilter[] InputFilters { get; protected set; }

        public virtual int FilterIndex
        {
            get { return 0; }
        }

        public virtual int LastDependentIndex { get; private set; }

        public void Initialize(int time = 1)
        {
            LastDependentIndex = time;
        }

        public IFilter<TTexture> Compile()
        {
            return this;
        }

        public TextureFormat OutputFormat
        {
            get
            {
                return OutputTexture != null
                    ? OutputTexture.Format
                    : Renderer.RenderQuality.GetTextureFormat();
            }
        }

        public virtual void Render()
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

        public void SetSize(TextureSize targetSize)
        {
            m_OutputSize = targetSize;
        }

        #region IFilter Implementation

        public override ITexture2D OutputTexture
        {
            get { return Renderer.InputRenderTarget; }
        }

        public override TextureSize OutputSize
        {
            get { return (m_OutputSize.IsEmpty ? Renderer.VideoSize : m_OutputSize); }
        }

        public override void Reset()
        {
            TexturePool.PutTempTexture(OutputTexture as ITargetTexture);
        }

        #endregion
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

        public override void Reset()
        {
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

        public override void Reset()
        {
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

        public override void Reset()
        {
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

        public override void Reset()
        {
        }
    }

    public sealed class TextureSourceFilter<TTexture> : BaseSourceFilter<TTexture>
        where TTexture : class, IBaseTexture
    {
        private readonly TTexture m_Texture;
        private readonly TextureSize m_Size;

        public TextureSourceFilter(TTexture texture)
        {
            m_Texture = texture;
            m_Size = m_Texture.GetSize();
        }

        public override TTexture OutputTexture
        {
            get { return m_Texture; }
        }

        public override TextureSize OutputSize
        {
            get { return m_Size; }
        }

        public override void Reset()
        {
        }
    }
}
