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
using IBaseFilter = Mpdn.RenderScript.IFilter<Mpdn.IBaseTexture>;

namespace Mpdn.RenderScript
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

        public void NewFrame()
        {
        }

        public void Render(ITextureCache cache)
        {
        }

        public virtual void Reset(ITextureCache cache)
        {
            if (typeof(TTexture) == typeof(ITexture))
                cache.PutTempTexture(OutputTexture as ITexture);
        }

        #endregion
    }

    public abstract class BaseSourceFilter : BaseSourceFilter<ITexture>, IFilter
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

        public override ITexture OutputTexture
        {
            get { return Renderer.InputRenderTarget; }
        }

        public override TextureSize OutputSize
        {
            get { return (m_OutputSize.IsEmpty ? Renderer.VideoSize : m_OutputSize); }
        }

        #endregion
    }

    public sealed class YSourceFilter : BaseSourceFilter
    {
        public override ITexture OutputTexture
        {
            get { return Renderer.TextureY; }
        }

        public override TextureSize OutputSize
        {
            get { return Renderer.LumaSize; }
        }

        public override void Reset(ITextureCache cache)
        {
        }
    }

    public sealed class USourceFilter : BaseSourceFilter
    {
        public override ITexture OutputTexture
        {
            get { return Renderer.TextureU; }
        }

        public override TextureSize OutputSize
        {
            get { return Renderer.ChromaSize; }
        }

        public override void Reset(ITextureCache cache)
        {
        }
    }

    public sealed class VSourceFilter : BaseSourceFilter
    {
        public override ITexture OutputTexture
        {
            get { return Renderer.TextureV; }
        }

        public override TextureSize OutputSize
        {
            get { return Renderer.ChromaSize; }
        }

        public override void Reset(ITextureCache cache)
        {
        }
    }

    public sealed class NullFilter : BaseSourceFilter
    {
        public override ITexture OutputTexture
        {
            get { return Renderer.OutputRenderTarget; }
        }

        public override TextureSize OutputSize
        {
            get { return Renderer.TargetSize; }
        }
    }


    public sealed class TextureSourceFilter : BaseSourceFilter
    {
        private readonly ITexture m_Texture;
        private readonly TextureSize m_Size;

        public TextureSourceFilter(ITexture texture)
        {
            m_Texture = texture;
            m_Size = new TextureSize(texture.Width, texture.Height);
        }

        public override ITexture OutputTexture
        {
            get { return m_Texture; }
        }

        public override TextureSize OutputSize
        {
            get { return m_Size; }
        }

        public override void Reset(ITextureCache cache)
        {
        }
    }

    public sealed class Texture3DSourceFilter : BaseSourceFilter<ITexture3D>
    {
        private readonly ITexture3D m_Texture;
        private readonly TextureSize m_Size;

        public Texture3DSourceFilter(ITexture3D texture)
        {
            m_Texture = texture;
            m_Size = new TextureSize(texture.Width, texture.Height, texture.Depth);
        }

        public override ITexture3D OutputTexture
        {
            get { return m_Texture; }
        }

        public override TextureSize OutputSize
        {
            get { return m_Size; }
        }

        public override void Reset(ITextureCache cache)
        {
        }
    }
}
