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

using Mpdn.Extensions.Framework.Filter;
using Mpdn.RenderScript;

namespace Mpdn.Extensions.Framework.RenderChain.TextureFilter
{
    using IBaseTextureFilter = IFilter<ITextureOutput<IBaseTexture>>;

    public interface ITextureOutput<out TTexture> : IFilterOutput
    where TTexture : class, IBaseTexture
    {
        TTexture Texture { get; }
        TextureSize Size { get; }
        TextureFormat Format { get; }
    }

    public class TextureOutput : FilterOutput, ITextureOutput<ITargetTexture>
    {
        private readonly TextureSize m_Size;
        private readonly TextureFormat m_Format;

        public TextureOutput(TextureSize size, TextureFormat format)
        {
            m_Size = size;
            m_Format = format;
        }

        public ITargetTexture Texture { get; protected set; }

        public TextureSize Size
        {
            get { return m_Size; }
        }

        public TextureFormat Format
        {
            get { return m_Format; }
        }

        public override void Allocate()
        {
            Texture = TexturePool.GetTexture(Size, Format);
        }

        public override void Deallocate()
        {
            if (Texture != null)
                TexturePool.PutTexture(Texture);

            Texture = null;
        }
    };

    public abstract class TextureFilter : Filter<ITextureOutput<IBaseTexture>, ITextureOutput<ITexture2D>>, ITextureFilter
    {
        protected ITextureOutput<ITargetTexture> Target { get; private set; }

        protected TextureFilter(TextureSize size, params IBaseTextureFilter[] inputFilters)
            : this(size, Renderer.RenderQuality.GetTextureFormat(), inputFilters)
        { }

        protected TextureFilter(TextureSize size, TextureFormat format, params IBaseTextureFilter[] inputFilters)
            : this(new TextureOutput(size, format), inputFilters)
        { }

        private TextureFilter(ITextureOutput<ITargetTexture> outputTarget, params IBaseTextureFilter[] inputFilters)
            : base(outputTarget, inputFilters)
        {
            Target = outputTarget;
        }
    }
}