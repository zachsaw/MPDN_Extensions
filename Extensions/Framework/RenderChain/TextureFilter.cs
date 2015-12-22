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

namespace Mpdn.Extensions.Framework.RenderChain
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
        public ITargetTexture Texture { get; protected set; }
        public TextureSize Size { get; set; }
        public TextureFormat Format { get; set; }

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

    public interface ITextureFilter<out TTexture> : IFilter<ITextureOutput<TTexture>> 
        where TTexture : class, IBaseTexture
    { }

    public interface ITextureFilter : ITextureFilter<ITexture2D>
    { }

    public interface IResizeableFilter : ITextureFilter, ITaggableFilter<ITextureOutput<ITexture2D>>
    {
        void SetSize(TextureSize outputSize);
    }

    public abstract class TextureFilter : Filter<ITextureOutput<IBaseTexture>, ITextureOutput<ITexture2D>>, ITextureFilter
    {
        protected ITextureOutput<ITargetTexture> Target { get; private set; }

        protected TextureFilter(params IBaseTextureFilter[] inputFilters)
            : base(inputFilters)
        { }

        protected abstract TextureSize OutputSize { get; }

        protected virtual TextureFormat OutputFormat
        {
            get { return Renderer.RenderQuality.GetTextureFormat(); }
        }

        protected override ITextureOutput<ITexture2D> DefineOutput()
        {
            return Target = new TextureOutput
            {
                Size = OutputSize,
                Format = OutputFormat
            };
        }
    }
}