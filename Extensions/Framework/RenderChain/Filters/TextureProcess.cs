using System;
using System.Linq;
using System.Collections.Generic;
using Mpdn.Extensions.Framework.Filter;

namespace Mpdn.Extensions.Framework.RenderChain.Filters
{
    public interface ITextureProcess<in TInput> : IProcess<TInput, ITextureFilter> { }

    public abstract class TextureProcess<TTexture> : ITextureProcess<ITextureFilter<TTexture>>
        where TTexture : IBaseTexture
    {
        protected abstract void Render(TTexture input, ITargetTexture output);

        protected abstract TextureSize CalcSize(TextureSize inputSize);

        protected abstract TextureFormat CalcFormat(TextureFormat inputFormat);

        private ITextureOutput Allocate(ITextureDescription input)
        {
            return new TextureOutput(CalcSize(input.Size), CalcFormat(input.Format));
        }

        public ITextureFilter ApplyTo(ITextureFilter<TTexture> input)
        {
            return new TextureFilter(
                from value in input
                select Allocate(input.Output).Do(Render, value));
        }
    }
}