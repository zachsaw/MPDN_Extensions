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
using SharpDX;

namespace Mpdn.Extensions.Framework.RenderChain.TextureFilter
{
    public class DefaultChromaScaler : IChromaScaler
    {
        public ITextureFilter ScaleChroma(ICompositionFilter composition)
        {
            var fullSizeChroma = new ResizeFilter(composition.Chroma, composition.TargetSize, TextureChannels.ChromaOnly,
                composition.ChromaOffset, Renderer.ChromaUpscaler, Renderer.ChromaDownscaler);

            return new MergeFilter(composition.Luma.SetSize(composition.TargetSize, tagged: true), fullSizeChroma)
                .ConvertToRgb()
                .Labeled(fullSizeChroma.Description().PrependToDescription("Chroma: "));
        }
    }

    public class CompositionFilter : TextureFilter, ICompositionFilter
    {
        public ITextureFilter Luma { get; private set; }
        public ITextureFilter Chroma { get; private set; }

        public TextureSize TargetSize { get { return Target.Size; } }
        public Vector2 ChromaOffset { get; private set; }

        protected readonly ITextureFilter Fallback;

        protected override IFilter<ITextureOutput<ITexture2D>> Optimize()
        {
            return Fallback
                .SetSize(TargetSize)
                .Compile();
        }

        public ITextureFilter SetSize(TextureSize outputSize)
        {
            return new CompositionFilter(Luma, Chroma, outputSize, ChromaOffset);
        }

        public void EnableTag() { }

        protected override void Render(IList<ITextureOutput<IBaseTexture>> inputs)
        {
            throw new NotImplementedException("Uncompiled Filter.");
        }

        public CompositionFilter(ITextureFilter luma, ITextureFilter chroma, TextureSize? targetSize = null, Vector2? chromaOffset = null, ITextureFilter fallback = null)
            : base(targetSize ?? (fallback != null ? fallback.Size() : luma.Size()), luma, chroma)
        {
            if (luma == null)
                throw new ArgumentNullException("luma");
            if (chroma == null)
                throw new ArgumentNullException("chroma");

            Luma = luma;
            Chroma = chroma;

            ChromaOffset = chromaOffset ?? Renderer.ChromaOffset;
            Fallback = fallback ?? new DefaultChromaScaler().ScaleChroma(this);
        }
    }
}