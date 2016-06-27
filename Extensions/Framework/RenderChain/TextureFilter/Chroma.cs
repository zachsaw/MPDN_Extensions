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
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Mpdn.Extensions.Framework.Filter;
using Mpdn.RenderScript;
using SharpDX;

namespace Mpdn.Extensions.Framework.RenderChain.TextureFilter
{
    public class DefaultChromaScaler : IChromaScaler
    {
        public ITextureFilter CreateChromaFilter(ITextureFilter lumaInput, ITextureFilter chromaInput, TextureSize targetSize, Vector2 chromaOffset)
        {
            var fullSizeChroma = new ResizeFilter(chromaInput, targetSize, TextureChannels.ChromaOnly,
                chromaOffset, Renderer.ChromaUpscaler, Renderer.ChromaDownscaler);
            fullSizeChroma.EnableTag();

            return new MergeFilter(lumaInput.SetSize(targetSize, tagged: true), fullSizeChroma)
                .ConvertToRgb()
                .Tagged(new ChromaScalerTag(chromaInput, fullSizeChroma.Description().PrependToStatus("Chroma: ")));
        }
    }

    public class InternalChromaScaler : IChromaScaler
    {
        private readonly VideoSourceFilter m_SourceFilter;

        public InternalChromaScaler(VideoSourceFilter sourceFilter)
        {
            m_SourceFilter = sourceFilter;
        }

        public ITextureFilter CreateChromaFilter(ITextureFilter lumaInput, ITextureFilter chromaInput, TextureSize targetSize, Vector2 chromaOffset)
        {
            if (lumaInput is YSourceFilter && chromaInput is ChromaSourceFilter && chromaOffset == Renderer.ChromaOffset)
                return m_SourceFilter;

            return null;
        }
    }

    public class CompositionFilter : TextureFilter, ICompositionFilter
    {
        public ITextureFilter Luma { get; private set; }
        public ITextureFilter Chroma { get; private set; }
        public TextureSize TargetSize { get; private set; }
        public Vector2 ChromaOffset { get; private set; }

        protected readonly ICompositionFilter Fallback;
        protected readonly IChromaScaler ChromaScaler;

        public CompositionFilter(ITextureFilter lumaInput, ITextureFilter chromaInput, IChromaScaler chromaScaler, TextureSize? targetSize = null, Vector2? chromaOffset = null, ICompositionFilter fallback = null)
        {
            if (lumaInput == null)
                throw new ArgumentNullException("lumaInput");
            if (chromaInput == null)
                throw new ArgumentNullException("chromaInput");

            Luma = lumaInput;
            Chroma = chromaInput;

            Fallback = fallback;
            ChromaScaler = chromaScaler ?? new DefaultChromaScaler();
            ChromaOffset = chromaOffset ?? Renderer.ChromaOffset;
            TargetSize = targetSize ?? Luma.Output.Size;
        }

        public ITextureFilter SetSize(TextureSize outputSize)
        {
            return Rebuild(targetSize: outputSize);
        }

        public ICompositionFilter Rebuild(IChromaScaler chromaScaler = null, TextureSize? targetSize = null, Vector2? chromaOffset = null, ICompositionFilter fallback = null)
        {
            return new CompositionFilter(Luma, Chroma, chromaScaler ?? ChromaScaler, targetSize ?? TargetSize, chromaOffset ?? ChromaOffset, fallback ?? Fallback)
                .Tagged(Tag);
        }

        public void EnableTag() { }

        protected override TextureSize OutputSize
        {
            get { return TargetSize; }
        }

        protected override IFilter<ITextureOutput<ITexture2D>> Optimize()
        {
            var Result = ChromaScaler.CreateChromaFilter(Luma, Chroma, TargetSize, ChromaOffset);
            if (Result != null)
                Result.Tag.AddPrefix(Tag);

            return (Result ?? Fallback)
                .SetSize(TargetSize, tagged: true)
                .Compile();
        }

        protected override void Render(IList<ITextureOutput<IBaseTexture>> inputs)
        {
            throw new NotImplementedException("Uncompiled Filter.");
        }
    }

    public class ChromaScalerTag : StringTag
    {
        private readonly FilterTag m_ChromaTag;

        public ChromaScalerTag(ITextureFilter chromaFilter, string label)
            : base(label)
        {
            m_ChromaTag = chromaFilter.Tag;
        }

        public override string CreateString(int minIndex = -1)
        {
            Initialize();

            var lumaPart = new EmptyTag();
            var chromaPart = new StringTag(Label);

            foreach (var tag in SubTags)
                if (tag.ConnectedTo(m_ChromaTag))
                    chromaPart.AddInput(tag);
                else
                    lumaPart.AddInput(tag);

            lumaPart.Initialize();
            chromaPart.Initialize();

            var luma = lumaPart
                .CreateString(minIndex)
                .FlattenStatus()
                .PrependToStatus("Luma: ");

            var chroma = chromaPart
                .CreateString(minIndex)
                .FlattenStatus();
            if (!chroma.StartsWith("Chroma: "))
                chroma = chroma.PrependToStatus(luma == "" ? "" : "Chroma: ");

            return chroma.AppendStatus(luma);
        }
    }

}