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
        public ITextureFilter CreateChromaFilter(ITextureFilter lumaInput, ITextureFilter chromaInput, TextureSize targetSize, Vector2 chromaOffset)
        {
            var fullSizeChroma = new ResizeFilter(chromaInput, targetSize, TextureChannels.ChromaOnly,
                chromaOffset, Renderer.ChromaUpscaler, Renderer.ChromaDownscaler);

            return new MergeFilter(lumaInput.SetSize(targetSize, tagged: true), fullSizeChroma)
                .ConvertToRgb()
                .Tagged(new ChromaScalerTag(chromaInput, fullSizeChroma.Status().PrependToStatus("Chroma: ")));
        }
    }

    public class InternalChromaScaler : IChromaScaler
    {
        private readonly ITextureFilter m_SourceFilter;

        public InternalChromaScaler(ITextureFilter sourceFilter)
        {
            m_SourceFilter = sourceFilter;
        }

        public ITextureFilter CreateChromaFilter(ITextureFilter lumaInput, ITextureFilter chromaInput, TextureSize targetSize, Vector2 chromaOffset)
        {
            if (lumaInput is YSourceFilter && chromaInput is ChromaSourceFilter && chromaOffset == Renderer.ChromaOffset)
                return m_SourceFilter.SetSize(targetSize, tagged: true);

            return null;
        }
    }

    public class CompositionFilter : TextureFilter, ICompositionFilter
    {
        public ITextureFilter Luma { get; private set; }
        public ITextureFilter Chroma { get; private set; }
        public TextureSize TargetSize { get; private set; }
        public Vector2 ChromaOffset { get; private set; }

        protected readonly ITextureFilter Result;
        protected readonly IChromaScaler ChromaScaler;

        public CompositionFilter(ITextureFilter lumaInput, ITextureFilter chromaInput, IChromaScaler chromaScaler, TextureSize? targetSize = null, Vector2? chromaOffset = null, ITextureFilter fallback = null)
        {
            if (lumaInput == null)
                throw new ArgumentNullException("lumaInput");
            if (chromaInput == null)
                throw new ArgumentNullException("chromaInput");

            Luma = lumaInput;
            Chroma = chromaInput;

            ChromaScaler = chromaScaler ?? new DefaultChromaScaler();
            ChromaOffset = chromaOffset ?? Renderer.ChromaOffset;
            TargetSize = targetSize ?? Luma.Output.Size;

            Result = ChromaScaler.CreateChromaFilter(Luma, Chroma, TargetSize, ChromaOffset) ?? fallback;
        }

        public ITextureFilter SetSize(TextureSize outputSize)
        {
            return Rebuild(targetSize: outputSize);
        }

        public ICompositionFilter Rebuild(IChromaScaler chromaScaler = null, TextureSize? targetSize = null, Vector2? chromaOffset = null)
        {
            return new CompositionFilter(Luma, Chroma, chromaScaler ?? ChromaScaler, targetSize ?? TargetSize, chromaOffset ?? ChromaOffset)
                .Tagged(Tag);
        }

        public void EnableTag() { }

        protected override TextureSize OutputSize
        {
            get { return Result.Output.Size; }
        }

        protected override TextureFormat OutputFormat
        {
            get { return Result.Output.Format; }
        }

        protected override IFilter<ITextureOutput<ITexture2D>> Optimize()
        {
            return Result.Compile();
        }

        protected override void Render(IList<ITextureOutput<IBaseTexture>> inputs)
        {
            throw new NotImplementedException();
        }
    }

    public class ChromaScalerTag : StringTag
    {
        private readonly ITextureFilter m_ChromaFilter;

        private FilterTag m_ChromaTag;

        public ChromaScalerTag(ITextureFilter chromaFilter, string label)
            : base(label)
        {
            m_ChromaFilter = chromaFilter;

            AddInput(m_ChromaFilter);
        }

        public override int Initialize(int count = 1)
        {
            if (!Initialized)
            {
                m_ChromaTag = m_ChromaFilter.Tag;
            }

            return base.Initialize(count);
        }

        public override string CreateString(int minIndex = -1)
        {
            Initialize();

            var lumaPart = new EmptyTag();
            var chromaPart = new StringTag(Label);

            foreach (var tag in SubTags)
                if (tag.ConnectedTo(m_ChromaTag))
                    chromaPart.AddInputLabel(tag);
                else
                    lumaPart.AddInputLabel(tag);

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