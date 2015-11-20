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

namespace Mpdn.Extensions.Framework.RenderChain
{
    public interface IChromaScaler
    {
        ITextureFilter CreateChromaFilter(ITextureFilter lumaInput, ITextureFilter chromaInput, TextureSize targetSize, Vector2 chromaOffset);
    }

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

    public static class ChromaHelper
    {
        public static ITextureFilter CreateChromaFilter(this IChromaScaler chromaScaler, ITextureFilter lumaInput, ITextureFilter chromaInput, Vector2 chromaOffset)
        {
            return chromaScaler.CreateChromaFilter(lumaInput, chromaInput, lumaInput.Output.Size, chromaOffset);
        }

        public static ITextureFilter MakeChromaFilter<TChromaScaler>(this TChromaScaler scaler, ITextureFilter input)
            where TChromaScaler : RenderChain, IChromaScaler
        {
            var chromaFilter = input as ChromaFilter;
            if (chromaFilter == null)
                return input;

            return chromaFilter.MakeNew(scaler)
                    .Tagged(new TemporaryTag("ChromaScaler"));
        }
    }

    public sealed class ChromaFilter : TextureFilter, IResizeableFilter
    {
        private readonly ITextureFilter m_Fallback;
        private IFilter<ITextureOutput<ITexture2D>> m_CompilationResult;

        public readonly ITextureFilter Luma;
        public readonly ITextureFilter Chroma;
        public readonly IChromaScaler ChromaScaler;
        public readonly Vector2 ChromaOffset;

        public TextureSize TargetSize;

        public ChromaFilter(ITextureFilter lumaInput, ITextureFilter chromaInput, TextureSize? targetSize = null, Vector2? chromaOffset = null, ITextureFilter fallback = null, IChromaScaler chromaScaler = null)
        {
            Luma = lumaInput;
            Chroma = chromaInput;
            TargetSize = targetSize ?? lumaInput.Output.Size;
            ChromaOffset = chromaOffset ?? Renderer.ChromaOffset;
            ChromaScaler = chromaScaler ?? new DefaultChromaScaler();

            m_Fallback = fallback ?? MakeNew();
            Tag = new EmptyTag();
        }

        public ChromaFilter MakeNew(IChromaScaler chromaScaler = null, TextureSize? targetSize = null, Vector2? chromaOffset = null)
        {
            return new ChromaFilter(Luma, Chroma, targetSize ?? TargetSize, chromaOffset ?? ChromaOffset, this, chromaScaler ?? ChromaScaler);
        }

        protected override TextureSize OutputSize
        {
            get { return TargetSize; }
        }

        protected override TextureFormat OutputFormat
        {
            get { return Renderer.RenderQuality.GetTextureFormat(); } // Not guaranteed
        }

        protected override void Render(IList<ITextureOutput<IBaseTexture>> inputs)
        {
            throw new NotImplementedException();
        }

        public void EnableTag() { /* ChromaScaler is *always* tagged */ }

        protected override IFilter<ITextureOutput<ITexture2D>> Optimize()
        {
            if (m_CompilationResult != null)
                return m_CompilationResult;

            var result = ChromaScaler.CreateChromaFilter(Luma, Chroma, TargetSize, ChromaOffset);

            if (result == null)
                result = m_Fallback;
            else
            {
                var chain = ChromaScaler as RenderChain;
                if (chain != null && result.Tag.IsEmpty())
                    result.AddTag(new ChromaScalerTag(Chroma, chain.Status));
            }

            result = result.SetSize(TargetSize, tagged: true);
            Tag.AddInput(result);

            return m_CompilationResult = result.Compile();
        }

        public void SetSize(TextureSize size)
        {
            TargetSize = size;
        }
    }
}