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
using SharpDX;

namespace Mpdn.Extensions.Framework.RenderChain
{
    public interface IChromaScaler
    {
        IFilter CreateChromaFilter(IFilter lumaInput, IFilter chromaInput, TextureSize targetSize, Vector2 chromaOffset);
    }

    public class DefaultChromaScaler : IChromaScaler
    {
        public IFilter CreateChromaFilter(IFilter lumaInput, IFilter chromaInput, TextureSize targetSize, Vector2 chromaOffset)
        {
            var fullSizeChroma = new ResizeFilter(chromaInput, targetSize, TextureChannels.ChromaOnly, 
                chromaOffset, Renderer.ChromaUpscaler, Renderer.ChromaDownscaler);

            return new MergeFilter(lumaInput.SetSize(targetSize), fullSizeChroma).ConvertToRgb();
        }
    }

    public class InternalChromaScaler : IChromaScaler
    {
        private readonly IFilter m_SourceFilter;

        public InternalChromaScaler(IFilter sourceFilter)
        {
            m_SourceFilter = sourceFilter;
        }

        public IFilter CreateChromaFilter(IFilter lumaInput, IFilter chromaInput, TextureSize targetSize, Vector2 chromaOffset)
        {
            if (lumaInput is YSourceFilter && chromaInput is ChromaSourceFilter && chromaOffset == Renderer.ChromaOffset)
                return m_SourceFilter.SetSize(targetSize);

            return new ChromaFilter(lumaInput, chromaInput, null, targetSize, chromaOffset);
        }
    }

    public static class ChromaHelper
    {
        public static IFilter CreateChromaFilter(this IChromaScaler chromaScaler, IFilter lumaInput, IFilter chromaInput, Vector2 chromaOffset)
        {
            return chromaScaler.CreateChromaFilter(lumaInput, chromaInput, lumaInput.OutputSize, chromaOffset);
        }

        public static IFilter CreateChromaFilter<TChromaScaler>(this TChromaScaler scaler, IFilter input)
            where TChromaScaler : RenderChain, IChromaScaler
        {
            var chromaFilter = input as ChromaFilter;
            if (chromaFilter != null)
            {
                var result = chromaFilter.MakeNew(scaler);
                if (result.FallbackOccurred)
                    scaler.MarkInactive();
                
                return result;
            }

            scaler.MarkInactive();
            return input;
        }
    }

    public sealed class ChromaFilter : BaseSourceFilter, IResizeableFilter
    {
        public IFilter Luma { get; private set; }
        public IFilter Chroma { get; private set; }
        public TextureSize TargetSize { get; private set; }
        public Vector2 ChromaOffset { get; private set; }
        public IChromaScaler ChromaScaler { get; private set; }
        public bool FallbackOccurred { get; private set; }

        public ChromaFilter(IFilter lumaInput, IFilter chromaInput, IChromaScaler chromaScaler = null, TextureSize? targetSize = null, Vector2? chromaOffset = null)
        {
            Luma = lumaInput;
            Chroma = chromaInput;
            ChromaOffset = chromaOffset ?? Renderer.ChromaOffset;
            ChromaScaler = chromaScaler ?? new DefaultChromaScaler();
            TargetSize = targetSize ?? Renderer.TargetSize;
        }

        public ChromaFilter MakeNew(IChromaScaler chromaScaler = null, TextureSize? targetSize = null, Vector2? chromaOffset = null)
        {
            return new ChromaFilter(Luma, Chroma, chromaScaler ?? ChromaScaler, targetSize ?? TargetSize, chromaOffset ?? ChromaOffset);
        }

        public override ITexture2D OutputTexture
        {
            get { throw new InvalidOperationException(); }
        }

        public override TextureSize OutputSize
        {
            get { return TargetSize; }
        }

        public override TextureFormat OutputFormat
        {
            get { return Renderer.RenderQuality.GetTextureFormat(); } // Not guaranteed
        }

        public override IFilter<ITexture2D> Compile()
        {
            var result = ChromaScaler.CreateChromaFilter(Luma, Chroma, TargetSize, ChromaOffset);
            if (result == null)
            {
                result = new ChromaFilter(Luma, Chroma, null, TargetSize, ChromaOffset);
                FallbackOccurred = true;
            }
            return result
                    .SetSize(OutputSize)
                    .Compile();
        }

        public void SetSize(TextureSize size)
        {
            TargetSize = size;
        }
    }
}