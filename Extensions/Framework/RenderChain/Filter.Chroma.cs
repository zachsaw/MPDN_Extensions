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
using Mpdn.RenderScript;
using SharpDX;

namespace Mpdn.Extensions.Framework.RenderChain
{
    public interface IChromaScaler
    {
        IFilter CreateChromaFilter(IFilter lumaInput, IFilter chromaInput, Vector2 chromaOffset);
    }

    public class DefaultChromaScaler : IChromaScaler
    {
        public IFilter CreateChromaFilter(IFilter lumaInput, IFilter chromaInput, Vector2 chromaOffset)
        {
            var fullSizeChroma = new ResizeFilter(chromaInput, lumaInput.OutputSize, TextureChannels.ChromaOnly, chromaOffset);
            return new MergeFilter(lumaInput, fullSizeChroma);
        }
    }

    public class ChromaFilter : Filter, IResizeableFilter
    {
        public IFilter Luma { get; set; }
        public IFilter Chroma { get; set; }
        public Vector2 ChromaOffset { get; set; }

        public IChromaScaler ChromaScaler { get; set; }

        public ChromaFilter(IFilter lumaInput, IFilter chromaInput, IChromaScaler chromaScaler = null, Vector2? chromaOffset = null)
        {
            Luma = lumaInput;
            Chroma = chromaInput;
            ChromaOffset = chromaOffset ?? Renderer.ChromaOffset;
            ChromaScaler = chromaScaler ?? new DefaultChromaScaler();
        }

        protected override void Render(IList<IBaseTexture> inputs)
        {
            throw new NotImplementedException();
        }

        protected override IFilter<ITexture2D> Optimize()
        {
            var result = ChromaScaler.CreateChromaFilter(Luma, Chroma, ChromaOffset).ConvertToRgb();

            if (result.OutputSize != OutputSize)
                throw new InvalidOperationException("Chroma scaler isn't allowed to change image size.");

            return result;
        }

        public override TextureSize OutputSize
        {
            get { return Luma.OutputSize; }
        }

        public override TextureFormat OutputFormat
        {
            get { return Chroma.OutputFormat; } // Possibly incorrect, might need to be fixed
        }

        public void SetSize(TextureSize size)
        {
            Luma = Luma.SetSize(size);
        }
    }
}