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

namespace Mpdn.Extensions.Framework.RenderChain.Filters
{
    public class DefaultChromaScaler : IChromaScaler
    {
        public static ITextureFilter ScaleChroma(ITextureFilter luma, ITextureFilter chroma, TextureSize targetSize, Vector2 chromaOffset)
        {
            var resizedChroma = new ResizeFilter(chroma, targetSize, TextureChannels.ChromaOnly, chromaOffset, Renderer.ChromaUpscaler, Renderer.ChromaDownscaler);
            resizedChroma.AddLabel(resizedChroma.ScaleDescription.AddPrefixToDescription("Chroma: "));

            return luma
                .SetSize(targetSize, tagged: true)
                .MergeWith(resizedChroma)
                .ConvertToRgb();
        }

        public ITextureFilter ScaleChroma(ICompositionFilter composition)
        {
            return ScaleChroma(composition.Luma, composition.Chroma, composition.TargetSize, composition.ChromaOffset);
        }
    }

    public class CompositionFilter : TextureFilter, ICompositionFilter, ICanUndo<RgbProcess>
    {
        public ITextureFilter Luma { get; private set; }
        public ITextureFilter Chroma { get; private set; }
        public Vector2 ChromaOffset { get; private set; }

        public TextureSize TargetSize { get { return Output.Size; } }

        public CompositionFilter(ITextureFilter luma, ITextureFilter chroma, TextureSize? targetSize = null, Vector2? chromaOffset = null)
            : this(luma, chroma, targetSize ?? luma.Size(), chromaOffset ?? Renderer.ChromaOffset)
        { }

        #region IResizeable Implementation

        public ITextureFilter ResizeTo(TextureSize outputSize)
        {
            return new CompositionFilter(Luma, Chroma, outputSize, ChromaOffset);
        }

        public void EnableTag() { }

        #endregion

        #region Implementation

        private readonly ITextureFilter m_Fallback;

        ITextureFilter ICanUndo<RgbProcess>.Undo()
        {
            return m_Fallback.ConvertToYuv();
        }

        private CompositionFilter(ITextureFilter luma, ITextureFilter chroma, TextureSize targetSize, Vector2 chromaOffset)
            : this(luma, chroma, DefaultChromaScaler.ScaleChroma(luma, chroma, targetSize, chromaOffset), targetSize, chromaOffset)
        { }

        private CompositionFilter(ITextureFilter luma, ITextureFilter chroma, ITextureFilter fallback, TextureSize targetSize, Vector2 chromaOffset)
            : base(fallback)
        {
            if (luma == null)
                throw new ArgumentNullException("luma");
            if (chroma == null)
                throw new ArgumentNullException("chroma");

            Luma = luma;
            Chroma = chroma;
            m_Fallback = fallback;

            ChromaOffset = chromaOffset;
        }

        #endregion
    }
}