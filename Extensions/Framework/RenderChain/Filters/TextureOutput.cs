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
using Mpdn.Extensions.Framework.Filter;
using Mpdn.RenderScript;

namespace Mpdn.Extensions.Framework.RenderChain.Filters
{
    public interface ITextureDescription
    {
        TextureSize Size { get; }
        TextureFormat Format { get; }
    }

    public interface ITextureOutput<out TTexture> : IFilterOutput<ITextureDescription, TTexture>, ITextureDescription
        where TTexture : IBaseTexture
    { }

    public interface ITextureOutput : ITextureOutput<ITargetTexture> { }

    public class TextureDescription : ITextureDescription, IEquatable<ITextureDescription>
    {
        private readonly TextureSize m_Size;
        private readonly TextureFormat? m_Format;

        public TextureDescription(TextureSize size, TextureFormat? format = null)
        {
            m_Size = size;
            m_Format = format;
        }

        public TextureSize Size { get { return m_Size; } }
        public TextureFormat Format { get { return m_Format ?? Renderer.RenderQuality.GetTextureFormat(); } }

        public bool Equals(ITextureDescription other)
        {
            return (m_Size == other.Size && (m_Format == null || m_Format == other.Format));
        }
    }

    public class TextureOutput : FilterOutput<ITextureDescription, ITargetTexture>, ITextureDescription, ITextureOutput
    {
        public TextureOutput(TextureSize size, TextureFormat format)
        {
            m_Size = size;
            m_Format = format;
        }

        #region ITextureDescription Implementation

        private readonly TextureSize m_Size;
        private readonly TextureFormat m_Format;

        public TextureSize Size
        {
            get { return m_Size; }
        }

        public TextureFormat Format
        {
            get { return m_Format; }
        }

        #endregion 

        #region IFilterResult Implementation

        private ITargetTexture m_Texture;

        public override ITextureDescription Description
        {
            get { return this; }
        }

        protected override ITargetTexture Value
        {
            get { return m_Texture; }
        }

        protected override void Allocate()
        {
            m_Texture = TexturePool.GetTexture(Size, Format);
        }

        protected override void Deallocate()
        {
            if (m_Texture != null)
                TexturePool.PutTexture(m_Texture);

            m_Texture = null;
        }

        #endregion
    }
}