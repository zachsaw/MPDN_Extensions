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
// 
using System;
using System.Collections.Generic;
using System.Drawing;
using SharpDX;
using ITexture3D = Mpdn.ISourceTexture3D;

namespace Mpdn.RenderScript
{
    public interface ITextureCache
    {
        ITargetTexture GetTexture(TextureSize textureSize, TextureFormat textureFormat);
        void PutTexture(ITargetTexture texture);
        void PutTempTexture(ITargetTexture texture);
    }

    public struct TextureSize
    {
        public readonly int Width;
        public readonly int Height;
        public readonly int Depth;

        public bool Is3D
        {
            get { return Depth != 1; }
        }

        public bool IsEmpty
        {
            get { return (Width == 0) || (Height == 0) || (Depth == 0); }
        }

        public TextureSize(int width, int height, int depth = 1)
        {
            Width = width;
            Height = height;
            Depth = depth;
        }

        public static bool operator ==(TextureSize a, TextureSize b)
        {
            return a.Equals(b);
        }

        public static bool operator !=(TextureSize a, TextureSize b)
        {
            return !a.Equals(b);
        }

        public bool Equals(TextureSize other)
        {
            return Width == other.Width && Height == other.Height && Depth == other.Depth;
        }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj))
                return false;

            return obj is TextureSize && Equals((TextureSize)obj);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                var hashCode = Width;
                hashCode = (hashCode * 397) ^ Height;
                hashCode = (hashCode * 397) ^ Depth;
                return hashCode;
            }
        }

        public static implicit operator TextureSize(Size size)
        {
            return new TextureSize(size.Width, size.Height);
        }

        public static explicit operator Size(TextureSize size)
        {
            return new Size(size.Width, size.Height);
        }

        public static implicit operator Vector2(TextureSize size)
        {
            return new Vector2(size.Width, size.Height);
        }
    }

    public static class TextureHelper
    {
        public static TextureSize GetSize(this IBaseTexture texture)
        {
            if (texture is ITexture2D)
            {
                var t = texture as ITexture2D;
                return new TextureSize(t.Width, t.Height);
            }
            if (texture is ITexture3D)
            {
                var t = texture as ITexture3D;
                return new TextureSize(t.Width, t.Height, t.Depth);
            }
            throw new ArgumentException("Invalid texture type");
        }
    }

    public static class TexturePool
    {
        private static readonly List<ITargetTexture> s_OldTextures = new List<ITargetTexture>();
        private static readonly List<ITargetTexture> s_SavedTextures = new List<ITargetTexture>();
        private static readonly List<ITargetTexture> s_TempTextures = new List<ITargetTexture>();

        public static ITargetTexture GetTexture(TextureSize textureSize, TextureFormat? textureFormat = null)
        {
            foreach (var list in new[] {s_SavedTextures, s_OldTextures})
            {
                var index = list.FindIndex(x => (x.GetSize() == textureSize) && (x.Format == textureFormat));
                if (index < 0) 
                    continue;

                var texture = list[index];
                list.RemoveAt(index);
                return texture;
            }

            return Renderer.CreateRenderTarget(textureSize.Width, textureSize.Height,
                textureFormat ?? Renderer.RenderQuality.GetTextureFormat());
        }

        public static void PutTempTexture(ITargetTexture texture)
        {
            s_TempTextures.Add(texture);
            s_SavedTextures.Add(texture);
        }

        public static void PutTexture(ITargetTexture texture)
        {
            s_SavedTextures.Add(texture);
        }

        public static void FlushTextures()
        {
            foreach (var texture in s_OldTextures)
            {
                DisposeHelper.Dispose(texture);
            }

            foreach (var texture in s_TempTextures)
            {
                s_SavedTextures.Remove(texture);
            }

            s_OldTextures.Clear();
            s_OldTextures.AddRange(s_SavedTextures);

            s_TempTextures.Clear();
            s_SavedTextures.Clear();
        }
    }
}