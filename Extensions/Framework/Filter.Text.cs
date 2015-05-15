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
using System.Drawing;
using System.Drawing.Imaging;
using System.Windows.Forms;
using SharpDX;
using Color = System.Drawing.Color;
using Point = System.Drawing.Point;
using Rectangle = System.Drawing.Rectangle;

namespace Mpdn.RenderScript
{
    public class TextFilter : BaseSourceFilter<ITexture>, IFilter
    {
        private string m_Text;
        private Font m_Font;
        private ITexture m_Texture;
        Func<TextureSize> m_Size;

        public TextFilter(string text, Func<TextureSize> size = null)
            : base()
        {
            m_Size = size ?? (() => Renderer.TargetSize);
            m_Text = text;
        }

        public override TextureSize OutputSize
        {
            get { return m_Size(); }
        }

        public override ITexture OutputTexture
        {
            get { return m_Texture; }
        }

        public override void Render(ITextureCache cache)
        {
            if (m_Texture != null && Renderer.TargetSize == m_Texture.GetSize())
                return;

            DisposeHelper.Dispose(ref m_Texture);
            m_Texture = Renderer.CreateTexture(Renderer.TargetSize);
            DrawText();
        }

        ~TextFilter()
        {
            DisposeHelper.Dispose(ref m_Font);
            DisposeHelper.Dispose(ref m_Texture);
        }

        #region Text rendering

        private Font textFont
        {
            get
            {
                return m_Font = m_Font ?? new Font(FontFamily.GenericMonospace, 11, FontStyle.Bold);
            }
        }

        private void DrawText()
        {
            var width = Renderer.TargetSize.Width;
            var height = Renderer.TargetSize.Height;
            using (var bmp = new Bitmap(width, height, PixelFormat.Format24bppRgb))
            {
                var bounds = new Rectangle(0, 0, bmp.Width, bmp.Height);
                using (var g = Graphics.FromImage(bmp))
                {
                    g.FillRectangle(Brushes.DarkSlateBlue, bounds);
                    TextRenderer.DrawText(g, m_Text, textFont, new Point(10, 10), Color.OrangeRed);
                }
                UpdateTexture(bmp);
            }
        }

        private unsafe void UpdateTexture(Bitmap bmp)
        {
            var width = bmp.Width;
            var height = bmp.Height;
            var bounds = new Rectangle(0, 0, width, height);

            var bmpData = bmp.LockBits(bounds, ImageLockMode.ReadOnly, bmp.PixelFormat);
            try
            {
                var pitch = width * 4;
                var tex = new Half[height, pitch];
                var bmpPtr = (byte*)bmpData.Scan0.ToPointer();
                for (int j = 0; j < height; j++)
                {
                    byte* ptr = bmpPtr + bmpData.Stride * j;
                    for (int i = 0; i < pitch; i += 4)
                    {
                        tex[j, (i + 3)] = 1; // a
                        tex[j, (i + 2)] = *ptr++ / 255.0f; // b
                        tex[j, (i + 1)] = *ptr++ / 255.0f; // g
                        tex[j, (i + 0)] = *ptr++ / 255.0f; // r
                    }
                }
                Renderer.UpdateTexture(OutputTexture, tex);
            }
            finally
            {
                bmp.UnlockBits(bmpData);
            }
        }

        #endregion
    }
}
