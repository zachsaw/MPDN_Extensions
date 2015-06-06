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
using System.Drawing;
using System.Drawing.Imaging;
using System.Drawing.Text;
using System.Windows.Forms;
using Mpdn.RenderScript;
using SharpDX;
using Color = System.Drawing.Color;
using Rectangle = System.Drawing.Rectangle;

namespace Mpdn.Extensions.Framework.Filter
{
    public class TextFilter : BaseSourceFilter<ITexture2D>, IFilter, IDisposable
    {
        private Font m_Font;
        private ISourceTexture m_Texture;

        public TextFilter(string text)
        {
            m_Texture = Renderer.CreateTexture(Renderer.TargetSize);
            DrawText(text);
        }

        public override TextureSize OutputSize
        {
            get { return m_Texture.GetSize(); }
        }

        public override ITexture2D OutputTexture
        {
            get { return m_Texture; }
        }

        public override void Reset()
        {
        }

        public override void Render()
        {
        }

        #region Text rendering

        private Font TextFont
        {
            get
            {
                return m_Font = m_Font ?? new Font(FontFamily.GenericMonospace, 11, FontStyle.Bold);
            }
        }

        private void DrawText(string text)
        {
            var size = Renderer.TargetSize;
            var width = size.Width;
            var height = size.Height;
            using (var bmp = new Bitmap(width, height, PixelFormat.Format24bppRgb))
            {
                var bounds = new Rectangle(0, 0, bmp.Width, bmp.Height);
                using (var g = Graphics.FromImage(bmp))
                {
                    g.FillRectangle(Brushes.DarkSlateBlue, bounds);
                    g.TextRenderingHint = TextRenderingHint.AntiAlias;
                    const int margin = 10;
                    var textBounds = new Rectangle(margin, margin, width-margin*2, height-margin*2);
                    TextRenderer.DrawText(g, text, TextFont, textBounds, Color.OrangeRed, Color.Transparent,
                        TextFormatFlags.WordBreak);
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
                Renderer.UpdateTexture(m_Texture, tex);
            }
            finally
            {
                bmp.UnlockBits(bmpData);
            }
        }

        #endregion

        #region Resource Disposal Methods

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected void Dispose(bool disposing)
        {
            DisposeHelper.Dispose(ref m_Font);
            DisposeHelper.Dispose(ref m_Texture);
        }

        ~TextFilter()
        {
            Dispose(false);
        }

        #endregion
    }
}
