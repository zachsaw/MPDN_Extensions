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
using System.Linq;
using System.Threading;
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.Framework.RenderChain;

namespace Mpdn.Extensions.PlayerExtensions
{
    public class ScriptChainOsdPainter : PlayerExtension<ScriptChainOsdPainterSettings>
    {
        private const int TEXT_HEIGHT = 16;
        private IText m_Text;
        private Size m_VideoBoxSize;

        public override ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("231E26CC-A588-4AF2-AE24-28DC610FA05B"),
                    Name = "Script Chain OSD Painter",
                    Description = "Paints script chain OSD"
                };
            }
        }

        public override void Initialize()
        {
            m_Text = Player.CreateText("Arial", TEXT_HEIGHT, TextFontStyle.Regular);
            m_VideoBoxSize = Gui.VideoBox.ClientSize;
            DynamicHotkeys.RegisterHotkey(Guid.NewGuid(), "Ctrl+K", () =>
            {
                Settings.Enabled = !Settings.Enabled;
                // TODO: Persist Settings.Enabled (create a config dialog)

                if (Player.State == PlayerState.Playing)
                    return;

                Gui.VideoBox.Invalidate();
            });

            Player.PaintOverlay += OnPaintOverlay;
            Gui.VideoBox.SizeChanged += VideoBoxResize;
        }

        public override void Destroy()
        {
            Gui.VideoBox.SizeChanged -= VideoBoxResize;
            Player.PaintOverlay -= OnPaintOverlay;

            m_Text.Dispose();
        }

        private void VideoBoxResize(object sender, EventArgs eventArgs)
        {
            m_VideoBoxSize = Gui.VideoBox.ClientSize;
        }

        public static long AtomicRead(long target)
        {
            return Interlocked.CompareExchange(ref target, 0, 0);
        }

        public static void AtomicWrite(ref long target, long value)
        {
            Interlocked.Exchange(ref target, value);
        }

        private void OnPaintOverlay(object sender, EventArgs eventArgs)
        {
            // Warning: This is called from a foreign thread

            if (!Settings.Enabled)
            {
                m_Text.Hide();
                return;
            }
            var text = "Render Chain:\r\n" + RenderChainDescription.Text.Replace(" > ", "\r\n > ");
            text = text.Trim('\r', '\n');
            var width = m_Text.MeasureWidth(text);
            var height = text.Count(c => c == '\n')*(TEXT_HEIGHT);
            var size = m_VideoBoxSize;
            const int rightOffset = 5;
            const int bottomOffset = 5;
            var location = new Point(size.Width - width - rightOffset - 25, 30);
            m_Text.Show(text, location, Color.FromArgb(255, 0xBB, 0xBB, 0xBB),
                Color.FromArgb(255*60/100, Color.Black), new Padding(5, 5, rightOffset, bottomOffset + height));
        }
    }

    public class ScriptChainOsdPainterSettings
    {
        public ScriptChainOsdPainterSettings()
        {
            Enabled = false;
        }

        public bool Enabled { get; set; }
    }
}
