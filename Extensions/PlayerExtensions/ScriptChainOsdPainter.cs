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
using System.Windows.Forms;
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.Framework.RenderChain;
using Mpdn.RenderScript;

namespace Mpdn.Extensions.PlayerExtensions
{
    public class ScriptChainOsdPainter : PlayerExtension<ScriptChainOsdPainterSettings, ScriptChainOsdPainterConfigDialog>
    {
        private const int TEXT_HEIGHT = 15;
        private Timer m_Timer;
        private IText m_Text;
        private Size m_VideoBoxSize;
        private bool m_Resizing;

        public override ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("231E26CC-A588-4AF2-AE24-28DC610FA05B"),
                    Name = "Scale Chain OSD Painter",
                    Description = "Paints scale chain OSD"
                };
            }
        }

        public override void Initialize()
        {
            base.Initialize();

            m_Text = Player.CreateText("Verdana", TEXT_HEIGHT, TextFontStyle.Regular);
            m_VideoBoxSize = Gui.VideoBox.ClientSize;

            m_Timer = new Timer { Interval = 25 };
            m_Timer.Tick += TimerOnTick;
            Player.Loaded += OnPlayerLoaded;
            Player.PaintOverlay += OnPaintOverlay;
            Gui.VideoBox.SizeChanged += VideoBoxResize;
            Player.FullScreenMode.Entering += EnteringFullScreenMode;
            Player.FullScreenMode.Entered += EnteredFullScreenMode;
        }

        public override void Destroy()
        {
            Player.FullScreenMode.Entering -= EnteringFullScreenMode;
            Player.FullScreenMode.Entered -= EnteredFullScreenMode;
            Gui.VideoBox.SizeChanged -= VideoBoxResize;
            Player.PaintOverlay -= OnPaintOverlay;
            Player.Loaded -= OnPlayerLoaded;
            m_Timer.Tick -= TimerOnTick;

            m_Text.Dispose();

            base.Destroy();
        }

        private void OnPlayerLoaded(object sender, EventArgs eventArgs)
        {
            DynamicHotkeys.RegisterHotkey(Guid.NewGuid(), "Ctrl+K", () =>
            {
                Settings.ShowOsd = !Settings.ShowOsd;

                if (Player.State == PlayerState.Playing)
                    return;

                Gui.VideoBox.Invalidate();
            });
        }

        private void EnteringFullScreenMode(object sender, EventArgs e)
        {
            m_Text.Hide();
        }

        private void EnteredFullScreenMode(object sender, EventArgs eventArgs)
        {
            UpdateVideoBoxSize();
        }

        private void TimerOnTick(object sender, EventArgs eventArgs)
        {
            UpdateVideoBoxSize();
        }

        private void VideoBoxResize(object sender, EventArgs eventArgs)
        {
            m_Resizing = true;
            m_Timer.Stop();
            m_Timer.Start();
        }

        private void OnPaintOverlay(object sender, EventArgs eventArgs)
        {
            // Warning: This is called from a foreign thread

            if (m_Resizing)
                return;

            if (!Settings.ShowOsd)
            {
                m_Text.Hide();
                return;
            }

            var script = Extension.RenderScript as RenderChainScript;
            var desc = script == null ? GetInternalScalerDesc() : script.Status;
            desc = desc.Trim();

            var descriptions = desc.Split(';')
                .Select(str => str.Trim())
                .Where(str => !string.IsNullOrEmpty(str))
                .ToArray();

            var text = string.Format("Render Chain\r\n    {0}", string.Join("\r\n    ", descriptions));
            var width = m_Text.MeasureWidth(text);
            var lineCount = text.Count(c => c == '\n');

            var height = lineCount*(TEXT_HEIGHT + 1);
            var size = m_VideoBoxSize;
            var location = new Point(size.Width - width - 30, 30);
            const int verticalPadding = 5;
            const int horizontalPadding = 15;
            m_Text.Show(text, location, Color.FromArgb(0xff, 0xbb, 0xcc, 0xdd),
                Color.FromArgb(255*40/100, Color.FromArgb(0xff, 0x00, 0x1f, 0x2f)),
                new Padding(horizontalPadding, verticalPadding, horizontalPadding, height + verticalPadding*2));
        }

        private void UpdateVideoBoxSize()
        {
            m_Timer.Stop();
            m_VideoBoxSize = Gui.VideoBox.ClientSize;
            m_Resizing = false;
        }

        private static string GetInternalScalerDesc()
        {
            var sourceFilter = new SourceFilter();
            sourceFilter.SetSize(Renderer.TargetSize);
            sourceFilter.Initialize();
            return sourceFilter.Status();
        }
    }

    public class ScriptChainOsdPainterSettings
    {
        public ScriptChainOsdPainterSettings()
        {
            ShowOsd = false;
        }

        public bool ShowOsd { get; set; }
    }
}
