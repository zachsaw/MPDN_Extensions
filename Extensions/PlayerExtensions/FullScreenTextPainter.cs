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
using System.Threading;
using System.Windows.Forms;
using Mpdn.Extensions.Framework;
using Timer = System.Windows.Forms.Timer;

namespace Mpdn.Extensions.PlayerExtensions
{
    public class FullScreenTextPainter : PlayerExtension
    {
        private const int WINDOWED_MODE_SEEKBAR_HEIGHT = 34; // Should probably be added to Player Extension API.
        private const int TEXT_HEIGHT = 20;
        private Timer m_Timer;
        private IText m_Text;
        private volatile bool m_FullScreenMode;
        private volatile bool m_SeekBarHover;
        private long m_Position;
        private long m_Duration;

        public override ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("D24FA2D6-B3BE-40C1-B3A5-25D1639EB994"),
                    Name = "Text Painter",
                    Description = "Paints current media time code on top of seek bar in full screen mode"
                };
            }
        }

        public override void Initialize()
        {
            m_Text = Player.CreateText("Verdana", TEXT_HEIGHT, TextFontStyle.Regular);
            m_Timer = new Timer {Interval = 100};
            m_Timer.Tick += TimerOnTick;
            m_Timer.Start();

            Gui.VideoBox.MouseMove += MouseMove;
            
            Player.PaintOverlay += OnPaintOverlay;
            Player.FullScreenMode.Entered += EnteredFullScreenMode;
            Player.FullScreenMode.Exited += ExitedFullScreenMode;
        }

        public override void Destroy()
        {
            Gui.VideoBox.MouseMove -= MouseMove;
            Player.PaintOverlay -= OnPaintOverlay;
            Player.FullScreenMode.Entered -= EnteredFullScreenMode;
            Player.FullScreenMode.Exited -= ExitedFullScreenMode;

            m_Timer.Dispose();
            m_Text.Dispose();
        }

        private void ExitedFullScreenMode(object sender, EventArgs e)
        {
            m_FullScreenMode = false;
        }

        private void EnteredFullScreenMode(object sender, EventArgs e)
        {
            m_FullScreenMode = true;
        }

        private int SeekBarTop
        {
            get { return m_FullScreenMode ? Gui.FullScreenSeekBarHeight : WINDOWED_MODE_SEEKBAR_HEIGHT - SeekBarBottom; }
        }

        private int SeekBarBottom
        {
            get { return Player.Config.Settings.GeneralSettings.AutoHideControlBar ? 0 : WINDOWED_MODE_SEEKBAR_HEIGHT; }
        }

        private void MouseMove(object sender, MouseEventArgs e)
        {
            m_SeekBarHover = e.Y > Gui.VideoBox.Height - SeekBarTop;
        }

        private void TimerOnTick(object sender, EventArgs eventArgs)
        {
            if (Player.State == PlayerState.Closed)
                return;

            var pos = Gui.VideoBox.PointToClient(Cursor.Position);
            m_SeekBarHover =
                   pos.X >= 0
                && pos.X < Gui.VideoBox.Width
                && pos.Y >=Gui.VideoBox.Height - SeekBarTop
                && pos.Y < Gui.VideoBox.Height + SeekBarBottom;

            AtomicWrite(ref m_Position, Media.Position);
            AtomicWrite(ref m_Duration, Media.Duration);
        }

        private static string GetTimeString(long usec)
        {
            TimeSpan duration = TimeSpan.FromMilliseconds(usec/1000d);
            return duration.ToString(@"hh\:mm\:ss\.fff");
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

            if (m_SeekBarHover && (m_FullScreenMode || !PlayerControl.PlayerSettings.GeneralSettings.ShowStatusBar))
            {
                var position = AtomicRead(m_Position);
                var duration = AtomicRead(m_Duration);
                var text = string.Format("{0} / {1}", GetTimeString(position), GetTimeString(duration));
                var width = m_Text.MeasureWidth(text);
                var size = Gui.VideoBox.Size; // Note: Should be fine, probably
                var seekBarHeight = SeekBarTop;
                const int rightOffset = 12;
                const int bottomOffset = 1;
                var location =
                    new Point(size.Width - width - rightOffset,
                              size.Height - seekBarHeight  - TEXT_HEIGHT - bottomOffset);
                m_Text.Show(text, location, Color.FromArgb(255, 0xBB, 0xBB, 0xBB),
                    Color.FromArgb(255*60/100, Color.Black), new Padding(5, 0, rightOffset, bottomOffset));
            }
            else
            {
                m_Text.Hide();
            }
        }
    }
}
