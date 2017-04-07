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
using System.IO;
using System.Linq;
using Mpdn.Extensions.PlayerExtensions.Playlist;

namespace Mpdn.Extensions.PlayerExtensions
{
    public class OnScreenDisplay : PlayerExtension<OnScreenDisplaySettings, OnScreenDisplayConfigDialog>
    {
        private const int OSD_DURATION = 10 * 3;

        private Playlist.Playlist m_PlaylistInstance;
        private static readonly Guid s_PlaylistGuid = new Guid("A1997E34-D67B-43BB-8FE6-55A71AE7184B");

        private Timer m_Timer;

        private IText m_titleText;
        private Point m_titleTextPos = new Point(20, 20);

        private IText m_durationText;
        private Point m_durationTextPos = new Point(20, 65);

        private IText m_chapterText;
        private Point m_chapterTextPos = new Point(20, 105);

        private IText m_CloseBtn;
        private Point m_closeBtnPos;
        private Padding m_closeBtnPadding = new Padding(37, 10, 34, 12);

        private IText m_NextBtn;
        private Point m_NextBtnPos;

        private IText m_PrevBtn;
        private Point m_PrevBtnPos;

        private Padding m_PrevNextBtnPadding = new Padding(17, 30, 21, 36);

        private volatile bool m_VideoFrameHover;
        private volatile bool m_CloseBtnHover;
        private volatile bool m_PrevBtnHover;
        private volatile bool m_NextBtnHover;

        private volatile bool m_showOsd = true;

        private long m_Position;
        private long m_Duration;

        private int m_osdDuration = 0;
        private int m_osdAlpha = 255;

        private Point m_lastMousePos;

        private Boolean m_isEnabled;

        public override ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("9E0BE305-0A23-4FE9-BD28-0CE05D871518"),
                    Name = "On-screen display",
                    Description = "Displays convenient controls and select information on an overlay"
                };
            }
        }

        public override void Initialize()
        {
            base.Initialize();

            m_isEnabled = Settings.EnableOnScreenDisplay;

            if (!m_isEnabled)
            {
                return;
            }

            m_titleText = Player.CreateText("Tahoma", 30, TextFontStyle.Bold);
            m_durationText = Player.CreateText("Tahoma", 28, TextFontStyle.Regular);
            m_chapterText = Player.CreateText("Tahoma", 25, TextFontStyle.Regular);

            m_CloseBtn = Player.CreateText("Verdana", 45, TextFontStyle.Regular);
            m_NextBtn = Player.CreateText("Verdana", 45, TextFontStyle.Regular);
            m_PrevBtn = Player.CreateText("Verdana", 45, TextFontStyle.Regular);

            m_Timer = new Timer { Interval = 100 };
            m_Timer.Tick += TimerOnTick;
            m_Timer.Start();

            Gui.VideoBox.MouseMove += MouseMove;
            Gui.VideoBox.MouseClick += MouseClick;

            Media.Loaded += OnMediaLoaded;
            Player.PaintOverlay += OnPaintOverlay;
        }

        public override void Destroy()
        {
            base.Destroy();

            if (!m_isEnabled)
            {
                return;
            }

            m_titleText.Dispose();
            m_durationText.Dispose();
            m_chapterText.Dispose();

            m_CloseBtn.Dispose();
            m_NextBtn.Dispose();
            m_PrevBtn.Dispose();

            m_Timer.Dispose();

            Gui.VideoBox.MouseMove -= MouseMove;
            Gui.VideoBox.MouseClick -= MouseClick;

            Media.Loaded -= OnMediaLoaded;
            Player.PaintOverlay -= OnPaintOverlay;
        }

        private PlaylistForm PlaylistForm
        {
            get
            {
                if (m_PlaylistInstance != null)
                    return m_PlaylistInstance.GetPlaylistForm;

                var playlist = Extension.PlayerExtensions.FirstOrDefault(t => t.Descriptor.Guid == s_PlaylistGuid);

                if (playlist == null) {
                    throw new Exception("OnScreenDisplay requires the Playlist extension for previous and next buttons");
                }

                m_PlaylistInstance = (Playlist.Playlist)playlist;
                return m_PlaylistInstance.GetPlaylistForm;
            }
        }

        private Chapter GetCurrentChapter()
        {
            if (Player.State == PlayerState.Closed ||
                Media.Chapters.Count == 0) return null;

            var chapters = Media.Chapters.OrderBy(chapter => chapter.Position);
            var pos = Media.Position;
            return chapters.TakeWhile(chapter => chapter.Position < Math.Max(pos - 1000000, 0)).LastOrDefault();
        }

        private bool IsHoveringOverControls()
        {
            if (m_CloseBtnHover || m_NextBtnHover || m_PrevBtnHover)
                return true;

            return false;
        }

        private void ShowOsd()
        {
            m_showOsd = true;
            m_osdAlpha = 255;
            m_osdDuration = 0;
        }

        private void OnMediaLoaded(object sender, EventArgs e)
        {
            ShowOsd();
        }

        private void MouseClick(object sender, MouseEventArgs e)
        {
            if (m_CloseBtnHover && e.Button == MouseButtons.Left)
            {
                Player.ActiveForm.Close();
            }

            if (m_PrevBtnHover && e.Button == MouseButtons.Left)
            {
                GuiThread.DoAsync(new Action(() => {
                    PlaylistForm.PlayPrevious();
                }));
            }

            if (m_NextBtnHover && e.Button == MouseButtons.Left)
            {
                GuiThread.DoAsync(new Action(() => {
                    PlaylistForm.PlayNext();
                }));
            }
        }

        private void MouseMove(object sender, MouseEventArgs e)
        {
            m_VideoFrameHover = e.X >= 0 && e.X < Gui.VideoBox.Width &&
                                e.Y >= 0 && e.Y <= Gui.VideoBox.Height;

            m_CloseBtnHover = e.X >= m_closeBtnPos.X - m_closeBtnPadding.Left && e.X <= m_closeBtnPos.X + m_closeBtnPadding.Horizontal &&
                              e.Y >= m_closeBtnPos.Y - m_closeBtnPadding.Top && e.Y <= m_closeBtnPos.Y + m_closeBtnPadding.Vertical + 35;

            m_NextBtnHover = e.X >= m_NextBtnPos.X - m_PrevNextBtnPadding.Left && e.X <= m_NextBtnPos.X + m_PrevNextBtnPadding.Horizontal + 7 &&
                             e.Y >= m_NextBtnPos.Y - m_PrevNextBtnPadding.Top && e.Y <= m_NextBtnPos.Y + m_PrevNextBtnPadding.Vertical + 16;

            m_PrevBtnHover = e.X >= m_PrevBtnPos.X - m_PrevNextBtnPadding.Left && e.X <= m_PrevBtnPos.X + m_PrevNextBtnPadding.Horizontal + 7 &&
                             e.Y >= m_PrevBtnPos.Y - m_PrevNextBtnPadding.Top && e.Y <= m_PrevBtnPos.Y + m_PrevNextBtnPadding.Vertical + 16;

            if (!m_showOsd && e.Location != m_lastMousePos)
            {
                m_lastMousePos = e.Location;
                ShowOsd();
            }
        }

        private void TimerOnTick(object sender, EventArgs eventArgs)
        {
            if (Player.State == PlayerState.Closed)
                return;

            if (m_osdDuration < OSD_DURATION)
            {
                m_osdDuration++;
            }
            else
            {
                m_showOsd = false;
            }

            AtomicWrite(ref m_Position, Media.Position);
            AtomicWrite(ref m_Duration, Media.Duration);
        }

        private static string GetTimeString(long usec)
        {
            TimeSpan duration = TimeSpan.FromMilliseconds(usec / 1000d);
            return duration.ToString(@"hh\:mm\:ss");
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
            m_closeBtnPos = new Point(Gui.VideoBox.Width - 60, 5);
            m_PrevBtnPos = new Point(15, Gui.VideoBox.Height / 2 - 35);
            m_NextBtnPos = new Point(Gui.VideoBox.Width - 45, Gui.VideoBox.Height / 2 - 35);

            var position = AtomicRead(m_Position);
            var duration = AtomicRead(m_Duration);
            var text = string.Format("{0} / {1}", GetTimeString(position), GetTimeString(duration));

            if ((m_VideoFrameHover && m_showOsd) || IsHoveringOverControls())
            {
                if (m_CloseBtnHover)
                    m_CloseBtn.Show("X", m_closeBtnPos, Color.FromArgb(m_osdAlpha, 255, 255, 255), Color.FromArgb(m_osdAlpha, Color.DarkRed), m_closeBtnPadding);
                else
                    m_CloseBtn.Show("X", m_closeBtnPos, Color.FromArgb(m_osdAlpha, 255, 255, 255), Color.FromArgb(m_osdAlpha, Color.Red), m_closeBtnPadding);

                if (m_NextBtnHover)
                    m_NextBtn.Show(">", m_NextBtnPos, Color.FromArgb(m_osdAlpha, 255, 255, 255), Color.FromArgb(m_osdAlpha, 50, 50, 50), m_PrevNextBtnPadding);
                else
                    m_NextBtn.Show(">", m_NextBtnPos, Color.FromArgb(m_osdAlpha, 255, 255, 255), Color.FromArgb(m_osdAlpha, Color.Black), m_PrevNextBtnPadding);

                if (m_PrevBtnHover)
                    m_PrevBtn.Show("<", m_PrevBtnPos, Color.FromArgb(m_osdAlpha, 255, 255, 255), Color.FromArgb(m_osdAlpha, 50, 50, 50), m_PrevNextBtnPadding);
                else
                    m_PrevBtn.Show("<", m_PrevBtnPos, Color.FromArgb(m_osdAlpha, 255, 255, 255), Color.FromArgb(m_osdAlpha, Color.Black), m_PrevNextBtnPadding);

                m_titleText.Show(Path.GetFileNameWithoutExtension(Media.FilePath), m_titleTextPos, Color.FromArgb(m_osdAlpha, 255, 255, 255));
                m_durationText.Show(text, m_durationTextPos, Color.FromArgb(m_osdAlpha, 255, 255, 255));

                if (GetCurrentChapter() != null)
                    m_chapterText.Show(GetCurrentChapter().Name, m_chapterTextPos, Color.FromArgb(m_osdAlpha, 255, 255, 255));
            }
            else
            {
                if (m_osdAlpha > 0)
                {
                    m_osdAlpha -= 5;

                    m_CloseBtn.Show("X", m_closeBtnPos, Color.FromArgb(m_osdAlpha, 255, 255, 255), Color.FromArgb(m_osdAlpha, Color.Red), m_closeBtnPadding);
                    m_PrevBtn.Show("<", m_PrevBtnPos, Color.FromArgb(m_osdAlpha, 255, 255, 255), Color.FromArgb(m_osdAlpha, Color.Black), m_PrevNextBtnPadding);
                    m_NextBtn.Show(">", m_NextBtnPos, Color.FromArgb(m_osdAlpha, 255, 255, 255), Color.FromArgb(m_osdAlpha, Color.Black), m_PrevNextBtnPadding);

                    m_titleText.Show(Path.GetFileNameWithoutExtension(Media.FilePath), m_titleTextPos, Color.FromArgb(m_osdAlpha, 255, 255, 255));
                    m_durationText.Show(text, m_durationTextPos, Color.FromArgb(m_osdAlpha, 255, 255, 255));

                    if (GetCurrentChapter() != null)
                        m_chapterText.Show(GetCurrentChapter().Name, m_chapterTextPos, Color.FromArgb(m_osdAlpha, 255, 255, 255));
                }
                else
                {
                    m_CloseBtn.Hide();
                    m_PrevBtn.Hide();
                    m_NextBtn.Hide();

                    m_titleText.Hide();
                    m_durationText.Hide();
                    m_chapterText.Hide();
                }
            }
        }
    }

    public class OnScreenDisplaySettings
    {
        public OnScreenDisplaySettings()
        {
            EnableOnScreenDisplay = false;
        }

        public bool EnableOnScreenDisplay { get; set; }
    }
}
