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
using System.IO;
using System.Linq;
using System.Windows.Forms;
using Microsoft.WindowsAPICodePack.Taskbar;
using Mpdn.Extensions.Framework;

namespace Mpdn.Extensions.PlayerExtensions
{
    public class TaskbarEnhancer : PlayerExtension
    {
        private const string TEXT_PAUSE = "Pause";
        private const string TEXT_PLAY = "Play";
        private const string TEXT_STOP = "Stop";
        private const string TEXT_FORWARD = "Forward 5s";
        private const string TEXT_BACKWARD = "Backward 5s";
        private const string TEXT_NEXT = "Next";
        private const string TEXT_PREV = "Previous";

        private static readonly string s_IconPath = Path.Combine(PathHelper.ExtensionsPath,
            @"PlayerExtensions\Images\TaskbarEnhancer\");

        private static readonly Icon s_PlayIcon = new Icon(Path.Combine(s_IconPath, "Play.ico"));
        private static readonly Icon s_PauseIcon = new Icon(Path.Combine(s_IconPath, "Pause.ico"));
        private static readonly Icon s_StopIcon = new Icon(Path.Combine(s_IconPath, "Stop.ico"));
        private static readonly Icon s_ForwardIcon = new Icon(Path.Combine(s_IconPath, "Forward.ico"));
        private static readonly Icon s_BackwardIcon = new Icon(Path.Combine(s_IconPath, "Rewind.ico"));
        private static readonly Icon s_NextIcon = new Icon(Path.Combine(s_IconPath, "Next.ico"));
        private static readonly Icon s_PrevIcon = new Icon(Path.Combine(s_IconPath, "Prev.ico"));

        private static ThumbnailToolBarButton s_PlayPauseButton;
        private static ThumbnailToolBarButton s_StopButton;
        private static ThumbnailToolBarButton s_ForwardButton;
        private static ThumbnailToolBarButton s_BackwardButton;
        private static ThumbnailToolBarButton s_NextButton;
        private static ThumbnailToolBarButton s_PrevButton;

        private readonly Guid m_PlaylistGuid = new Guid("A1997E34-D67B-43BB-8FE6-55A71AE7184B");

        private Timer m_UpdateTimer;
        private Playlist.Playlist m_Playlist;
        private IntPtr m_MpdnFormHandle;

        public override ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("51A69EE0-4D5E-41D1-ABB4-BF99D1D502B6"),
                    Name = "Taskbar Enhancer",
                    Description = "Enhances the MPDN taskbar button"
                };
            }
        }

        public override void Initialize()
        {
            base.Initialize();

            m_MpdnFormHandle = PlayerControl.Form.Handle;

            m_UpdateTimer = new Timer();
            m_UpdateTimer.Tick += UpdateTimerTick;
            m_Playlist = GetPlaylistInstance();

            PlayerControl.PlayerStateChanged += PlayerStateChanged;
            PlayerControl.MediaLoaded += MediaLoaded;

            CreateToolBarButtons();
        }

        private Playlist.Playlist GetPlaylistInstance()
        {
            return PlayerControl.PlayerExtensions.FirstOrDefault(t => t.Descriptor.Guid == m_PlaylistGuid) 
                as Playlist.Playlist;
        }

        public override void Destroy()
        {
            PlayerControl.MediaLoaded -= MediaLoaded;
            PlayerControl.PlayerStateChanged -= PlayerStateChanged;

            DisposeHelper.Dispose(ref m_UpdateTimer);

            base.Destroy();
        }

        public override IList<Verb> Verbs
        {
            get { return new Verb[0]; }
        }

        private void MediaLoaded(object sender, EventArgs eventArgs)
        {
            Taskbar.SetProgressState(TaskbarProgressBarState.Paused);
            m_UpdateTimer.Start();
        }

        private void PlayerStateChanged(object sender, PlayerStateEventArgs e)
        {
            switch (e.NewState)
            {
                case PlayerState.Closed:
                    Taskbar.SetProgressState(TaskbarProgressBarState.NoProgress);
                    m_UpdateTimer.Stop();
                    s_PlayPauseButton.Icon = s_PlayIcon;
                    s_PlayPauseButton.Tooltip = TEXT_PLAY;
                    Taskbar.SetOverlayIcon(m_MpdnFormHandle, null, "");
                    break;
                case PlayerState.Stopped:
                    Taskbar.SetProgressState(TaskbarProgressBarState.NoProgress);
                    m_UpdateTimer.Start();
                    s_PlayPauseButton.Icon = s_PlayIcon;
                    s_PlayPauseButton.Tooltip = TEXT_PLAY;
                    Taskbar.SetOverlayIcon(m_MpdnFormHandle, null, "");
                    break;
                case PlayerState.Playing:
                    Taskbar.SetProgressState(TaskbarProgressBarState.Normal);
                    m_UpdateTimer.Start();
                    s_PlayPauseButton.Icon = s_PauseIcon;
                    s_PlayPauseButton.Tooltip = TEXT_PAUSE;
                    Taskbar.SetOverlayIcon(m_MpdnFormHandle, s_PlayIcon, "Playing");
                    break;
                case PlayerState.Paused:
                    Taskbar.SetProgressState(TaskbarProgressBarState.Paused);
                    m_UpdateTimer.Start();
                    s_PlayPauseButton.Icon = s_PlayIcon;
                    s_PlayPauseButton.Tooltip = TEXT_PLAY;
                    Taskbar.SetOverlayIcon(m_MpdnFormHandle, s_PauseIcon, "Paused");
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }


        private void UpdateTimerTick(object sender, EventArgs eventArgs)
        {
            if (PlayerControl.PlayerState == PlayerState.Closed || PlayerControl.PlayerState == PlayerState.Stopped)
                return;

            var duration = Math.Max(1, PlayerControl.MediaDuration);
            Taskbar.SetProgressValue((int) (PlayerControl.MediaPosition*1000/duration), 1000);
        }

        private static TaskbarManager Taskbar
        {
            get { return TaskbarManager.Instance; }
        }

        private void CreateToolBarButtons()
        {
            var buttons = new List<ThumbnailToolBarButton>();

            s_PlayPauseButton = new ThumbnailToolBarButton(s_PlayIcon, TEXT_PLAY);
            s_PlayPauseButton.Click += PlayPauseClick;

            s_StopButton = new ThumbnailToolBarButton(s_StopIcon, TEXT_STOP);
            s_StopButton.Click += StopClick;

            s_ForwardButton = new ThumbnailToolBarButton(s_ForwardIcon, TEXT_FORWARD);
            s_ForwardButton.Click += ForwardClick;

            s_BackwardButton = new ThumbnailToolBarButton(s_BackwardIcon, TEXT_BACKWARD);
            s_BackwardButton.Click += BackwardClick;

            buttons.Add(s_BackwardButton);
            buttons.Add(s_PlayPauseButton);
            buttons.Add(s_StopButton);
            buttons.Add(s_ForwardButton);

            if (m_Playlist != null)
            {
                s_NextButton = new ThumbnailToolBarButton(s_NextIcon, TEXT_NEXT);
                s_NextButton.Click += NextClick;

                s_PrevButton = new ThumbnailToolBarButton(s_PrevIcon, TEXT_PREV);
                s_PrevButton.Click += PrevClick;

                buttons.Add(s_NextButton);
                buttons.Insert(0, s_PrevButton);
            }

            Taskbar.ThumbnailToolBars.AddButtons(m_MpdnFormHandle, buttons.ToArray());
        }

        private void PrevClick(object sender, ThumbnailButtonClickedEventArgs e)
        {
            if (m_Playlist == null) 
                return;

            m_Playlist.GetPlaylistForm.PlayPrevious();
        }

        private void NextClick(object sender, ThumbnailButtonClickedEventArgs e)
        {
            if (m_Playlist == null)
                return;

            m_Playlist.GetPlaylistForm.PlayNext();
        }

        private static void BackwardClick(object sender, ThumbnailButtonClickedEventArgs e)
        {
            Jump(-5);
        }

        private static void ForwardClick(object sender, ThumbnailButtonClickedEventArgs e)
        {
            Jump(5);
        }   

        private static void StopClick(object sender, ThumbnailButtonClickedEventArgs e)
        {
            switch (PlayerControl.PlayerState)
            {
                case PlayerState.Paused:
                case PlayerState.Playing:
                    PlayerControl.StopMedia();
                    break;
            }
        }

        private static void PlayPauseClick(object sender, ThumbnailButtonClickedEventArgs e)
        {
            switch (PlayerControl.PlayerState)
            {
                case PlayerState.Closed:
                    return;
                case PlayerState.Stopped:
                case PlayerState.Paused:
                    PlayerControl.PlayMedia();
                    s_PlayPauseButton.Icon = s_PauseIcon;
                    s_PlayPauseButton.Tooltip = TEXT_PAUSE;
                    return;
                case PlayerState.Playing:
                    PlayerControl.PauseMedia();
                    s_PlayPauseButton.Icon = s_PlayIcon;
                    s_PlayPauseButton.Tooltip = TEXT_PLAY;
                    return;
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        private static void Jump(double time)
        {
            if (PlayerControl.PlayerState == PlayerState.Closed)
                return;

            var pos = PlayerControl.MediaPosition;
            var nextPos = pos + (long)Math.Round(time * 1000 * 1000);
            nextPos = Math.Max(0, Math.Min(PlayerControl.MediaDuration, nextPos));
            PlayerControl.SeekMedia(nextPos);
        }
    }
}
