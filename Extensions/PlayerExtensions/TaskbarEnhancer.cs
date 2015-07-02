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
using System.Windows.Forms;
using Microsoft.WindowsAPICodePack.Taskbar;
using Mpdn.Extensions.Framework;

namespace Mpdn.Extensions.PlayerExtensions
{
    public class TaskbarEnhancer : PlayerExtension
    {
        private Timer m_UpdateTimer;

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

            m_UpdateTimer = new Timer();
            m_UpdateTimer.Tick += UpdateTimerTick;

            PlayerControl.PlayerStateChanged += PlayerStateChanged;
            PlayerControl.MediaLoaded += MediaLoaded;
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
                    break;
                case PlayerState.Stopped:
                    Taskbar.SetProgressState(TaskbarProgressBarState.NoProgress);
                    m_UpdateTimer.Start();
                    break;
                case PlayerState.Playing:
                    Taskbar.SetProgressState(TaskbarProgressBarState.Normal);
                    m_UpdateTimer.Start();
                    break;
                case PlayerState.Paused:
                    Taskbar.SetProgressState(TaskbarProgressBarState.Paused);
                    m_UpdateTimer.Start();
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }


        private void UpdateTimerTick(object sender, EventArgs eventArgs)
        {
            if (PlayerControl.PlayerState == PlayerState.Closed || PlayerControl.PlayerState == PlayerState.Stopped)
                return;

            Taskbar.SetProgressValue(
                (int) (PlayerControl.MediaPosition*1000/PlayerControl.MediaDuration), 1000);
        }

        private static TaskbarManager Taskbar
        {
            get { return TaskbarManager.Instance; }
        }
    }
}
