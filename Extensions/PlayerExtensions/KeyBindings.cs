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
using System.Linq;
using System.Windows.Forms;
using Mpdn.Extensions.Framework;

namespace Mpdn.Extensions.PlayerExtensions
{
    public class KeyBindings : PlayerExtension
    {
        public override ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("E3E54699-0B2B-4B1B-8F6B-4739273670CD"),
                    Name = "Key Bindings",
                    Description = "Extra shortcut key bindings"
                };
            }
        }

        public override void Initialize()
        {
            base.Initialize();
            Player.KeyDown += PlayerKeyDown;
        }

        public override void Destroy()
        {
            Player.KeyDown -= PlayerKeyDown;
            base.Destroy();
        }

        private void PlayerKeyDown(object sender, PlayerControlEventArgs<KeyEventArgs> e)
        {
            switch (e.InputArgs.KeyData)
            {
                case Keys.Control | Keys.Enter:
                    e.OutputArgs = new KeyEventArgs(Keys.Alt | Keys.Enter);
                    break;
                case Keys.Escape:
                    if (Player.FullScreenMode.Active)
                    {
                        Player.FullScreenMode.Active = false;
                    }
                    break;
                case Keys.F11:
                    ToggleMode();
                    break;
                case Keys.Shift | Keys.PageDown:
                    SelectAudioTrack(true);
                    break;
                case Keys.Shift | Keys.PageUp:
                    SelectAudioTrack(false);
                    break;
                case Keys.Alt | Keys.Shift | Keys.PageDown:
                    SelectSubtitleTrack(true);
                    break;
                case Keys.Alt | Keys.Shift | Keys.PageUp:
                    SelectSubtitleTrack(false);
                    break;
            }
        }

        private void SelectAudioTrack(bool next)
        {
            if (Player.State == PlayerState.Closed)
                return;

            var activeTrack = Media.AudioTrack;
            if (activeTrack == null)
                return;

            var tracks = Media.AudioTracks;
            var audioTrack = next
                ? tracks.SkipWhile(track => !track.Equals(activeTrack)).Skip(1).FirstOrDefault()
                : tracks.TakeWhile(track => !track.Equals(activeTrack)).LastOrDefault();
            if (audioTrack != null)
            {
                Media.SelectAudioTrack(audioTrack);
            }
        }

        private void SelectSubtitleTrack(bool next)
        {
            if (Player.State == PlayerState.Closed)
                return;

            var activeTrack = Media.SubtitleTrack;
            if (activeTrack == null)
                return;

            var tracks = Media.SubtitleTracks;
            var subtitleTrack = next
                ? tracks.SkipWhile(track => !track.Equals(activeTrack)).Skip(1).FirstOrDefault()
                : tracks.TakeWhile(track => !track.Equals(activeTrack)).LastOrDefault();
            if (subtitleTrack != null)
            {
                Media.SelectSubtitleTrack(subtitleTrack);
            }
        }

        private void ToggleMode()
        {
            Player.FullScreenMode.Active = !Player.FullScreenMode.Active;
        }
    }
}
