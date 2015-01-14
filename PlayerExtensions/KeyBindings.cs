using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Forms;

namespace Mpdn.PlayerExtensions.Example
{
    public class KeyBindings : IPlayerExtension
    {
        private IPlayerControl m_PlayerControl;

        public ExtensionDescriptor Descriptor
        {
            get
            {
                return new ExtensionDescriptor
                {
                    Guid = new Guid("E3E54699-0B2B-4B1B-8F6B-4739273670CD"),
                    Name = "Key Bindings",
                    Description = "Extra shortcut key bindings",
                    Copyright = "Copyright Example © 2014-2015. All rights reserved."
                };
            }
        }

        public void Initialize(IPlayerControl playerControl)
        {
            m_PlayerControl = playerControl;
            m_PlayerControl.KeyDown += PlayerKeyDown;
        }

        public void Destroy()
        {
            m_PlayerControl.KeyDown -= PlayerKeyDown;
        }

        public IList<Verb> Verbs
        {
            get { return new Verb[0]; }
        }

        public bool ShowConfigDialog(IWin32Window owner)
        {
            return false;
        }

        private void PlayerKeyDown(object sender, PlayerControlEventArgs<KeyEventArgs> e)
        {
            switch (e.InputArgs.KeyData)
            {
                case Keys.Control | Keys.Enter:
                    e.OutputArgs = new KeyEventArgs(Keys.Alt | Keys.Enter);
                    break;
                case Keys.Escape:
                    if (m_PlayerControl.InFullScreenMode)
                    {
                        m_PlayerControl.GoWindowed();
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
            if (m_PlayerControl.PlayerState == PlayerState.Closed)
                return;

            var activeTrack = m_PlayerControl.ActiveAudioTrack;
            if (activeTrack == null)
                return;

            var tracks = m_PlayerControl.AudioTracks;
            var audioTrack = next
                ? tracks.SkipWhile(track => !track.Equals(activeTrack)).Skip(1).FirstOrDefault()
                : tracks.TakeWhile(track => !track.Equals(activeTrack)).LastOrDefault();
            if (audioTrack != null)
            {
                m_PlayerControl.SelectAudioTrack(audioTrack);
            }
        }

        private void SelectSubtitleTrack(bool next)
        {
            if (m_PlayerControl.PlayerState == PlayerState.Closed)
                return;

            var activeTrack = m_PlayerControl.ActiveSubtitleTrack;
            if (activeTrack == null)
                return;

            var tracks = m_PlayerControl.SubtitleTracks;
            var subtitleTrack = next
                ? tracks.SkipWhile(track => !track.Equals(activeTrack)).Skip(1).FirstOrDefault()
                : tracks.TakeWhile(track => !track.Equals(activeTrack)).LastOrDefault();
            if (subtitleTrack != null)
            {
                m_PlayerControl.SelectSubtitleTrack(subtitleTrack);
            }
        }

        private void ToggleMode()
        {
            if (m_PlayerControl.InFullScreenMode)
            {
                m_PlayerControl.GoWindowed();
            }
            else
            {
                m_PlayerControl.GoFullScreen();
            }
        }
    }
}
