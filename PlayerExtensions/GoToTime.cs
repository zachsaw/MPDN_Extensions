using System;
using System.Collections.Generic;
using System.Windows.Forms;

namespace Mpdn.PlayerExtensions.ZachSaw
{
    public class GoToTime : IPlayerExtension
    {
        private IPlayerControl m_PlayerControl;

        public ExtensionDescriptor Descriptor
        {
            get
            {
                return new ExtensionDescriptor
                {
                    Guid = new Guid("7C3BA1E2-EE7B-47D2-B174-6AE76D65EC04"),
                    Name = "Go To Time",
                    Description = "Jump to a specified timecode in media",
                    Copyright = "Copyright Zach Saw © 2014. All rights reserved."
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
            get
            {
                return new[]
                {
                    new Verb(Category.Play, string.Empty, "Go To...", "Ctrl+G", string.Empty, GotoPosition)
                };
            }
        }

        private void GotoPosition()
        {
            using (var form = new GoToTimeForm())
            {
                if (form.ShowDialog(m_PlayerControl.Form) != DialogResult.OK)
                    return;

                if (m_PlayerControl.PlayerState == PlayerState.Closed)
                    return;

                if (m_PlayerControl.PlayerState == PlayerState.Stopped)
                {
                    m_PlayerControl.PauseMedia(false);
                }

                m_PlayerControl.SeekMedia(form.Position * 1000);
            }
        }

        private void PlayerKeyDown(object sender, PlayerKeyEventArgs e)
        {
            switch (e.Key.KeyData)
            {
                case Keys.Control | Keys.G:
                    GotoPosition();
                    break;
            }
        }
    }
}
