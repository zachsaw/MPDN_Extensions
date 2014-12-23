using System;
using System.Collections.Generic;
using System.Windows.Forms;

namespace Mpdn.PlayerExtensions.Example
{
    public class KeyRemap : IPlayerExtension
    {
        private IPlayerControl m_PlayerControl;

        public ExtensionDescriptor Descriptor
        {
            get
            {
                return new ExtensionDescriptor
                {
                    Guid = new Guid("2FC1DF6F-5A0E-4B95-A364-9DD0D756AA67"),
                    Name = "Key / Mouse Remapper",
                    Description = "Remaps keys and mouse buttons",
                    Copyright = "Copyright Example © 2014. All rights reserved."
                };
            }
        }

        public void Initialize(IPlayerControl playerControl)
        {
            m_PlayerControl = playerControl;
            m_PlayerControl.KeyDown += PlayerKeyDown;
            m_PlayerControl.MouseClick += PlayerMouseClick;
            m_PlayerControl.MouseDoubleClick += PlayerMouseDoubleClick;
            m_PlayerControl.MouseWheel += PlayerMouseWheel;
        }

        public void Destroy()
        {
            m_PlayerControl.KeyDown -= PlayerKeyDown;
            m_PlayerControl.MouseClick -= PlayerMouseClick;
            m_PlayerControl.MouseDoubleClick -= PlayerMouseDoubleClick;
            m_PlayerControl.MouseWheel -= PlayerMouseWheel;
        }

        public IList<Verb> Verbs
        {
            get { return new Verb[0]; }
        }

        private void PlayerKeyDown(object sender, PlayerControlEventArgs<KeyEventArgs> e)
        {
            switch (e.InputArgs.KeyData)
            {
                case Keys.Control | Keys.Enter:
                    e.OutputArgs = new KeyEventArgs(Keys.Alt | Keys.Enter); // Replace Alt+Enter with Ctrl+Enter
                    break;
                // Uncomment the following to suppress Alt+Enter
                //case Keys.Alt | Keys.Enter:
                //    e.Handled = true; // Suppress original Alt+Enter
                //    break;
            }
        }

        private void PlayerMouseWheel(object sender, PlayerControlEventArgs<MouseEventArgs> e)
        {
            var pos = m_PlayerControl.MediaPosition;
            pos += e.InputArgs.Delta*1000000/40;
            pos = Math.Max(pos, 0);
            m_PlayerControl.SeekMedia(pos);
            e.Handled = true;
        }

        private void PlayerMouseDoubleClick(object sender, PlayerControlEventArgs<MouseEventArgs> e)
        {
            e.Handled = true;
        }

        private void PlayerMouseClick(object sender, PlayerControlEventArgs<MouseEventArgs> e)
        {
            if (e.InputArgs.Button == MouseButtons.Middle)
            {
                ToggleMode();
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
