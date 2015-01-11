using System;
using System.Collections.Generic;
using System.Windows.Forms;

namespace Mpdn.PlayerExtensions.Example
{
    public class MouseControl : IPlayerExtension
    {
        private IPlayerControl m_PlayerControl;
        
        public ExtensionDescriptor Descriptor
        {
            get
            {
                return new ExtensionDescriptor
                {
                    Guid = new Guid("DCF7797B-9D36-41F3-B28C-0A92793B94F5"),
                    Name = "MouseControl",
                    Description = "Use mousewheel to seek & forward/back buttons to navigate playlist/folder",
                    Copyright = "Copyright Example © 2015. All rights reserved."
                };
            }
        }

        public void Initialize(IPlayerControl playerControl)
        {
            m_PlayerControl = playerControl;
            m_PlayerControl.MouseClick += PlayerMouseClick;
            m_PlayerControl.MouseWheel += PlayerMouseWheel;
        }

        public void Destroy()
        {
            m_PlayerControl.MouseClick -= PlayerMouseClick;
            m_PlayerControl.MouseWheel -= PlayerMouseWheel;
        }

        public IList<Verb> Verbs
        {
            get { return new Verb[0]; }
        }

        private void PlayerMouseWheel(object sender, PlayerControlEventArgs<MouseEventArgs> e)
        {
            var pos = m_PlayerControl.MediaPosition;
            pos += e.InputArgs.Delta*1000000/40;
            pos = Math.Max(pos, 0);
            m_PlayerControl.SeekMedia(pos);
            e.Handled = true;
        }
                
        private void PlayerMouseClick(object sender, PlayerControlEventArgs<MouseEventArgs> e)
        {
            if (PlaylistForm.PlaylistCount <= 1)
            {
                if (e.InputArgs.Button == MouseButtons.XButton2)
                {
                    SendKeys.Send("^{PGDN}");
                }
                else if (e.InputArgs.Button == MouseButtons.XButton1)
                {
                    SendKeys.Send("^{PGUP}");
                }
            }
            else
            {
                if (e.InputArgs.Button == MouseButtons.XButton2)
                {
                    SendKeys.Send("^%n");
                }
                else if (e.InputArgs.Button == MouseButtons.XButton1)
                {
                    SendKeys.Send("^%b");
                }
            }
        }
    }
}
