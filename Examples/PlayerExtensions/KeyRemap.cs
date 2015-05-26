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

namespace Mpdn.Extensions.PlayerExtensions
{
    public class KeyRemap : IPlayerExtension
    {
        public ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("2FC1DF6F-5A0E-4B95-A364-9DD0D756AA67"),
                    Name = "Key / Mouse Remapper",
                    Description = "Remaps keys and mouse buttons",
                    Copyright = "Copyright Example © 2014-2015. All rights reserved."
                };
            }
        }

        public void Initialize()
        {
            PlayerControl.KeyDown += PlayerKeyDown;
            PlayerControl.MouseClick += PlayerMouseClick;
            PlayerControl.MouseDoubleClick += PlayerMouseDoubleClick;
            PlayerControl.MouseWheel += PlayerMouseWheel;
        }

        public void Destroy()
        {
            PlayerControl.KeyDown -= PlayerKeyDown;
            PlayerControl.MouseClick -= PlayerMouseClick;
            PlayerControl.MouseDoubleClick -= PlayerMouseDoubleClick;
            PlayerControl.MouseWheel -= PlayerMouseWheel;
        }

        public IList<Verb> Verbs
        {
            get { return new Verb[0]; }
        }

        public bool HasConfigDialog()
        {
            return false;
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
            if (PlayerControl.PlayerState == PlayerState.Closed)
                return;

            var pos = PlayerControl.MediaPosition;
            pos += e.InputArgs.Delta*1000000/40;
            pos = Math.Max(pos, 0);
            PlayerControl.SeekMedia(pos);
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
            if (PlayerControl.InFullScreenMode)
            {
                PlayerControl.GoWindowed();
            }
            else
            {
                PlayerControl.GoFullScreen();
            }
        }
    }
}
