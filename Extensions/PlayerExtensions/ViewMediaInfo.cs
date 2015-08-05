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
using Mpdn.Extensions.Framework;

namespace Mpdn.Extensions.PlayerExtensions
{
    public class ViewMediaInfo : PlayerExtension
    {
        private readonly PlayerMenuItem m_MenuItem = new PlayerMenuItem(initiallyDisabled: true);

        public override ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("6FD61379-FF5D-4143-8A7B-97516BB7822F"),
                    Name = "Media Info",
                    Description = "View media info of current file"
                };
            }
        }

        public override IList<Verb> Verbs
        {
            get
            {
                return new[]
                {
                    new Verb(Category.View, string.Empty, "Media Info...", "Ctrl+Shift+I", string.Empty, ShowMediaInfoDialog, m_MenuItem)
                };
            }
        }

        public override void Initialize()
        {
            base.Initialize();
            Player.StateChanged += PlayerStateChanged;
        }

        public override void Destroy()
        {
            Player.StateChanged -= PlayerStateChanged;
            base.Destroy();
        }

        private void PlayerStateChanged(object sender, PlayerStateEventArgs e)
        {
            m_MenuItem.Enabled = e.NewState != PlayerState.Closed;
        }

        private static void ShowMediaInfoDialog()
        {
            if (Player.State == PlayerState.Closed)
                return;

            Player.FullScreenMode.Active = false;

            using (var form = new ViewMediaInfoForm(Media.FilePath))
            {
                form.ShowDialog(Gui.VideoBox);
            }
        }
    }
}
