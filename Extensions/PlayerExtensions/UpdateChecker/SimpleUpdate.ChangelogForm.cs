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
using System.ComponentModel;
using System.Windows.Forms;
using Mpdn.Extensions.Framework;

namespace Mpdn.Extensions.PlayerExtensions.UpdateChecker
{
    public partial class ChangelogForm : Form
    {
        public ChangelogForm(SimpleUpdateForm.UpdateType type, Version version)
        {
            InitializeComponent();
            Icon = Gui.Icon;
            switch (type)
            {
                case SimpleUpdateForm.UpdateType.Player:
                    changeLogWebViewer.SetChangelog(ChangelogHelper.ParseMpdnChangelog(version.ChangelogLines));
                    changeLogWebViewer.BeforeLoadPreviousChangelog +=
                        (sender, args) => { args.ChangelogLines = ChangelogHelper.LoadPreviousMpdnChangelog(); };
                    break;
                case SimpleUpdateForm.UpdateType.Extensions:
                    changeLogWebViewer.SetChangelog(ChangelogHelper.ParseExtensionChangelog(version.ChangelogLines));
                    changeLogWebViewer.BeforeLoadPreviousChangelog +=
                        (sender, args) => { args.ChangelogLines = ChangelogHelper.LoadPreviousExtensionsChangelog(); };
                    break;
                default:
                    throw new InvalidEnumArgumentException("Only takes Player and Extensions");
            }
        }

        private void CloseButtonClick(object sender, EventArgs e)
        {
            Close();
        }
    }
}