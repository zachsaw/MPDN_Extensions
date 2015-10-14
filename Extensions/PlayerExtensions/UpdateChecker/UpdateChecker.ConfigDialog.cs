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

using System.Windows.Controls;
using Mpdn.Extensions.Framework.Config;

namespace Mpdn.Extensions.PlayerExtensions.UpdateChecker
{
    public partial class UpdateCheckerConfigDialog : UpdateCheckerConfigBase
    {
        public UpdateCheckerConfigDialog()
        {
            InitializeComponent();
            simpleModeTooltip.SetToolTip(simpleModeCheckbox, "This mode disable the display of the changelog and it updates silently the player. Only for installed version.");
            if (!RegistryHelper.IsPlayerInstalled())
            {
                simpleModeCheckbox.Enabled = false;
                simpleModeCheckbox.Text += " (Only for installed version)";
            }

        }

        protected override void LoadSettings()
        {
            checkBoxCheckUpdate.Checked = Settings.CheckForUpdate;
            simpleModeCheckbox.Checked = Settings.UseSimpleUpdate;
        }

        protected override void SaveSettings()
        {
            Settings.CheckForUpdate = checkBoxCheckUpdate.Checked;
            Settings.UseSimpleUpdate = simpleModeCheckbox.Checked;
        }
    }
    public class UpdateCheckerConfigBase : ScriptConfigDialog<UpdateCheckerSettings>
    {
    }

}
