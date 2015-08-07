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

using System.Linq;
using Mpdn.Extensions.Framework.Config;

namespace Mpdn.Extensions.PlayerExtensions
{
    namespace GitHub
    {
        public partial class FullScreenDisplaySelectorConfigDialog : FullScreenDisplaySelectorConfigBase
        {
            public FullScreenDisplaySelectorConfigDialog()
            {
                InitializeComponent();

                dropdownDisplays.Items.AddRange(Displays.AllDisplays.Select(d => d.DeviceName).Cast<object>().ToArray());
                dropdownDisplays.SelectedIndex = 0;
            }

            protected override void LoadSettings()
            {
                checkBoxEnabled.Checked = Settings.Enabled;
                dropdownDisplays.SelectedIndex = Settings.Monitor;
            }

            protected override void SaveSettings()
            {
                Settings.Enabled = checkBoxEnabled.Checked;
                Settings.Monitor = dropdownDisplays.SelectedIndex;
            }
        }

        public class FullScreenDisplaySelectorConfigBase : ScriptConfigDialog<FullScreenDisplaySelectorSettings>
        {
        }
    }
}