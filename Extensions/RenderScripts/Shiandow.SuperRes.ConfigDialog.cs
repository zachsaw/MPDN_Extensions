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

using System;
using System.Linq;
using Mpdn.Extensions.Framework.Config;
using Mpdn.Extensions.Framework.RenderChain;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Shiandow.SuperRes
    {
        public partial class SuperResConfigDialog : SuperResConfigDialogBase
        {
            protected int? selectedIndex;

            protected Preset selectedPreset
            {
                get
                {
                    return Settings.Options.ElementAtOrDefault(selectedIndex ?? -1);
                }
            }

            public SuperResConfigDialog()
            {
                InitializeComponent();
            }

            protected override void LoadSettings()
            {
                PrescalerBox.DataSource = Settings.Options;
                PrescalerBox.DisplayMember = "Name";
                PrescalerBox.SelectedIndex = Settings.SelectedIndex;

                PassesSetter.Value = Settings.Passes;
                StrengthSetter.Value = (Decimal)Settings.Strength;
                SoftnessSetter.Value = (Decimal)Settings.Softness;
                
                UpdateGui();
            }

            protected override void SaveSettings()
            {
                Settings.SelectedIndex = PrescalerBox.SelectedIndex;
                Settings.Passes = (int)PassesSetter.Value;
                Settings.Strength = (float)StrengthSetter.Value;
                Settings.Softness = (float)SoftnessSetter.Value;
            }

            private void SelectionChanged(object sender, EventArgs e)
            {
                UpdateGui();
            }

            private void UpdateGui()
            {
                var preset = PrescalerBox.SelectedValue as Preset;
                ConfigButton.Enabled = (preset != null) && preset.HasConfigDialog();
            }

            private void ConfigButtonClick(object sender, EventArgs e)
            {
                ((Preset) PrescalerBox.SelectedValue).ShowConfigDialog(Owner);
            }

            private void ModifyButtonClick(object sender, EventArgs e)
            {
                var groupUi = new Mpdn.ScriptGroup.ScriptGroupScript() { Settings = Settings };
                groupUi.ShowConfigDialog(Owner);

                PrescalerBox.DataSource = Settings.Options;
                PrescalerBox.SelectedIndex = Settings.SelectedIndex;
            }
        }

        public class SuperResConfigDialogBase : ScriptConfigDialog<SuperRes>
        {
        }
    }
}
