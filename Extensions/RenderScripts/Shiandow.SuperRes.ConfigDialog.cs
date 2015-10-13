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
using Mpdn.Extensions.Framework.Chain;
using Mpdn.Extensions.Framework.Config;
using Mpdn.Extensions.Framework.RenderChain;
using Mpdn.Extensions.RenderScripts.Mpdn.ScriptGroup;
using Mpdn.RenderScript;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Shiandow.SuperRes
    {
        public partial class SuperResConfigDialog : SuperResConfigDialogBase
        {
            protected int? selectedIndex;

            protected Preset<IFilter,IRenderScript> selectedPreset
            {
                get
                {
                    return Settings.PrescalerGroup.Options.ElementAtOrDefault(selectedIndex ?? -1);
                }
            }

            public SuperResConfigDialog()
            {
                InitializeComponent();
            }

            protected override void LoadSettings()
            {
                PrescalerBox.DataSource = Settings.PrescalerGroup.Options;
                PrescalerBox.DisplayMember = "Name";
                PrescalerBox.SelectedIndex = Settings.PrescalerGroup.SelectedIndex;

                PassesSetter.Value = Settings.Passes;
                StrengthSetter.Value = (decimal)Settings.Strength;
                SoftnessSetter.Value = (decimal)Settings.Softness;

                HQBox.Checked = Settings.HQdownscaling;
                
                UpdateGui();
            }

            protected override void SaveSettings()
            {
                Settings.PrescalerGroup.SelectedIndex = PrescalerBox.SelectedIndex;

                Settings.Passes = (int)PassesSetter.Value;
                Settings.Strength = (float)StrengthSetter.Value;
                Settings.Softness = (float)SoftnessSetter.Value;

                Settings.HQdownscaling = HQBox.Checked;
            }

            private void SelectionChanged(object sender, EventArgs e)
            {
                UpdateGui();
            }

            private void UpdateGui()
            {
                var preset = PrescalerBox.SelectedValue as Preset<IFilter, IRenderScript>;
                ConfigButton.Enabled = (preset != null) && preset.HasConfigDialog();
            }

            private void ConfigButtonClick(object sender, EventArgs e)
            {
                ((Preset<IFilter, IRenderScript>) PrescalerBox.SelectedValue).ShowConfigDialog(Owner);
            }

            private void ModifyButtonClick(object sender, EventArgs e)
            {
                SaveSettings();

                var groupUi = new ScriptGroupScript { Settings = Settings.PrescalerGroup };
                groupUi.ShowConfigDialog(Owner);

                LoadSettings();
            }
        }

        public class SuperResConfigDialogBase : ScriptConfigDialog<SuperRes>
        {
        }
    }
}
