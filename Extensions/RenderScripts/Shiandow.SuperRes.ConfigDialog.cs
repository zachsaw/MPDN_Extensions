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
using System.Linq;
using Mpdn.Config;
using Mpdn.RenderScript.Mpdn.Presets;

namespace Mpdn.RenderScript
{
    namespace Shiandow.SuperRes
    {
        public partial class SuperResConfigDialog : SuperResConfigDialogBase
        {
            protected SuperResPreset selectedPreset = null;

            public SuperResConfigDialog()
            {
                InitializeComponent();
            }

            protected override void LoadSettings()
            {
                PrescalerBox.DataSource = Settings.Options;
                PrescalerBox.DisplayMember = "Name";
                PrescalerBox.SelectedIndex = Settings.SelectedIndex;
                
                UpdateGui();
            }

            protected void LoadOption()
            {
                var option = selectedPreset;

                PassesSetter.Value = (Decimal)option.Passes;
                StrengthSetter.Value = (Decimal)option.Strength;
                SharpnessSetter.Value = (Decimal)option.Sharpness;
                AntiAliasingSetter.Value = (Decimal)option.AntiAliasing;
                AntiRingingSetter.Value = (Decimal)option.AntiRinging;
                FastBox.Checked = option.FastMethod;
            }

            protected override void SaveSettings()
            {
                Settings.SelectedIndex = PrescalerBox.SelectedIndex;
                Settings.Hotkey = HotkeyBox.Text;
                SaveOption();
            }

            protected void SaveOption()
            {
                var option = selectedPreset;
                if (option == null)
                    return;

                option.Passes = (int)PassesSetter.Value;
                option.Strength = (float)StrengthSetter.Value;
                option.Sharpness = (float)SharpnessSetter.Value;
                option.AntiAliasing = (float)AntiAliasingSetter.Value;
                option.AntiRinging = (float)AntiRingingSetter.Value;
                option.FastMethod = FastBox.Checked;
            }

            private void SelectionChanged(object sender, EventArgs e)
            {
                SaveOption();
                selectedPreset = (SuperResPreset)PrescalerBox.SelectedValue;
                LoadOption();
                UpdateGui();
            }

            private void UpdateGui()
            {
                AntiRingingSetter.Enabled = !FastBox.Checked;
                ConfigButton.Enabled =
                    (PrescalerBox.SelectedValue as Preset != null) ? 
                    (PrescalerBox.SelectedValue as Preset).HasConfigDialog() : false;
            }

            private void ConfigButton_Click(object sender, EventArgs e)
            {
                (PrescalerBox.SelectedValue as Preset).ShowConfigDialog(Owner);
            }
        }

        public class SuperResConfigDialogBase : ScriptConfigDialog<SuperRes>
        {
        }
    }
}
