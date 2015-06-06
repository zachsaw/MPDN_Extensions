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

            protected SuperResPreset selectedPreset
            {
                get
                {
                    return (SuperResPreset)Settings.Options.ElementAtOrDefault(selectedIndex ?? -1);
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
                HotkeyBox.Text = Settings.Hotkey;
                
                UpdateGui();
            }

            protected void LoadOption()
            {
                var option = selectedPreset;
                if (option == null)
                    return;

                PassesSetter.Value = option.Passes;
                StrengthSetter.Value = (Decimal)option.Strength;
                SharpnessSetter.Value = (Decimal)option.Sharpness;
                AntiAliasingSetter.Value = (Decimal)option.AntiAliasing;
                AntiRingingSetter.Value = (Decimal)option.AntiRinging;
                SoftnessSetter.Value = (Decimal)option.Softness;
            }

            protected override void SaveSettings()
            {
                Settings.SelectedIndex = PrescalerBox.SelectedIndex;
                Settings.Hotkey = HotkeyBox.Text;
                SaveOption();
            }

            protected void SaveOption()
            {
                SuperResPreset option = selectedPreset;
                if (option == null) 
                    return;

                option.Passes = (int)PassesSetter.Value;
                option.Strength = (float)StrengthSetter.Value;
                option.Sharpness = (float)SharpnessSetter.Value;
                option.AntiAliasing = (float)AntiAliasingSetter.Value;
                option.AntiRinging = (float)AntiRingingSetter.Value;
                option.Softness = (float)SoftnessSetter.Value;
            }

            private void SelectionChanged(object sender, EventArgs e)
            {
                SaveOption();
                selectedIndex = PrescalerBox.SelectedIndex;
                LoadOption();
                UpdateGui();
            }

            private void UpdateGui()
            {
                var preset = PrescalerBox.SelectedValue as Preset;
                ConfigButton.Enabled = (preset != null) && preset.HasConfigDialog();
            }

            private void ConfigButton_Click(object sender, EventArgs e)
            {
                ((Preset) PrescalerBox.SelectedValue).ShowConfigDialog(Owner);
            }
        }

        public class SuperResConfigDialogBase : ScriptConfigDialog<SuperRes>
        {
        }
    }
}
