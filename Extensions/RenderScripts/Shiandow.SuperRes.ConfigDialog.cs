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
using Mpdn.Config;

namespace Mpdn.RenderScript
{
    namespace Shiandow.SuperRes
    {
        public partial class SuperResConfigDialog : SuperResConfigDialogBase
        {
            public SuperResConfigDialog()
            {
                InitializeComponent();
            }

            protected override void LoadSettings()
            {
                PassesSetter.Value = (Decimal)Settings.Passes;
                StrengthSetter.Value = (Decimal)Settings.Strength;
                SharpnessSetter.Value = (Decimal)Settings.Sharpness;
                AntiAliasingSetter.Value = (Decimal)Settings.AntiAliasing;
                AntiRingingSetter.Value = (Decimal)Settings.AntiRinging;
                DoublerBox.SelectedIndex = (int)Settings.ImageDoubler;
                FastBox.Checked = Settings.FastMethod;
                UpdateGui();
            }

            protected override void SaveSettings()
            {
                Settings.Passes = (int)PassesSetter.Value;
                Settings.Strength = (float)StrengthSetter.Value;
                Settings.Sharpness = (float)SharpnessSetter.Value;
                Settings.AntiAliasing = (float)AntiAliasingSetter.Value;
                Settings.AntiRinging = (float)AntiRingingSetter.Value;
                Settings.ImageDoubler = (SuperRes.SuperResDoubler)DoublerBox.SelectedIndex;
                Settings.FastMethod = FastBox.Checked;
            }

            private void ValueChanged(object sender, EventArgs e)
            {
                UpdateGui();
            }

            private void UpdateGui()
            {
                AntiRingingSetter.Enabled = !FastBox.Checked;
            }
        }

        public class SuperResConfigDialogBase : ScriptConfigDialog<SuperRes>
        {
        }
    }
}
