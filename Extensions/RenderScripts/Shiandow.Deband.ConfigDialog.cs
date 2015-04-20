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
    namespace Shiandow.Deband
    {
        public partial class DebandConfigDialog : DebandConfigDialogBase
        {
            public DebandConfigDialog()
            {
                InitializeComponent();
            }

            protected override void LoadSettings()
            {
                MaxBitdepthSetter.Value = (Decimal)Settings.maxbitdepth;
                ThresholdSetter.Value = (Decimal)Settings.threshold;
                MarginSetter.Value = (Decimal)Settings.margin;
                AdvancedBox.Checked = Settings.advancedMode;
                LegacyBox.Checked = Settings.legacyMode;

                UpdateGui();
            }

            protected override void SaveSettings()
            {
                Settings.maxbitdepth = (int)MaxBitdepthSetter.Value;
                Settings.threshold = (float)ThresholdSetter.Value;
                Settings.margin = (float)MarginSetter.Value;
                Settings.advancedMode = AdvancedBox.Checked;
                Settings.legacyMode = LegacyBox.Checked;
            }

            private void ValueChanged(object sender, EventArgs e)
            {
                UpdateGui();
            }

            private void UpdateGui()
            {
                panel1.Enabled = AdvancedBox.Checked;
                //MarginSetter.Enabled = LegacyBox.Checked;
                UpdateText();
            }

            private void UpdateText() 
            {
                var a = (double)ThresholdSetter.Value;
                var b = (double)MarginSetter.Value;
                var x = (10*a + b + Math.Sqrt(36*a*a - 12*a*b + 33*b*b))/16;
                var y = (a + b - x) / b;
                var z = x * y * (3 * y - 2 * y * y) - a;
                MaxErrorLabel.Text = String.Format("(maximum error {0:N2} bit)",LegacyBox.Checked ? z : 0);
            }
        }

        public class DebandConfigDialogBase : ScriptConfigDialog<Deband>
        {
        }
    }
}
