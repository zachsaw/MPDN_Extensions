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
using System.Linq;
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.Framework.Config;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Mpdn.Jinc2D
    {
        public partial class Jinc2DConfigDialog : Jinc2DConfigDialogBase
        {
            private const ScalerTaps MIN_SCALER_TAPS = ScalerTaps.Four;
            private const ScalerTaps MAX_SCALER_TAPS = ScalerTaps.Eight;

            public Jinc2DConfigDialog()
            {
                InitializeComponent();

                var vals = Enum.GetValues(typeof (ScalerTaps))
                    .Cast<ScalerTaps>()
                    .Where(val => (int) val >= (int) MIN_SCALER_TAPS && (int) val <= (int) MAX_SCALER_TAPS)
                    .Select(val => new ComboBoxItem(val.ToDescription(), val));

                comboBoxTapCount.Items.AddRange(vals.ToArray<object>());
            }

            protected override void LoadSettings()
            {
                comboBoxTapCount.SelectedIndex = (int) Settings.TapCount - (int) MIN_SCALER_TAPS;
                checkBoxAntiRinging.Checked = Settings.AntiRingingEnabled;
                AntiRingingStrengthSetter.Value = (decimal) Settings.AntiRingingStrength;
            }

            protected override void SaveSettings()
            {
                Settings.TapCount = ((ComboBoxItem) comboBoxTapCount.SelectedItem).Value;
                Settings.AntiRingingEnabled = checkBoxAntiRinging.Checked;
                Settings.AntiRingingStrength = (float) AntiRingingStrengthSetter.Value;
            }

            private class ComboBoxItem
            {
                private readonly string m_Text;
                public readonly ScalerTaps Value;

                public ComboBoxItem(string text, ScalerTaps value)
                {
                    m_Text = text;
                    Value = value;
                }

                public override string ToString()
                {
                    return m_Text;
                }
            }
        }

        public class Jinc2DConfigDialogBase : ScriptConfigDialog<Jinc2D>
        {
        }
    }
}
