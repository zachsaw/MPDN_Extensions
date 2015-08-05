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
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.Framework.Config;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Mpdn.EwaScaler
    {
        public partial class EwaScalerConfigDialog : EwaScalerConfigDialogBase
        {
            private const ScalerTaps MIN_SCALER_TAPS = ScalerTaps.Four;
            private const ScalerTaps MAX_SCALER_TAPS = ScalerTaps.Eight;

            private readonly ICustomLinearScaler[] m_Scalers;

            public EwaScalerConfigDialog()
            {
                InitializeComponent();

                var privateScalers = new[] {typeof (EwaScaler.JincScaler)};
                var publicScalers = Extension.Assemblies
                    .SelectMany(a => a.GetTypes())
                    .Where(t =>
                        t.IsPublic && !t.IsAbstract && t.GetConstructor(Type.EmptyTypes) != null &&
                        typeof (ICustomLinearScaler).IsAssignableFrom(t));
                m_Scalers = privateScalers.Concat(publicScalers)
                    .Select(t => (ICustomLinearScaler) Activator.CreateInstance(t))
                    .ToArray();
                var scalers = m_Scalers.Select(s => new ComboBoxItem<ICustomLinearScaler>(s.Name, s));
                comboBoxScaler.Items.AddRange(scalers.ToArray<object>());
            }

            protected override void LoadSettings()
            {
                comboBoxScaler.SelectedIndex = GetSelectedScalerIndex(Settings.Scaler.Guid, m_Scalers);
                comboBoxTapCount.SelectedIndex = (int) Settings.TapCount -
                                                 Math.Max((int) Settings.Scaler.MinTapCount, (int) MIN_SCALER_TAPS);
                checkBoxAntiRinging.Checked = Settings.AntiRingingEnabled;
                setterAntiRingStrength.Value = (decimal) Settings.AntiRingingStrength;
            }

            protected override void SaveSettings()
            {
                var scaler = comboBoxScaler.SelectedItem;
                if (scaler == null)
                    return;

                var tapCount = comboBoxTapCount.SelectedItem;
                if (tapCount == null)
                    return;

                Settings.Scaler = ((ComboBoxItem<ICustomLinearScaler>) scaler).Value;
                Settings.TapCount = ((ComboBoxItem<ScalerTaps>) tapCount).Value;
                Settings.AntiRingingEnabled = checkBoxAntiRinging.Enabled && checkBoxAntiRinging.Checked;
                Settings.AntiRingingStrength = (float) setterAntiRingStrength.Value;
            }

            private int GetSelectedScalerIndex(Guid guid, IList<ICustomLinearScaler> customLinearScalers)
            {
                for (int i = 0; i < customLinearScalers.Count; i++)
                {
                    if (customLinearScalers[i].Guid == guid)
                        return i;
                }
                return -1;
            }

            private void ScalerSelectionChanged(object sender, EventArgs e)
            {
                var scaler = comboBoxScaler.SelectedItem;
                if (scaler == null)
                    return;

                var s = ((ComboBoxItem<ICustomLinearScaler>) scaler).Value;
                var min = Math.Max((int) s.MinTapCount, (int) MIN_SCALER_TAPS);
                var max = Math.Min((int) s.MaxTapCount, (int) MAX_SCALER_TAPS);
                var vals = Enum.GetValues(typeof (ScalerTaps))
                    .Cast<ScalerTaps>()
                    .Where(v => (int) v >= min && (int) v <= max)
                    .Select(v => new ComboBoxItem<ScalerTaps>(v.ToDescription(), v));

                var tapCount = comboBoxTapCount.SelectedItem;
                var taps = tapCount != null ? ((ComboBoxItem<ScalerTaps>)tapCount).Value : ScalerTaps.Four;
                comboBoxTapCount.Items.Clear();
                comboBoxTapCount.Items.AddRange(vals.ToArray<object>());
                comboBoxTapCount.SelectedIndex = Math.Min((int) taps - min, max - min);

                checkBoxAntiRinging.Enabled = s.AllowDeRing;
                setterAntiRingStrength.Enabled = s.AllowDeRing;
                labelStrength.Enabled = s.AllowDeRing;
            }

            private class ComboBoxItem<T>
            {
                private readonly string m_Text;
                public readonly T Value;

                public ComboBoxItem(string text, T value)
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

        public class EwaScalerConfigDialogBase : ScriptConfigDialog<EwaScaler>
        {
        }
    }
}
