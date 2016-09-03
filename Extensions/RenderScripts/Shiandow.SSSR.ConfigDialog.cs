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
        public partial class SSSRConfigDialog : SSSRConfigDialogBase
        {
            protected int? selectedIndex;

            protected Preset<ITextureFilter,IRenderScript> selectedPreset
            {
                get
                {
                    return Settings.PrescalerGroup.Options.ElementAtOrDefault(selectedIndex ?? -1);
                }
            }

            public SSSRConfigDialog()
            {
                InitializeComponent();
            }

            private void Initialize()
            {
                PrescalerBox.DataSource = Settings.PrescalerGroup.Options;
                PrescalerBox.DisplayMember = "Name";
                PrescalerBox.SelectedIndex = Math.Min(Settings.PrescalerGroup.SelectedIndex,
                    PrescalerBox.Items.Count - 1);
                ModeBox.DataSource = Enum.GetValues(typeof(SSSRMode))
                    .Cast<SSSRMode>()
                    .Select(p => new { Key = (int)p, Value = p.ToString() })
                    .ToList();
                ModeBox.SelectedIndex = Enum.GetValues(typeof(SSSRMode))
                    .Cast<SSSRMode>()
                    .ToList()
                    .IndexOf(Settings.Mode);
                ModeBox.DisplayMember = "Value";
                ModeBox.ValueMember = "Key";
            }

            protected override void LoadSettings()
            {
                Initialize();

                PassesSetter.Value = Settings.Passes;
                LinearLightBox.Checked = Settings.LinearLight;
                OverSharpSetter.Value = (decimal)Settings.OverSharp;
                LocalitySetter.Value = (decimal)Settings.Locality;
                
                UpdateGui();
            }

            protected override void SaveSettings()
            {
                Settings.Passes = (int)PassesSetter.Value;
                Settings.LinearLight = LinearLightBox.Checked;
                Settings.OverSharp = (float)OverSharpSetter.Value;
                Settings.Locality = (float)LocalitySetter.Value;

                Settings.PrescalerGroup.SelectedIndex = PrescalerBox.SelectedIndex;
                Settings.Mode = (SSSRMode)ModeBox.SelectedValue;
            }

            private void SelectionChanged(object sender, EventArgs e)
            {
                UpdateGui();
            }

            private void UpdateGui()
            {
                var preset = PrescalerBox.SelectedValue as Preset<ITextureFilter, IRenderScript>;
                ConfigButton.Enabled = (preset != null) && preset.HasConfigDialog();
            }

            private void ConfigButtonClick(object sender, EventArgs e)
            {
                ((Preset<ITextureFilter, IRenderScript>) PrescalerBox.SelectedValue).ShowConfigDialog(Owner);
            }

            private void ModifyButtonClick(object sender, EventArgs e)
            {
                SaveSettings();

                var groupUi = new ScriptGroupScript { Settings = Settings.PrescalerGroup };
                groupUi.ShowConfigDialog(Owner);

                LoadSettings();
            }
        }

        public class SSSRConfigDialogBase : ScriptConfigDialog<SSSR>
        {
        }
    }
}
