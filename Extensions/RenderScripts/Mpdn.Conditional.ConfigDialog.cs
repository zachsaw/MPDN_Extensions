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
using System.Diagnostics;
using System.Windows.Forms;
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.Framework.Config;
using Mpdn.Extensions.Framework.Controls;
using Mpdn.Extensions.RenderScripts.Mpdn.Presets;
using Mpdn.Extensions.RenderScripts.Mpdn.ScriptedRenderChain;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Mpdn.Conditional
    {
        public partial class ConditionalConfigDialog : ConditionalConfigDialogBase
        {
            private IRenderChainUi m_ScriptGroupScript;

            public ConditionalConfigDialog()
            {
                InitializeComponent();
            }

            protected override void LoadSettings()
            {
                conditionBox.Text = Settings.Condition;

                m_ScriptGroupScript = new ScriptGroupScript().CreateNew(true);
                var presetGroup = (PresetGroup) m_ScriptGroupScript.Chain;
                foreach (var option in presetGroup.Options)
                {
                    comboBoxPreset.Items.Add(option);
                }
                comboBoxPreset.SelectedIndex = presetGroup.GetPresetIndex(Settings.Preset);

                UpdateControls();
            }

            protected override void SaveSettings()
            {
                var presetGroup = (PresetGroup) m_ScriptGroupScript.Chain;
                Settings.Condition = conditionBox.Text;
                Settings.Preset = comboBoxPreset.SelectedIndex < 0
                    ? Guid.Empty
                    : presetGroup.Options[comboBoxPreset.SelectedIndex].Guid;
            }

            private void DialogClosing(object sender, FormClosingEventArgs e)
            {
                if (DialogResult != DialogResult.OK)
                    return;

                string error;
                using (new HourGlass())
                {
                    if (ValidateSyntax(conditionBox.Text, out error))
                        return;
                }

                e.Cancel = true;
                MessageBox.Show(this, error, "Syntax Error", MessageBoxButtons.OK, MessageBoxIcon.Stop);
            }

            private static bool ValidateSyntax(string condition, out string error)
            {
                error = string.Empty;
                using (var engine = new MpdnScriptEngine())
                {
                    try
                    {
                        engine.Execute(null, null, CreateJsCode(Parser.BuildCondition(condition)), "Conditional");
                    }
                    catch (Exception ex)
                    {
                        error = ex.Message;
                        return false;
                    }
                    return true;
                }
            }

            private static string CreateJsCode(string condition)
            {
                return string.Format("if ({0}) Debug.Assert(true);", condition);
            }

            private void ConfigButtonClick(object sender, EventArgs e)
            {
                var preset = (Preset) comboBoxPreset.SelectedItem;
                if (preset != null && preset.Script.HasConfigDialog())
                {
                    if (preset.Script.ShowConfigDialog(this))
                    {
                        ((IPersistentConfig) m_ScriptGroupScript).Save();
                    }
                }
            }

            private void PresetSelectedIndexChanged(object sender, EventArgs e)
            {
                UpdateControls();
            }

            private void UpdateControls()
            {
                var preset = (Preset) comboBoxPreset.SelectedItem;
                configButton.Enabled = preset != null && preset.Script.HasConfigDialog();
            }

            private void LinkClicked(object sender, LinkLabelLinkClickedEventArgs e)
            {
                Process.Start("https://github.com/zachsaw/MPDN_Extensions/wiki/The-Conditional-Render-Script");
            }
        }

        public class ConditionalConfigDialogBase : ScriptConfigDialog<Conditional>
        {
        }
    }
}
