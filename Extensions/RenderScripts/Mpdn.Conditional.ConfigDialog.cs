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
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.Framework.Chain;
using Mpdn.Extensions.Framework.Config;
using Mpdn.Extensions.Framework.Controls;
using Mpdn.Extensions.Framework.Exceptions;
using Mpdn.Extensions.Framework.RenderChain;
using Mpdn.Extensions.Framework.Scripting;
using Mpdn.RenderScript;

namespace Mpdn.Extensions.RenderScripts
{
    using Preset = Preset<IFilter, IRenderScript>;
    using BoxItem = ComboBoxItem<Preset<IFilter, IRenderScript>>;

    namespace Mpdn.Conditional
    {
        public partial class ConditionalConfigDialog : ConditionalConfigDialogBase
        {
            public ConditionalConfigDialog()
            {
                InitializeComponent();

                MinimumSize = new Size(Width, Height);
                MaximumSize = new Size(int.MaxValue, Height);
            }

            protected override void LoadSettings()
            {
                conditionBox.Text = Settings.Condition;

                var renderScripts = Extension.RenderScripts
                    .Where(script => script is IRenderChainUi)
                    .Select(x => (x as IRenderChainUi).CreateNew())
                    .Concat(new[] {RenderChainUi.Identity}).ToList();
                foreach (var s in renderScripts)
                {
                    if (s.Category.ToLowerInvariant() == "hidden")
                        continue;

                    comboBoxPreset.Items.Add(CreateBoxItem(s.ToPreset()));
                }
                if (Settings.Preset != null && Settings.Preset.Script != null)
                {
                    var guid = Settings.Preset.Script.Descriptor.Guid;
                    comboBoxPreset.SelectedIndex = comboBoxPreset
                        .Items
                        .Cast<ComboBoxItem<Preset<IFilter, IRenderScript>>>()
                        .Select(item => item.Value.Script)
                        .ToList()
                        .FindIndex(ui => ui.Descriptor.Guid == guid);

                    var index = comboBoxPreset.SelectedIndex;
                    if (index >= 0)
                    {
                        comboBoxPreset.Items[index] = CreateBoxItem(Settings.Preset);
                    }
                }

                UpdateControls();
            }

            private static BoxItem CreateBoxItem(Preset preset)
            {
                return new BoxItem(preset.Name, preset);
            }

            protected override void SaveSettings()
            {
                Settings.Condition = conditionBox.Text;
                var item = (BoxItem) comboBoxPreset.SelectedItem;
                Settings.Preset = item == null ? null : item.Value;
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
                using (var engine = new ScriptEngine())
                {
                    try
                    {
                        engine.Execute(null, CreateJsCode(Parser.BuildCondition(condition)), "Conditional");
                    }
                    catch (MpdnScriptEngineException ex)
                    {
                        error = ex.Message;
                        return false;
                    }
                    catch (Exception)
                    {
                        // ignore general exceptions for mock clip
                        return true;
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
                var item = (BoxItem) comboBoxPreset.SelectedItem;
                if (item == null || !item.Value.HasConfigDialog()) 
                    return;

                if (item.Value.ShowConfigDialog(this))
                {
                    UpdateControls();
                }
            }

            private void PresetSelectedIndexChanged(object sender, EventArgs e)
            {
                UpdateControls();
            }

            private void UpdateControls()
            {
                var item = (BoxItem) comboBoxPreset.SelectedItem;
                if (item == null)
                {
                    configButton.Enabled = false;
                    labelDesc.Text = string.Empty;
                    return;
                }
                configButton.Enabled = item.Value.HasConfigDialog();
                labelDesc.Text = item.Value.Description;
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
