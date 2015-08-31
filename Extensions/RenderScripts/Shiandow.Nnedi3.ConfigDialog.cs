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
using Mpdn.Extensions.Framework.Controls;
using Mpdn.Extensions.Framework.RenderChain;
using Mpdn.Extensions.RenderScripts.Shiandow.NNedi3.Filters;
using CodePath = System.Tuple<string, int>;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Shiandow.NNedi3
    {
        public partial class NNedi3ConfigDialog : NNedi3ConfigDialogBase
        {
            private readonly CodePath[] m_CodePaths =
            {
                new CodePath("Prefer Scalar", (int) NNedi3Path.ScalarMad),
                new CodePath("Prefer Vector", (int) NNedi3Path.VectorDot),
                new CodePath("Avoid Branches", (int) NNedi3Path.UnrolledVectorDot),
                new CodePath("Prefer Scalar & Small Code", (int) NNedi3Path.ScalarMadSmall),
                new CodePath("Prefer Vector & Small Code", (int) NNedi3Path.VectorDotSmall)
            };

            public NNedi3ConfigDialog()
            {
                InitializeComponent();
                UpdateCodePaths(false);
            }

            protected override void LoadSettings()
            {
                PopulateChromaScalers();
                comboBoxNeurons1.SelectedIndex = (int) Settings.Neurons1;
                comboBoxNeurons2.SelectedIndex = (int) Settings.Neurons2;
                comboBoxPath.SelectedIndex = (int) Settings.CodePath;
                checkBoxStructured.Checked = Settings.Structured;
                SelectChromaScaler();
            }

            private void SelectChromaScaler()
            {
                comboBoxChroma.SelectedIndex = 0;
                int i = 0;
                foreach (ComboBoxItem<Preset> item in comboBoxChroma.Items)
                {
                    if (item.Value.Script.Descriptor.Guid == Settings.ChromaScalerGuid)
                    {
                        comboBoxChroma.SelectedIndex = i;
                        return;
                    }
                    i++;
                }
            }

            protected override void SaveSettings()
            {
                Settings.Neurons1 = (NNedi3Neurons) comboBoxNeurons1.SelectedIndex;
                Settings.Neurons2 = (NNedi3Neurons) comboBoxNeurons2.SelectedIndex;
                Settings.CodePath = (NNedi3Path) ((ComboBoxItem<CodePath>)comboBoxPath.SelectedItem).Value.Item2;
                Settings.Structured = checkBoxStructured.Checked;
                Settings.ChromaScalers = comboBoxChroma.Items.Cast<ComboBoxItem<Preset>>().Select(s => s.Value).ToList();
                Settings.ChromaScalerGuid = SelectedChromaScaler.Script.Descriptor.Guid;
            }

            private void StructuredCheckedChanged(object sender, EventArgs e)
            {
                UpdateCodePaths(checkBoxStructured.Checked);
            }

            private void UpdateCodePaths(bool structured)
            {
                var item = (ComboBoxItem<CodePath>) comboBoxPath.SelectedItem;
                var oldSelection = item == null ? -1 : item.Value.Item2;

                comboBoxPath.Items.Clear();
                foreach (var p in m_CodePaths)
                {
                    comboBoxPath.Items.Add(new ComboBoxItem<CodePath>(p.Item1, p));
                }

                comboBoxPath.SelectedIndex = oldSelection;

                if (!structured)
                    return;

                // Structured version doesn't have "Avoid Branches" code path
                const int avoidBranchesIndex = 2;
                if (comboBoxPath.SelectedIndex == avoidBranchesIndex)
                {
                    comboBoxPath.SelectedIndex = 0;
                }
                comboBoxPath.Items.RemoveAt(avoidBranchesIndex);
            }

            private void PopulateChromaScalers()
            {
                var chromaScalers = Extension.RenderScripts
                    .OfType<IRenderChainUi>()
                    .Select(
                        x =>
                            Settings.ChromaScalers.FirstOrDefault(s => s.Script.Descriptor.Guid == x.Descriptor.Guid) ??
                            x.MakeNewPreset())
                    .Where(x => x.Chain.GetType().GetInterfaces().Contains(typeof (IChromaScaler)))
                    .OrderBy(x => x.Name);

                var dataSource =
                    (new[] {RenderChainUi.Identity.ToPreset()}).Concat(chromaScalers)
                        .Select(x => new ComboBoxItem<Preset>(x.Name, x))
                        .ToList();

                comboBoxChroma.DataSource = dataSource;
                buttonConfig.Enabled = false;
            }

            private void ChromaSelectedIndexChanged(object sender, EventArgs e)
            {
                buttonConfig.Enabled = SelectedChromaScaler.HasConfigDialog();
            }

            private Preset SelectedChromaScaler
            {
                get { return ((ComboBoxItem<Preset>) comboBoxChroma.SelectedItem).Value; }
            }

            private void ButtonConfigClick(object sender, EventArgs e)
            {
                SelectedChromaScaler.ShowConfigDialog(this);
            }
        }

        public class NNedi3ConfigDialogBase : ScriptConfigDialog<NNedi3>
        {
        }
    }
}
