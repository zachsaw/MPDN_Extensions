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
using System.Windows.Forms;
using Mpdn.Extensions.Framework.RenderChain;

namespace Mpdn.Extensions.Framework.Controls
{
    public partial class ChromaScalerSelector : UserControl
    {
        public ChromaScalerSelector()
        {
            InitializeComponent();
        }

        public List<ChromaScalerPreset> ChromaScalers
        {
            get { return comboBoxChroma.Items.Cast<ComboBoxItem<ChromaScalerPreset>>().Select(s => s.Value).ToList(); }
        }

        public Guid SelectedChromScalerGuid
        {
            get { return SelectedChromaScaler.Script.Descriptor.Guid; }
        }

        public void Initialize(IList<ChromaScalerPreset> chromaScalers, Guid selected)
        {
            PopulateChromaScalers(chromaScalers, comboBoxChroma);
            SelectChromaScaler(selected);
        }

        private Preset SelectedChromaScaler
        {
            get { return ((ComboBoxItem<ChromaScalerPreset>) comboBoxChroma.SelectedItem).Value; }
        }

        private void SelectChromaScaler(Guid selected)
        {
            comboBoxChroma.SelectedIndex = 0;
            int i = 0;
            foreach (ComboBoxItem<ChromaScalerPreset> item in comboBoxChroma.Items)
            {
                if (item.Value.Script.Descriptor.Guid == selected)
                {
                    comboBoxChroma.SelectedIndex = i;
                    return;
                }
                i++;
            }
        }

        private static void PopulateChromaScalers(IEnumerable<ChromaScalerPreset> scalers, ComboBox boxChroma)
        {
            var chromaScalers = Extension.RenderScripts
                .OfType<IRenderChainUi>()
                .Select(
                    x =>
                        scalers.FirstOrDefault(s => s.Script.Descriptor.Guid == x.Descriptor.Guid) ??
                        x.MakeNewChromaScalerPreset())
                .Where(x => x.Chain.GetType().GetInterfaces().Contains(typeof (IChromaScaler)))
                .OrderBy(x => x.Name);

            var dataSource =
                (new[] {RenderChainUi.Identity.ToChromaScalerPreset()}).Concat(chromaScalers)
                    .Select(x => new ComboBoxItem<ChromaScalerPreset>(x.Name, x))
                    .ToList();

            boxChroma.DataSource = dataSource;
        }

        private void ChromaSelectedIndexChanged(object sender, EventArgs e)
        {
            buttonConfig.Enabled = SelectedChromaScaler.HasConfigDialog();
        }

        private void ButtonConfigClick(object sender, EventArgs e)
        {
            SelectedChromaScaler.ShowConfigDialog(this);
        }
    }
}
