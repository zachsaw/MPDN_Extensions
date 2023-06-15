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
using System.Windows.Forms;
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.Framework.Config;
using Mpdn.Extensions.RenderScripts.Mpdn.NeoCnn.Ai.Scaler;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Mpdn.Resizer
    {
        public partial class NeoCnnConfigDialog : NeoCnnConfigDialogBase
        {
            public NeoCnnConfigDialog()
            {
                InitializeComponent();

                var descs = EnumHelpers.GetDescriptions<ResizerOption>();
                foreach (var desc in descs)
                {
                    listBox.Items.Add(desc);
                }

                listBox.SelectedIndex = 0;
            }

            protected override void LoadSettings()
            {
                // listBox.SelectedIndex = (int) Settings.ResizerOption;
            }

            protected override void SaveSettings()
            {
                // Settings.ResizerOption = (ResizerOption) listBox.SelectedIndex;
            }

            private void ListBoxDoubleClick(object sender, EventArgs e)
            {
                DialogResult = DialogResult.OK;
            }
        }

        public class NeoCnnConfigDialogBase : ScriptConfigDialog<NeoCnn.Ai.Scaler.NeoCnn>
        {
        }
    }
}
