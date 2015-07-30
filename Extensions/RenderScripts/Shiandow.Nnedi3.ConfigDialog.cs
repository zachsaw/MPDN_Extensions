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
using Mpdn.Extensions.Framework.Config;
using Mpdn.Extensions.RenderScripts.Shiandow.NNedi3.Filters;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Shiandow.NNedi3
    {
        public partial class NNedi3ConfigDialog : NNedi3ConfigDialogBase
        {
            public NNedi3ConfigDialog()
            {
                InitializeComponent();
            }

            protected override void LoadSettings()
            {
                comboBoxNeurons1.SelectedIndex = (int) Settings.Neurons1;
                comboBoxNeurons2.SelectedIndex = (int) Settings.Neurons2;
                comboBoxPath.SelectedIndex = (int) Settings.CodePath;
            }

            protected override void SaveSettings()
            {
                Settings.Neurons1 = (NNedi3Neurons) comboBoxNeurons1.SelectedIndex;
                Settings.Neurons2 = (NNedi3Neurons) comboBoxNeurons2.SelectedIndex;
                Settings.CodePath = (NNedi3Path) comboBoxPath.SelectedIndex;
            }
        }

        public class NNedi3ConfigDialogBase : ScriptConfigDialog<NNedi3>
        {
        }
    }
}
