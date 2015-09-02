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

using System.Windows.Forms;

namespace Mpdn.Extensions.RenderScripts
{
    public partial class NeuronsSelector : UserControl
    {
        public NeuronsSelector()
        {
            InitializeComponent();
            comboBoxNeurons1.SelectedIndex = 0;
            comboBoxNeurons2.SelectedIndex = 0;
        }

        public void Initialize(int neurons1, int neurons2)
        {
            comboBoxNeurons1.SelectedIndex = neurons1;
            comboBoxNeurons2.SelectedIndex = neurons2;
        }

        public int Neurons1 { get { return comboBoxNeurons1.SelectedIndex; } }
        public int Neurons2 { get { return comboBoxNeurons2.SelectedIndex; } }
    }
}
