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

namespace Mpdn.Extensions.RenderScripts
{
    namespace Mpdn.OclNNedi3
    {
        public partial class OclNNedi3ConfigDialog : OclNNedi3ConfigDialogBase
        {
            public OclNNedi3ConfigDialog()
            {
                InitializeComponent();
            }

            protected override void LoadSettings()
            {
                neuronsSelector.Initialize((int) Settings.Neurons1, (int) Settings.Neurons2);
                chromaSelector.Initialize(Settings.ChromaScalers, Settings.ChromaScalerGuid);
            }

            protected override void SaveSettings()
            {
                Settings.Neurons1 = (OclNNedi3Neurons) neuronsSelector.Neurons1;
                Settings.Neurons2 = (OclNNedi3Neurons) neuronsSelector.Neurons2;
                Settings.ChromaScalers = chromaSelector.ChromaScalers;
                Settings.ChromaScalerGuid = chromaSelector.SelectedChromScalerGuid;
            }
        }

        public class OclNNedi3ConfigDialogBase : ScriptConfigDialog<OclNNedi3>
        {
        }
    }
}
