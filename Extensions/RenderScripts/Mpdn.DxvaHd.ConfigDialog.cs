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

using Mpdn.DxvaHd;
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.Framework.Config;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Mpdn.DxvaHd
    {
        public partial class DxvaHdScalerConfigDialog : DxvaHdScalerConfigDialogBase
        {
            public DxvaHdScalerConfigDialog()
            {
                InitializeComponent();

                var descs = EnumHelpers.GetDescriptions<DxvaHdQuality>();
                foreach (var desc in descs)
                {
                    comboBoxQuality.Items.Add(desc);
                }
            }

            protected override void LoadSettings()
            {
                comboBoxQuality.SelectedIndex = (int) Settings.Quality;
                checkBoxYuvMode.Checked = Settings.YuvMode;
            }

            protected override void SaveSettings()
            {
                Settings.Quality = (DxvaHdQuality) comboBoxQuality.SelectedIndex;
                Settings.YuvMode = checkBoxYuvMode.Checked;
            }
        }

        public class DxvaHdScalerConfigDialogBase : ScriptConfigDialog<DxvaHdScaler>
        {
        }
    }
}
