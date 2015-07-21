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
    namespace Mpdn.Jinc2D
    {
        public partial class Jinc2DConfigDialog : Jinc2DConfigDialogBase
        {
            public Jinc2DConfigDialog()
            {
                InitializeComponent();
            }

            protected override void LoadSettings()
            {
                checkBoxAntiRinging.Checked = Settings.AntiRingingEnabled;
                AntiRingingStrengthSetter.Value = (decimal) Settings.AntiRingingStrength;
            }

            protected override void SaveSettings()
            {
                Settings.AntiRingingEnabled = checkBoxAntiRinging.Checked;
                Settings.AntiRingingStrength = (float)AntiRingingStrengthSetter.Value;
            }
        }

        public class Jinc2DConfigDialogBase : ScriptConfigDialog<Jinc2D>
        {
        }
    }
}
