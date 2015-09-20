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
using Mpdn.Extensions.Framework.Config;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Hylian.SuperXbr
    {
        public partial class SuperXbrConfigDialog : SuperXbrConfigDialogBase
        {
            public SuperXbrConfigDialog()
            {
                InitializeComponent();
            }

            protected override void LoadSettings()
            {
                EdgeStrengthSetter.Value = (decimal)Settings.EdgeStrength;
                SharpnessSetter.Value = (decimal)Settings.Sharpness;
                FastBox.Checked = Settings.FastMethod;
                ExtraPassBox.Checked = Settings.ThirdPass;
            }

            protected override void SaveSettings()
            {
                Settings.EdgeStrength = (float)EdgeStrengthSetter.Value;
                Settings.Sharpness = (float)SharpnessSetter.Value;
                Settings.FastMethod = FastBox.Checked;
                Settings.ThirdPass = ExtraPassBox.Checked;
            }
        }

        public class SuperXbrConfigDialogBase : ScriptConfigDialog<SuperXbr>
        {
        }
    }
}
