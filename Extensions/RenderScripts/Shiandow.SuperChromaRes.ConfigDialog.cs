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
using Mpdn.Extensions.Framework.Config;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Shiandow.SuperRes
    {
        public partial class SuperChromaResConfigDialog : SuperChromaResConfigDialogBase
        {
            public SuperChromaResConfigDialog()
            {
                InitializeComponent();
            }

            protected override void LoadSettings()
            {
                PassesSetter.Value = (Decimal)Settings.Passes;
                StrengthSetter.Value = (Decimal)Settings.Strength;
                SoftnessSetter.Value = (Decimal)Settings.Softness;
                PrescalerBox.Checked = Settings.Prescaler;
            }

            protected override void SaveSettings()
            {
                Settings.Passes = (int)PassesSetter.Value;
                Settings.Strength = (float)StrengthSetter.Value;
                Settings.Softness = (float)SoftnessSetter.Value;
                Settings.Prescaler = PrescalerBox.Checked; 
            }
        }

        public class SuperChromaResConfigDialogBase : ScriptConfigDialog<SuperChromaRes>
        {
        }
    }
}
