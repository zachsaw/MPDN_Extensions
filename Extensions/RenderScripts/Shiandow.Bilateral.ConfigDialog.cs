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
using System.Linq;
using Mpdn.Extensions.Framework.Config;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Shiandow.Bilateral
    {
        public partial class BilateralConfigDialog : BilateralConfigDialogBase
        {
            public BilateralConfigDialog()
            {
                InitializeComponent();
            }

            protected override void LoadSettings()
            {
                Initialize();

                StrengthSetter.Value = (decimal)Settings.Strength;
            }

            private void Initialize()
            {
                ModeBox.DataSource = Enum.GetValues(typeof(BilateralMode))
                    .Cast<BilateralMode>()
                    .Select(p => new { Key = (int)p, Value = p.ToString() })
                    .ToList();
                ModeBox.SelectedIndex = Enum.GetValues(typeof(BilateralMode))
                    .Cast<BilateralMode>()
                    .ToList()
                    .IndexOf(Settings.Mode);
                ModeBox.DisplayMember = "Value";
                ModeBox.ValueMember = "Key";
            }

            protected override void SaveSettings()
            {
                Settings.Strength = (float)StrengthSetter.Value;
            }
        }

        public class BilateralConfigDialogBase : ScriptConfigDialog<Bilateral>
        { }
    }
}
