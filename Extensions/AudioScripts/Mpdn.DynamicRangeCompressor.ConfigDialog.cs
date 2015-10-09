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

namespace Mpdn.Extensions.AudioScripts
{
    namespace Mpdn
    {
        public partial class DynamicRangeCompressorConfigDialog : DynamicRangeCompressorConfigDialogBase
        {
            public DynamicRangeCompressorConfigDialog()
            {
                InitializeComponent();
            }

            protected override void LoadSettings()
            {
                textBoxThreshold.Value = (decimal) Settings.ThresholddB;
                textBoxGain.Value = (decimal) Settings.MakeupGaindB;
                textBoxRatio.Value = (decimal) Settings.Ratio;
                textBoxAttack.Value = (decimal) (Settings.AttackMs/1000.0);
                textBoxRelease.Value = (decimal) (Settings.ReleaseMs/1000.0);
            }

            protected override void SaveSettings()
            {
                Settings.ThresholddB = (float) textBoxThreshold.Value;
                Settings.MakeupGaindB = (float) textBoxGain.Value;
                Settings.Ratio = (float) textBoxRatio.Value;
                Settings.AttackMs = (int) (textBoxAttack.Value*1000);
                Settings.ReleaseMs = (int) (textBoxRelease.Value*1000);
            }
        }

        public class DynamicRangeCompressorConfigDialogBase : ScriptConfigDialog<DynamicRangeCompressor>
        {
        }
    }
}