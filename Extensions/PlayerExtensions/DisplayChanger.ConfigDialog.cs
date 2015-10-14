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

using System.Linq;
using System.Windows.Forms;
using Mpdn.Extensions.Framework.Config;

namespace Mpdn.Extensions.PlayerExtensions
{
    public partial class DisplayChangerConfigDialog : DisplayChangerConfigBase
    {
        public DisplayChangerConfigDialog()
        {
            InitializeComponent();

            labelFormat.Text = string.Format(labelFormat.Text, VideoSpecifier.FormatHelp);
            labelExample.Text = string.Format(labelExample.Text, VideoSpecifier.ExampleHelp);
        }

        protected override void LoadSettings()
        {
            checkBoxActivate.Checked = Settings.Activate;
            checkBoxRestore.Checked = Settings.Restore;
            checkBoxRestoreExit.Checked = Settings.RestoreOnExit;
            checkBoxHighestRate.Checked = Settings.HighestRate;
            checkBoxRestricted.Checked = Settings.Restricted;
            textBoxVideoTypes.Text = Settings.VideoTypes;
        }

        protected override void SaveSettings()
        {
            Settings.Activate = checkBoxActivate.Checked;
            Settings.Restore = checkBoxRestore.Checked;
            Settings.RestoreOnExit = checkBoxRestoreExit.Checked;
            Settings.HighestRate = checkBoxHighestRate.Checked;
            Settings.Restricted = checkBoxRestricted.Checked;
            Settings.VideoTypes = textBoxVideoTypes.Text;
        }

        private void TextBoxVideoTypesValidating(object sender, System.ComponentModel.CancelEventArgs e)
        {
            var s = textBoxVideoTypes.Text.Split(' ');
            var valid = s.All(VideoSpecifier.IsValid);
            errorProvider.SetError(textBoxVideoTypes, !valid ? "Error: Invalid video type specifier" : string.Empty);
        }

        private void DialogClosing(object sender, FormClosingEventArgs e)
        {
            if (DialogResult != DialogResult.OK)
                return;

            var error = errorProvider.GetError(textBoxVideoTypes);
            if (error == string.Empty) 
                return;

            e.Cancel = true;
            ActiveControl = textBoxVideoTypes;
            // Force error provider to blink error icon
            errorProvider.SetError(textBoxVideoTypes, string.Empty);
            errorProvider.SetError(textBoxVideoTypes, error);
        }
    }

    public class DisplayChangerConfigBase : ScriptConfigDialog<DisplayChangerSettings>
    {
    }
}
