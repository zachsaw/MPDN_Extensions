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
using System.Diagnostics;
using System.Windows.Forms;
using Mpdn.Extensions.Framework;

namespace Mpdn.Extensions.PlayerExtensions
{
    public partial class UpdateCheckerNewVersionForm : Form
    {
        private readonly UpdateCheckerSettings m_settings;

        public UpdateCheckerNewVersionForm(VersionHelpers.Version version, UpdateCheckerSettings settings)
        {
            InitializeComponent();
            m_settings = settings;
            Text += ": " + version;
            changelogBox.Text = version.Changelog;
        }

        private void forgetUpdate_Click(object sender, EventArgs e)
        {
            m_settings.ForgetMpdnVersion = true;
            Close();
        }

        private void downloadButton_Click(object sender, EventArgs e)
        {
            Process.Start(UpdateChecker.WebsiteUrl);
            Close();
        }

        private void checkBoxDisable_CheckedChanged(object sender, EventArgs e)
        {
            m_settings.CheckForUpdate = !checkBoxDisable.Checked;
        }
    }
}
