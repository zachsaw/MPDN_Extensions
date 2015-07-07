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
using System.Collections.Generic;
using System.Diagnostics;
using System.Windows.Forms;
using Mpdn.Extensions.Framework;

namespace Mpdn.Extensions.PlayerExtensions
{
    public partial class UpdateCheckerNewExtensionForm : Form
    {
        private readonly UpdateCheckerSettings _settings;
        private string _chosenDownload = null;
        private readonly List<VersionHelpers.GitHubVersion.GitHubAsset> _files;

        public UpdateCheckerNewExtensionForm(VersionHelpers.ExtensionVersion version, UpdateCheckerSettings settings)
        {
            InitializeComponent();
            Icon = PlayerControl.ApplicationIcon;
            downloadButton.ContextMenuStrip = new ContextMenuStrip();
            _files = version.Files;
            foreach (var file in _files)
            {
                downloadButton.ContextMenuStrip.Items.Add(file.name);
            }
            
            downloadButton.ContextMenuStrip.ItemClicked +=
                delegate(object sender, ToolStripItemClickedEventArgs args)
                {
                    _chosenDownload = args.ClickedItem.Text;
                    downloadButton_Click(sender, args);
                };
            CancelButton = CloseButton;

            _settings = settings;
            Text += ": " + version;
            changelogBox.Text = version.Changelog;
        }

        private void downloadButton_Click(object sender, EventArgs e)
        {
            if (_chosenDownload == null)
            {
                _chosenDownload = _files[0].name;
            }
            string url = null;
            foreach (var file in _files)
            {
                if (file.name == _chosenDownload)
                {
                    url = file.browser_download_url;
                    break;
                }
            }
            if (url != null) Process.Start(url);
            Close();
        }

        private void forgetUpdate_Click(object sender, EventArgs e)
        {
            _settings.ForgetExtensionVersion = true;
            Close();
        }

        private void checkBoxDisable_CheckedChanged(object sender, EventArgs e)
        {
            _settings.CheckForUpdate = !checkBoxDisable.Checked;
        }
    }
}