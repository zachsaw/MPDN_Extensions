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
using System.ComponentModel;
using System.Linq;
using System.Net;
using System.Reflection;
using System.Windows.Forms;
using Mpdn.Extensions.Framework;

namespace Mpdn.Extensions.PlayerExtensions.UpdateChecker
{
    public partial class SimpleUpdateForm : Form
    {
        public enum UpdateType
        {
            Player,
            Extensions,
            Both
        }

        private readonly UpdateCheckerSettings m_Settings;
        private readonly UpdateType m_Type;
        private WebFile m_DownloadingWebFile;

        public SimpleUpdateForm(UpdateType type, UpdateCheckerSettings settings)
        {
            InitializeComponent();
            Icon = Gui.Icon;
            m_Type = type;
            m_Settings = settings;
            Closing += OnClosing;
        }

        private void OnClosing(object sender, CancelEventArgs cancelEventArgs)
        {
            if (m_DownloadingWebFile != null)
            {
                m_DownloadingWebFile.CancelDownload();
            }
        }

        private void ForgetUpdateButtonClick(object sender, EventArgs e)
        {
            switch (m_Type)
            {
                case UpdateType.Both:
                    m_Settings.ForgetExtensionVersion = true;
                    m_Settings.ForgetMpdnVersion = true;
                    break;
                case UpdateType.Extensions:
                    m_Settings.ForgetExtensionVersion = true;
                    break;
                case UpdateType.Player:
                    m_Settings.ForgetMpdnVersion = true;
                    break;
            }
        }

        private void InstallButtonClick(object sender, EventArgs e)
        {
            WebFile installer = null;
            downloadProgressBar.Visible = true;
            installButton.Enabled = false;
            switch (m_Type)
            {
                case UpdateType.Both:
                    UpdateBoth();
                    break;
                case UpdateType.Extensions:
                    installer = UpdateExtensions();
                    break;
                case UpdateType.Player:
                    installer = UpdatePlayer();
                    break;
            }

            if (installer == null) return;

            m_DownloadingWebFile = installer;
            m_DownloadingWebFile.Downloaded += InstallerOnDownloaded;
            m_DownloadingWebFile.DownloadFailed += InstallerOnDownloadFailed;
            m_DownloadingWebFile.DownloadProgressChanged += InstallerOnDownloadProgressChanged;
            m_DownloadingWebFile.DownloadFile();
        }

        private void InstallerOnDownloadProgressChanged(object sender, DownloadProgressChangedEventArgs downloadProgressChangedEventArgs)
        {
            GuiThread.Do(() =>
            {
                downloadProgressBar.Value = downloadProgressChangedEventArgs.ProgressPercentage;
            });
        }

        private void InstallerOnDownloadFailed(object sender, Exception error)
        {
            var file = ((WebFile) sender);
            GuiThread.Do(() =>
            {
                downloadProgressBar.Visible = false;
                installButton.Enabled = true;
                MessageBox.Show(Gui.VideoBox, string.Format("Problem while downloading: {0}\n{1}", file.FileUri, error.Message),
               "Download Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            });
           

        }

        private void InstallerOnDownloaded(object sender)
        {
            ((WebFile) sender).Start();
            Application.Exit();
        }

        private WebFile UpdatePlayer()
        {
            var arch = ArchitectureHelper.GetPlayerArtchitecture();
            var installer =
                m_Settings.MpdnVersionOnServer.GenerateSplitButtonItemList().First(file => file.Name.Contains(arch) && file.IsFile && file.Name.Contains(".exe"));
            downloadProgressBar.CustomText = installer.Name;
            return new TemporaryWebFile(new Uri(installer.Url));
        }

        private WebFile UpdateExtensions()
        {
            var installer =
               m_Settings.ExtensionVersionOnServer.Files.First(file => file.name.Contains(".exe"));
            downloadProgressBar.CustomText = installer.name;
            return new TemporaryWebFile(new Uri(installer.browser_download_url));
        }

        private void UpdateBoth()
        {
            m_DownloadingWebFile = UpdatePlayer();
            m_DownloadingWebFile.DownloadProgressChanged += InstallerOnDownloadProgressChanged;
            m_DownloadingWebFile.DownloadFailed += InstallerOnDownloadFailed;
            m_DownloadingWebFile.Downloaded += ((sender) =>
            {
                var filePath = ((WebFile) sender).FilePath;
                m_DownloadingWebFile = UpdateExtensions();

                m_DownloadingWebFile.DownloadFailed += InstallerOnDownloadFailed;
                m_DownloadingWebFile.DownloadProgressChanged += InstallerOnDownloadProgressChanged;
                m_DownloadingWebFile.Downloaded += ((o) =>
                {
                    var downloadedExtensionInstaller = (WebFile) o;
                    downloadedExtensionInstaller.Start(string.Format("/ARCH={0} /INSTALLER=\"{1}\"", ArchitectureHelper.GetPlayerArtchitecture(), filePath));
                    Application.Exit();

                });
            });
            m_DownloadingWebFile.DownloadFile();

        }
    }
}