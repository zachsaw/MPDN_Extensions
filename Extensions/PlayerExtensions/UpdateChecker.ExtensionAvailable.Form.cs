using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Windows.Forms;

namespace Mpdn.Extensions.PlayerExtensions.UpdateChecker
{
    public partial class UpdateCheckerNewExtensionForm : Form
    {
        private readonly List<GitHubVersion.GitHubAsset> m_Files;
        private readonly UpdateCheckerSettings m_Settings;
        private string m_ChosenDownload;

        public UpdateCheckerNewExtensionForm(ExtensionVersion version, UpdateCheckerSettings settings)
        {
            InitializeComponent();
            Icon = PlayerControl.ApplicationIcon;
            downloadButton.ContextMenuStrip = new ContextMenuStrip();
            m_Files = version.Files;
            foreach (var file in m_Files)
            {
                downloadButton.ContextMenuStrip.Items.Add(file.name);
            }

            downloadButton.ContextMenuStrip.ItemClicked +=
                delegate(object sender, ToolStripItemClickedEventArgs args)
                {
                    m_ChosenDownload = args.ClickedItem.Text;
                    DownloadButtonClick(sender, args);
                };
            CancelButton = CloseButton;

            m_Settings = settings;
            Text += ": " + version;
            changelogBox.Text = version.Changelog;
        }

        private void DownloadButtonClick(object sender, EventArgs e)
        {
            if (m_ChosenDownload == null)
            {
                var chosenDownload = m_Files.First(file => file.name.Contains("Installer")).name;
                if (chosenDownload != null)
                    m_ChosenDownload = m_Files[0].name;
            }
            string url =
                (m_Files.Where(file => file.name == m_ChosenDownload).Select(file => file.browser_download_url))
                    .FirstOrDefault();
            if (url != null) Process.Start(url);
            Close();
        }

        private void ForgetUpdateClick(object sender, EventArgs e)
        {
            m_Settings.ForgetExtensionVersion = true;
            Close();
        }

        private void CheckBoxDisableCheckedChanged(object sender, EventArgs e)
        {
            m_Settings.CheckForUpdate = !checkBoxDisable.Checked;
        }
    }
}