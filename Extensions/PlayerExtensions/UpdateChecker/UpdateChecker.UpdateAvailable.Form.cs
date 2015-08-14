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
using System.Linq;
using System.Windows.Forms;
using Mpdn.Extensions.Framework;

namespace Mpdn.Extensions.PlayerExtensions.UpdateChecker
{
    public partial class UpdateAvailableForm : Form
    {
        protected readonly UpdateCheckerSettings Settings;
        protected readonly List<SplitButtonToolStripItem> SplitMenuChoices;
        private SplitButtonToolStripItem m_ChosenDownload;
        private WebFile m_File;

        public UpdateAvailableForm(Version version, UpdateCheckerSettings settings)
        {
            InitializeComponent();
            downloadProgressBar.DisplayStyle = TextProgressBar.ProgressBarDisplayText.Both;
            Icon = Gui.Icon;
            downloadButton.ContextMenuStrip = new ContextMenuStrip();
            SplitMenuChoices = version.GenerateSplitButtonItemList();

            foreach (var choice in SplitMenuChoices)
            {
                downloadButton.ContextMenuStrip.Items.Add(choice);
            }


            downloadButton.ContextMenuStrip.ItemClicked +=
                delegate(object sender, ToolStripItemClickedEventArgs args)
                {
                    m_ChosenDownload = (SplitButtonToolStripItem) args.ClickedItem;
                    DownloadButtonClick(sender, args);
                };
            CancelButton = CloseButton;

            Settings = settings;
            Text = "New Player available: " + version;
            changelogViewerWebBrowser.BeforeLoadPreviousChangelog +=
                ChangelogViewerWebBrowserOnBeforeLoadPreviousChangelog;
            SetChangelog(version);
        }

        public override sealed string Text
        {
            get { return base.Text; }
            set { base.Text = value; }
        }

        private void ChangelogViewerWebBrowserOnBeforeLoadPreviousChangelog(object sender,
            ChangelogWebViewer.LoadingChangelogEvent args)
        {
            args.ChangelogLines = LoadPreviousChangelog();
        }

        private void SetChangelog(Version version)
        {
            changelogViewerWebBrowser.SetChangelog(ParseChangeLog(version.ChangelogLines));
        }

        public virtual List<string> LoadPreviousChangelog()
        {
            return ChangelogHelper.LoadPreviousMpdnChangelog();
        }

        protected virtual List<string> ParseChangeLog(List<string> changelog)
        {
            return ChangelogHelper.ParseMpdnChangelog(changelog);
        }

        private void DownloadButtonClick(object sender, EventArgs e)
        {
            if (m_ChosenDownload == null)
            {
                m_ChosenDownload = DefaultDownloadFile();
            }
            var url = m_ChosenDownload.Url;

            if (url != null)
            {
                SetLastMenuChoiceUsed(m_ChosenDownload.Name);
                if (m_ChosenDownload.IsFile)
                {
                    downloadProgressBar.CustomText = m_ChosenDownload.Name;
                    DownloadFile(url);
                }
                else
                {
                    Process.Start(url);
                    Close();
                }
            }
            else
            {
                Close();
            }
        }

        protected virtual void SetLastMenuChoiceUsed(string name)
        {
            Settings.LastMpdnReleaseChosen = name;
        }

        protected virtual SplitButtonToolStripItem DefaultDownloadFile()
        {
            if (Settings.LastMpdnReleaseChosen != null)
            {
                return SplitMenuChoices.First(file => file.Name == Settings.LastMpdnReleaseChosen);
            }

            return
                SplitMenuChoices.First(
                    file => file.Name.Contains(ArchitectureHelper.GetPlayerArtchitecture().ToString())) ??
                SplitMenuChoices[0];
        }

        private void DownloadFile(string url)
        {
            downloadButton.Enabled = false;
            downloadProgressBar.Visible = true;

            m_File = new TemporaryWebFile(new Uri(url));
            m_File.Downloaded += o =>
            {
                m_File.Start();
                GuiThread.DoAsync(() =>
                {
                    ResetDlButtonProgressBar();
                    Close();
                });
            };
            m_File.DownloadProgressChanged +=
                (o, args) => GuiThread.DoAsync(() => { downloadProgressBar.Value = args.ProgressPercentage; });

            m_File.DownloadFailed += (sender, error) => GuiThread.DoAsync(() =>
            {
                ResetDlButtonProgressBar();
                MessageBox.Show(error.Message, "Download Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            });

            m_File.Cancelled += sender =>
            {
                Trace.WriteLine("Download Cancelled");
                ResetDlButtonProgressBar();
            };

            m_File.DownloadFile();
        }

        private void ResetDlButtonProgressBar()
        {
            downloadButton.Enabled = true;
            downloadProgressBar.Visible = false;
        }

        private void ForgetUpdateClick(object sender, EventArgs e)
        {
            ForgetVersion();
            Close();
        }

        protected virtual void ForgetVersion()
        {
            Settings.ForgetMpdnVersion = true;
        }

        private void CheckBoxDisableCheckedChanged(object sender, EventArgs e)
        {
            Settings.CheckForUpdate = !checkBoxDisable.Checked;
        }

        private void CloseButtonClick(object sender, EventArgs e)
        {
            if (m_File == null)
            {
                return;
            }

            m_File.CancelDownload();
        }

        #region SplitButtonToolStripItem

        public sealed class SplitButtonToolStripItem : ToolStripMenuItem
        {
            public SplitButtonToolStripItem(string name, string url, bool isFile)
            {
                Url = url;
                IsFile = isFile;
                Name = name;
                Text = name;
            }

            public SplitButtonToolStripItem(string name, string url) : this(name, url, true)
            {
            }

            public SplitButtonToolStripItem(GitHubVersion.GitHubAsset asset)
                : this(asset.name, asset.browser_download_url)
            {
            }

            public string Url { get; private set; }
            public bool IsFile { get; private set; }
        }

        #endregion

        #region ProgressBarWithText

        #endregion
    }

    public class ExtensionUpdateAvailableForm : UpdateAvailableForm
    {
        public ExtensionUpdateAvailableForm(Version version, UpdateCheckerSettings settings)
            : base(version, settings)
        {
            Text = "New Extensions available: " + version;
        }

        protected override void ForgetVersion()
        {
            Settings.ForgetExtensionVersion = true;
        }

        protected override SplitButtonToolStripItem DefaultDownloadFile()
        {
            return SplitMenuChoices.First(file => file.Name.Contains("Installer")) ?? SplitMenuChoices[0];
        }

        protected override void SetLastMenuChoiceUsed(string name)
        {
        }

        protected override List<string> ParseChangeLog(List<string> changelog)
        {
            return ChangelogHelper.ParseExtensionChangelog(changelog);
        }

        public override List<string> LoadPreviousChangelog()
        {
            return ChangelogHelper.LoadPreviousExtensionsChangelog();
        }
    }
}