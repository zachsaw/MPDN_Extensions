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
using System.Net;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using System.Windows.Forms;
using CommonMark;
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.Framework.Controls;
using Newtonsoft.Json;

namespace Mpdn.Extensions.PlayerExtensions.UpdateChecker
{
    [ComVisible(true)]
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
            SetChangelog(version);
            changelogViewerWebBrowser.IsWebBrowserContextMenuEnabled = false;
        }

        protected static List<string> HtmlHeaders
        {
            get
            {
                return new List<string>
                {
                    "<!doctype html>",
                    "<html>",
                    "<head>",
                    "<style>" +
                    "body { background: #fff; margin: 0 auto; } " +
                    "h1 { font-size: 15px; color: #1562b6; padding-top: 5px; border: 0px !important; border-bottom: 2px solid #1562b6 !important; }" +
                    "h2 { font-size: 13px; color: #1562b6; padding-top: 5px; border: 0px !important; border-bottom: 1px solid #1562b6 !important; }" +
                    ".center {text-align: center}" +
                    "</style>",
                    "</head>",
                    "<body>"
                };
            }
        }

        public override sealed string Text
        {
            get { return base.Text; }
            set { base.Text = value; }
        }

        private void SetChangelog(Version version)
        {
            var lines = HtmlHeaders;
            lines.AddRange(ParseChangeLog(version.ChangelogLines));
            lines.Add(
                "<div class=\"center\"><a href=\"#\" onclick=\"window.external.LoadPreviousChangeLog();\">Load previous changelogs</a></div>");
            lines.Add("</body>");
            lines.Add("</html>");
            changelogViewerWebBrowser.DocumentText = string.Join("\n", lines);
            changelogViewerWebBrowser.ObjectForScripting = this;
        }

        public virtual void LoadPreviousChangeLog()
        {
            var html = HtmlHeaders;
            using (new HourGlass())
            {
                var webClient = new WebClient();
                WebClientHelper.SetHeaders(webClient);
                var changelog = webClient.DownloadString(string.Format("{0}ChangeLog.txt", UpdateChecker.LatestFolderUrl));
                html.Add("<h1>Changelogs</h1>");
                foreach (var line in Regex.Split(changelog, "\r\n|\r|\n"))
                {
                    if (line.Contains("Changelog") && Version.ContainsVersionString(line))
                    {
                        html.Add(string.Format("<h2>{0}</h2><ol>", line));
                    } else if (string.IsNullOrWhiteSpace(line))
                    {
                        html.Add("</ol>");
                    }
                    else
                    {
                        html.Add(string.Format("<li>{0}</li>", line));
                    }
                }
            }
            html.Add("</body>");
            html.Add("</html>");
            changelogViewerWebBrowser.DocumentText = string.Join("\n", html);
        }

        protected virtual List<string> ParseChangeLog(List<string> changelog)
        {
            var lines = new List<string> {"<h1>Changelog</h1>", "<div id='changelog'><ol>"};

            lines.AddRange(
                changelog.Select(line => string.IsNullOrWhiteSpace(line) ? null : string.Format("<li>{0}</li>", line)));
            lines.Add("</ol></div>");
            return lines;
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

          return SplitMenuChoices.First(file => file.Name.Contains(ArchitectureHelper.GetPlayerArtchitecture().ToString())) ?? SplitMenuChoices[0];
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

        #region ProgressBarWithText

        #endregion

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
    }

    [ComVisible(true)]
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
            var lines = new List<string>
            {
                "<h1>Changelog</h1>",
                "<div id='changelog'>",
                CommonMarkConverter.Convert(string.Join("\n", changelog)),
                "</div>"
            };
            return lines;
        }

        public override void LoadPreviousChangeLog()
        {

            var html = HtmlHeaders;
            html.Add("<h1>Changelogs</h1>");
            using (new HourGlass())
            {
                var webClient = new WebClient();
                WebClientHelper.SetHeaders(webClient);
                var releases = JsonConvert.DeserializeObject<List<GitHubVersion>>(webClient.DownloadString("https://api.github.com/repos/zachsaw/MPDN_Extensions/releases"));
                foreach (var gitHubVersion in releases)
                {
                    html.Add(string.Format("<h2>{0}</h2>", gitHubVersion.tag_name));
                    html.Add(CommonMarkConverter.Convert(gitHubVersion.body));
                }
            }
            html.Add("</body>");
            html.Add("</html>");
            changelogViewerWebBrowser.DocumentText = string.Join("\n", html);

        }
    }
}