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
using System.Net;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Windows.Forms;
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.Framework.Controls;
using Newtonsoft.Json;

namespace Mpdn.Extensions.PlayerExtensions
{
    public class UpdateCheckerExtension : PlayerExtension<UpdateCheckerSettings, UpdateCheckerConfigDialog>
    {
        private UpdateChecker m_checker;
        private ExtensionUpdateChecker m_extChecker;

        public override ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("9294ff0e-cbb1-49d4-9721-d1a438852968"),
                    Name = "Update Checker",
                    Description = "Check for new version of MPDN"
                };
            }
        }

        public override IList<Verb> Verbs
        {
            get
            {
                return new[]
                {
                    new Verb(Category.Help, String.Empty, "Check for Update...", ManualUpdateCheck)
                };
            }
        }

        private void ManualUpdateCheck()
        {
            var newVersion = false;
            using (new HourGlass())
            {
                m_checker.CheckVersion();
                m_extChecker.CheckVersion();
            }

            if (Settings.MpdnVersionOnServer > VersionHelpers.ApplicationVersion)
            {
                new UpdateCheckerNewVersionForm(Settings.MpdnVersionOnServer, Settings).ShowDialog(PlayerControl.VideoPanel);
                newVersion = true;
            }

            if (Settings.ExtensionVersionOnServer > VersionHelpers.ExtensionDllVersion)
            {
                new UpdateCheckerNewExtensionForm(Settings.ExtensionVersionOnServer, Settings).ShowDialog(
                    PlayerControl.VideoPanel);
                newVersion = true;
            }
            
            if (!newVersion)
            {
                MessageBox.Show(PlayerControl.VideoPanel, "You have the latest release.");
            }
        }

        public override void Initialize()
        {
            base.Initialize();
            m_checker = new UpdateChecker(Settings, new Uri("http://mpdn.zachsaw.com/LatestVersion.txt"));
            m_extChecker = new ExtensionUpdateChecker(Settings, new Uri("https://api.github.com/repos/zachsaw/MPDN_Extensions/releases/latest"));
            PlayerControl.PlayerLoaded += PlayerControl_PlayerLoaded;
        }

        private void PlayerControl_PlayerLoaded(object sender, EventArgs e)
        {
            if (!Settings.CheckForUpdate)
                return;

            if (!Settings.ForgetMpdnVersion && Settings.MpdnVersionOnServer > VersionHelpers.ApplicationVersion)
            {
                new UpdateCheckerNewVersionForm(Settings.MpdnVersionOnServer, Settings).ShowDialog(PlayerControl.VideoPanel);
            }
            if (!Settings.ForgetExtensionVersion &&
                Settings.ExtensionVersionOnServer > VersionHelpers.ExtensionDllVersion)
            {
                new UpdateCheckerNewExtensionForm(Settings.ExtensionVersionOnServer, Settings).ShowDialog(
                    PlayerControl.VideoPanel);
            }
            m_checker.CheckVersionAsync();
            m_extChecker.CheckVersionAsync();

        }

        public override void Destroy()
        {
            base.Destroy();
            PlayerControl.PlayerLoaded -= PlayerControl_PlayerLoaded;
        }
    }

    public class UpdateCheckerSettings
    {
        public UpdateCheckerSettings()
        {
            CheckForUpdate = true;
            ForgetMpdnVersion = false;
        }
        
        public bool CheckForUpdate { get; set; }
        public VersionHelpers.Version MpdnVersionOnServer { get; set; }
        public VersionHelpers.ExtensionVersion ExtensionVersionOnServer { get; set; }
        public bool ForgetMpdnVersion { get; set; }
        public bool ForgetExtensionVersion { get; set; }
    }

    #region UpdateChecker

    public class UpdateChecker
    {
        public static readonly string WebsiteUrl = "http://mpdn.zachsaw.com/Latest/";
        protected readonly UpdateCheckerSettings m_settings;
        protected readonly WebClient m_WebClient = new WebClient();
        protected readonly Uri ChangelogUrl;

        public UpdateChecker(UpdateCheckerSettings settings, Uri url)
        {
            m_settings = settings;
            m_WebClient.DownloadStringCompleted += DownloadStringCompleted;
            ChangelogUrl = url;
        }

        protected void SetHeaders()
        {
            var version = VersionHelpers.ExtensionDllVersion;

            m_WebClient.Headers.Add("User-Agent",
                string.Format(
                    "Mozilla/5.0 (compatible; Windows NT {0}; MPDN/{1}; MPDN_Extensions/{2}; +http://mpdn.zachsaw.com/)",
                    Environment.OSVersion.Version, Application.ProductVersion, version));
        }

        private void DownloadStringCompleted(object sender, DownloadStringCompletedEventArgs e)
        {
            string changelog;
            try
            {
                changelog = e.Result;
            }
            catch (Exception)
            {
                return;
            }
          
            ParseChangelog(changelog);
        }

        protected virtual void ParseChangelog(string changelog)
        {
            VersionHelpers.Version serverVersion = null;
            foreach (var line in Regex.Split(changelog, "\r\n|\r|\n"))
            {
                if (VersionHelpers.Version.VERSION_REGEX.IsMatch(line))
                {
                    if (serverVersion == null)
                    {
                        serverVersion = new VersionHelpers.Version(line);
                    }
                    else
                    {
                        break;
                    }
                }
                else if (serverVersion != null)
                {
                    serverVersion.Changelog += line.Trim() + Environment.NewLine;
                }
            }

            PlayerControl.VideoPanel.BeginInvoke((MethodInvoker) (() =>
            {
                if (m_settings.MpdnVersionOnServer == serverVersion)
                    return;

                m_settings.MpdnVersionOnServer = serverVersion;
                m_settings.ForgetMpdnVersion = false;
            }));
        }

        public void CheckVersion()
        {
            SetHeaders();
            var changelog = m_WebClient.DownloadString(ChangelogUrl);
            ParseChangelog(changelog);

        }
        public void CheckVersionAsync()
        {
            SetHeaders();
            // DownloadStringAsync isn't fully async!
            // It blocks when it is detecting proxy settings and especially noticeable if user is behind a proxy server
            Task.Factory.StartNew(() => m_WebClient.DownloadStringAsync(ChangelogUrl));
        }
    }
    #region ExtensionUpdateChecker

    public class ExtensionUpdateChecker : UpdateChecker
    {
        public ExtensionUpdateChecker(UpdateCheckerSettings settings, Uri url) : base(settings, url)
        {
        }

        protected override void ParseChangelog(string changelog)
        {
            var result = JsonConvert.DeserializeObject<VersionHelpers.GitHubVersion>(changelog);
            var version = new VersionHelpers.ExtensionVersion(result.tag_name);
            var changelogStarted = false;
            foreach (string line in Regex.Split(result.body, "\r\n|\r|\n"))
            {
                if (changelogStarted)
                {
                    version.Changelog += line.Trim() + Environment.NewLine;
                }
                if (line.StartsWith("#### Changelog"))
                {
                    changelogStarted = true;
                }
            }
            version.Files = result.assets;

            PlayerControl.VideoPanel.BeginInvoke((MethodInvoker)(() =>
            {
                if (m_settings.ExtensionVersionOnServer == version)
                    return;

                m_settings.ExtensionVersionOnServer = version;
                m_settings.ForgetExtensionVersion = false;
            }));

        }

    }

    #endregion
    #endregion
}