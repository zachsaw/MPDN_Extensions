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

namespace Mpdn.Extensions.PlayerExtensions.UpdateChecker
{
    public class UpdateCheckerExtension : PlayerExtension<UpdateCheckerSettings, UpdateCheckerConfigDialog>
    {
        private Version m_CurrentVersion;
        private UpdateChecker m_Checker;
        private ExtensionUpdateChecker m_ExtChecker;

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
                    new Verb(Category.Help, String.Empty, "Check for Updates...", ManualUpdateCheck)
                };
            }
        }

        private void ManualUpdateCheck()
        {
            var newVersion = false;
            using (new HourGlass())
            {
                m_Checker.CheckVersion();
                m_ExtChecker.CheckVersion();
            }

            if (Settings.MpdnVersionOnServer > m_CurrentVersion)
            {
                new UpdateAvailableForm(Settings.MpdnVersionOnServer, Settings).ShowDialog(Gui.VideoBox);
                newVersion = true;
            }

            if (Settings.ExtensionVersionOnServer > ExtensionUpdateChecker.GetExtensionsVersion())
            {
                new ExtensionUpdateAvailableForm(Settings.ExtensionVersionOnServer, Settings).ShowDialog(
                    Gui.VideoBox);
                newVersion = true;
            }
            
            if (!newVersion)
            {
                MessageBox.Show(Gui.VideoBox, "You have the latest release.");
            }
        }

        public override void Initialize()
        {
            base.Initialize();
            m_Checker = new UpdateChecker(Settings, new Uri("http://mpdn.zachsaw.com/LatestVersion.txt"));
            m_ExtChecker = new ExtensionUpdateChecker(Settings, new Uri("https://api.github.com/repos/zachsaw/MPDN_Extensions/releases/latest"));
            Player.Loaded += PlayerControlPlayerLoaded;
        }

        private void PlayerControlPlayerLoaded(object sender, EventArgs e)
        {
            if (!Settings.CheckForUpdate)
                return;

            m_CurrentVersion = new Version(Application.ProductVersion);
            if (!Settings.ForgetMpdnVersion && Settings.MpdnVersionOnServer > m_CurrentVersion)
            {
                new UpdateAvailableForm(Settings.MpdnVersionOnServer, Settings).ShowDialog(Gui.VideoBox);
            }
            if (!Settings.ForgetExtensionVersion &&
                Settings.ExtensionVersionOnServer > ExtensionUpdateChecker.GetExtensionsVersion())
            {
                new ExtensionUpdateAvailableForm(Settings.ExtensionVersionOnServer, Settings).ShowDialog(
                    Gui.VideoBox);
            }
            m_Checker.CheckVersionAsync();
            m_ExtChecker.CheckVersionAsync();

        }

        public override void Destroy()
        {
            base.Destroy();
            Player.Loaded -= PlayerControlPlayerLoaded;
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
        public Version MpdnVersionOnServer { get; set; }
        public ExtensionVersion ExtensionVersionOnServer { get; set; }
        public bool ForgetMpdnVersion { get; set; }
        public bool ForgetExtensionVersion { get; set; }
    }

    public class UpdateChecker
    {
        public static readonly string WebsiteUrl = "http://mpdn.zachsaw.com/Latest/";
        protected readonly UpdateCheckerSettings Settings;
        protected readonly WebClient WebClient = new WebClient();
        protected readonly Uri ChangelogUrl;

        public UpdateChecker(UpdateCheckerSettings settings, Uri url)
        {
            Settings = settings;
            WebClient.DownloadStringCompleted += DownloadStringCompleted;
            ChangelogUrl = url;
        }

        protected void SetHeaders()
        {
            var version = ExtensionUpdateChecker.GetExtensionsVersion();

            WebClient.Headers.Add("User-Agent",
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
            Version serverVersion = null;
            foreach (var line in Regex.Split(changelog, "\r\n|\r|\n"))
            {
                if (Version.ContainsVersionString(line))
                {
                    if (serverVersion == null)
                    {
                        serverVersion = new Version(line);
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

            GuiThread.DoAsync(() =>
            {
                if (Settings.MpdnVersionOnServer == serverVersion)
                    return;

                Settings.MpdnVersionOnServer = serverVersion;
                Settings.ForgetMpdnVersion = false;
            });
        }

        public void CheckVersion()
        {
            SetHeaders();
            var changelog = WebClient.DownloadString(ChangelogUrl);
            ParseChangelog(changelog);

        }
        public void CheckVersionAsync()
        {
            SetHeaders();
            // DownloadStringAsync isn't fully async!
            // It blocks when it is detecting proxy settings and especially noticeable if user is behind a proxy server
            Task.Factory.StartNew(() => WebClient.DownloadStringAsync(ChangelogUrl));
        }
    }

    public class ExtensionUpdateChecker : UpdateChecker
    {
        public ExtensionUpdateChecker(UpdateCheckerSettings settings, Uri url)
            : base(settings, url)
        {
        }

        public static Version GetExtensionsVersion()
        {
            var asm = typeof(UpdateChecker).Assembly;
            var ver = FileVersionInfo.GetVersionInfo(asm.Location);
            return new Version(ver.ToString());
        }

        protected override void ParseChangelog(string changelog)
        {
            var result = JsonConvert.DeserializeObject<GitHubVersion>(changelog);
            var version = new ExtensionVersion(result.tag_name);
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

            GuiThread.DoAsync(() =>
            {
                if (Settings.ExtensionVersionOnServer == version)
                    return;

                Settings.ExtensionVersionOnServer = version;
                Settings.ForgetExtensionVersion = false;
            });
        }
    }
}