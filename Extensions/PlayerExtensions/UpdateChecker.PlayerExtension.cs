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
        private UpdateChecker.Version m_currentVersion;
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
            using (new HourGlass())
            {
                m_checker.CheckVersion();
                m_extChecker.CheckVersion();
            }
            if (Settings.MpdnVersionOnServer > m_currentVersion)
            {
                new UpdateCheckerNewVersionForm(Settings.MpdnVersionOnServer, Settings).ShowDialog(PlayerControl.VideoPanel);
            }
            else
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

            m_currentVersion = new UpdateChecker.Version(Application.ProductVersion);
            if (!Settings.ForgetMpdnVersion && Settings.MpdnVersionOnServer > m_currentVersion)
            {
                new UpdateCheckerNewVersionForm(Settings.MpdnVersionOnServer, Settings).ShowDialog(PlayerControl.VideoPanel);
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
        public UpdateChecker.Version MpdnVersionOnServer { get; set; }
        public ExtensionUpdateChecker.ExtensionVersion ExtensionVersionOnServer { get; set; }
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
            var version = ExtensionUpdateChecker.GetExtensionsVersion();

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
            Version serverVersion = null;
            foreach (var line in Regex.Split(changelog, "\r\n|\r|\n"))
            {
                if (Version.VERSION_REGEX.IsMatch(line))
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

        #region Version

        public class Version
        {
            public static readonly Regex VERSION_REGEX = new Regex(@"([0-9]+)\.([0-9]+)\.([0-9]+)");

            public Version()
            {
            }

            public Version(string version)
            {
                var matches = VERSION_REGEX.Match(version);
                Major = uint.Parse(matches.Groups[1].Value);
                Minor = uint.Parse(matches.Groups[2].Value);
                Revision = uint.Parse(matches.Groups[3].Value);
            }

            public uint Major { get; set; }
            public uint Minor { get; set; }
            public uint Revision { get; set; }
            public string Changelog { get; set; }


            public static bool operator >(Version v1, Version v2)
            {
                if (v1 == null)
                    return false;
                if (v2 == null)
                    return true;
                if (v1 == v2)
                    return false;
                var iv1 = GetInteger(v1);
                var iv2 = GetInteger(v2);
                return iv1 > iv2;
            }

            private static int GetInteger(Version v)
            {
                return (int) (((v.Major & 0xFF) << 24) + (v.Minor << 12) + v.Revision);
            }

            public static bool operator <(Version v1, Version v2)
            {
                if (v1 == null)
                    return true;
                if (v2 == null)
                    return false;
                if (v1 == v2)
                    return false;

                if (v1.Major < v2.Major)
                    return true;
                if (v1.Minor < v2.Minor)
                    return true;
                if (v1.Revision < v2.Revision)
                    return true;
                return false;
            }


            public static bool operator ==(Version v1, Version v2)
            {
                if (ReferenceEquals(null, v1) && ReferenceEquals(null, v2))
                    return true;
                if (ReferenceEquals(null, v1) || ReferenceEquals(null, v2))
                    return false;

                return v1.Equals(v2);
            }

            public static bool operator !=(Version v1, Version v2)
            {
                return !(v1 == v2);
            }

            protected bool Equals(Version other)
            {
                return Major == other.Major && Minor == other.Minor && Revision == other.Revision;
            }

            public override bool Equals(object obj)
            {
                if (ReferenceEquals(null, obj)) return false;
                if (ReferenceEquals(this, obj)) return true;
                if (obj.GetType() != this.GetType()) return false;
                return Equals((Version)obj);
            }

            public override int GetHashCode()
            {
                unchecked
                {
                    var hashCode = (int)Major;
                    hashCode = (hashCode * 397) ^ (int)Minor;
                    hashCode = (hashCode * 397) ^ (int)Revision;
                    return hashCode;
                }
            }

            public override string ToString()
            {
                return Major + "." + Minor + "." + Revision;
            }
        }

        #endregion
    }
    #region ExtensionUpdateChecker

    public class ExtensionUpdateChecker : UpdateChecker
    {
        public ExtensionUpdateChecker(UpdateCheckerSettings settings, Uri url) : base(settings, url)
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

            PlayerControl.VideoPanel.BeginInvoke((MethodInvoker)(() =>
            {
                if (m_settings.ExtensionVersionOnServer == version)
                    return;

                m_settings.ExtensionVersionOnServer = version;
                m_settings.ForgetExtensionVersion = false;
            }));

        }
        public class GitHubVersion
        {
            public class GitHubAsset
            {
                public string name { get; set; }
                public string browser_download_url { get; set; }
            }
            public string tag_name { get; set; }
            public string body { get; set; }
            public List<GitHubAsset> assets { get; set; }
        }

        public class ExtensionVersion : Version
        {
            public ExtensionVersion(string version) : base(version)
            {
            }

            public List<GitHubVersion.GitHubAsset> Files { get; set; }
        }

    }
    #endregion
    #endregion
}