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
using System.Net;
using System.Text.RegularExpressions;
using System.Windows.Forms;
using OpenSubtitles.PlayerExtensions;

namespace Mpdn.PlayerExtensions.GitHub
{
    public class UpdateCheckerExtension : PlayerExtension<UpdateCheckerSettings, UpdateCheckerConfigDialog>
    {
        private UpdateChecker.Version m_currentVersion;
        private UpdateChecker m_checker;

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
                    new Verb(Category.View, string.Empty, "Toggle Update Checker", "Ctrl+Shift+U", string.Empty,
                        ToggleUpdateChecker)
                };
            }
        }

        public override void Initialize()
        {
            base.Initialize();
            m_checker = new UpdateChecker(Settings);
            PlayerControl.PlayerLoaded += PlayerControl_PlayerLoaded;
        }

        private void PlayerControl_PlayerLoaded(object sender, EventArgs e)
        {
            if (!Settings.CheckForUpdate)
                return;
            m_currentVersion = new UpdateChecker.Version(Application.ProductVersion);
            if (!Settings.ForgetVersion && Settings.VersionOnServer > m_currentVersion)
            {
                new UpdateCheckerNewVersionForm(Settings.VersionOnServer, Settings).ShowDialog(PlayerControl.Form);
            }
            m_checker.CheckVersion();

        }

        public override void Destroy()
        {
            base.Destroy();
            PlayerControl.PlayerLoaded -= PlayerControl_PlayerLoaded;
        }

        private void ToggleUpdateChecker()
        {
            Settings.CheckForUpdate = !Settings.CheckForUpdate;
        }
    }

    public class UpdateCheckerSettings
    {
        public UpdateCheckerSettings()
        {
            CheckForUpdate = true;
            ForgetVersion = false;
        }


        public bool CheckForUpdate { get; set; }
        public UpdateChecker.Version VersionOnServer { get; set; }
        public bool ForgetVersion { get; set; }
    }
    #region UpdateChecker

    public class UpdateChecker
    {
        private static readonly Uri ChangelogUrl = new Uri("http://mpdn.zachsaw.com/Latest/ChangeLog.txt");
        public static readonly string WebsiteUrl = "http://mpdn.zachsaw.com/Latest/";
        private readonly UpdateCheckerSettings m_settings;
        private readonly WebClient m_WebClient = new WebClient();

        public UpdateChecker(UpdateCheckerSettings settings)
        {
            m_settings = settings;
            m_WebClient.DownloadStringCompleted += m_WebClient_DownloadStringCompleted;
        }

        private void m_WebClient_DownloadStringCompleted(object sender, DownloadStringCompletedEventArgs e)
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
            if (m_settings.VersionOnServer != serverVersion)
            {
                m_settings.VersionOnServer = serverVersion;
                m_settings.ForgetVersion = false;
            }
        }

        public void CheckVersion()
        {
            m_WebClient.DownloadStringAsync(ChangelogUrl);
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
                if (v1.Major > v2.Major)
                    return true;
                if (v1.Minor > v2.Minor)
                    return true;
                if (v1.Revision > v2.Revision)
                    return true;
                return false;
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
                return "v" + Major + "." + Minor + "." + Revision;
            }
        }

        #endregion
    }
    #endregion
}