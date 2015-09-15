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
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Windows.Forms;
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.Framework.Controls;
using Mpdn.Extensions.PlayerExtensions.Exceptions;
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
                    new Verb(Category.Help, string.Empty, "Check for Updates...", ManualUpdateCheck)
                };
            }
        }

        private void ManualUpdateCheck()
        {
            using (new HourGlass())
            {
                try
                {
                    m_Checker.CheckVersion();
                    m_ExtChecker.CheckVersion();
                }
                catch (InternetConnectivityException e)
                {
                    MessageBox.Show(Gui.VideoBox, 
                        string.Format("You need an internet connection to check for updates:\n{0}", e.InnerException.Message),
                        "Internet Connection", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    Trace.WriteLine(e);
                    return;
                }
             
            }
            
            if (!DisplayUpdateForm(true))
            {
                MessageBox.Show(Gui.VideoBox, "You have the latest release.", "Up-to-date",MessageBoxButtons.OK,MessageBoxIcon.Information);
            }
        }

        public override void Initialize()
        {
            base.Initialize();
            m_Checker = new UpdateChecker(Settings, new Uri(string.Format("{0}LatestVersion.txt", UpdateChecker.MpdnRepoUrl)));
            m_ExtChecker = new ExtensionUpdateChecker(Settings, new Uri("https://api.github.com/repos/zachsaw/MPDN_Extensions/releases/latest"));
            Player.Loaded += PlayerControlPlayerLoaded;
        }

        private void PlayerControlPlayerLoaded(object sender, EventArgs e)
        {
            if (!Settings.CheckForUpdate)
                return;

            m_CurrentVersion = new Version(Application.ProductVersion);

            DisplayUpdateForm();

            m_Checker.CheckVersionAsync();
            m_ExtChecker.CheckVersionAsync();

        }

        private bool DisplayUpdateForm(bool force = false)
        {
            var playerNeedUpdate = Settings.MpdnVersionOnServer > m_CurrentVersion;
            var extensionNeedUpdate = Settings.ExtensionVersionOnServer > ExtensionUpdateChecker.GetExtensionsVersion();
            //Check API Version match when both updates available
            if (playerNeedUpdate && extensionNeedUpdate &&
                Settings.MpdnVersionOnServer.ExtensionApiVersion !=
                Settings.ExtensionVersionOnServer.ExtensionApiVersion)
            {
                return false;
            }
            //Don't update player if the update is going to break the extensions.
            if (playerNeedUpdate && Settings.MpdnVersionOnServer.ExtensionApiVersion != Extension.InterfaceVersion)
            {
                return false;
            }
            //Don't update the extension if the new extensions aren't going to work with the current player.
            if (extensionNeedUpdate &&
                Settings.ExtensionVersionOnServer.ExtensionApiVersion != Extension.InterfaceVersion)
            {
                return false;
            }
            return Settings.UseSimpleUpdate
                ? DisplaySimpleForm(force, playerNeedUpdate, extensionNeedUpdate)
                : DisplayAdvancedForm(force, playerNeedUpdate, extensionNeedUpdate);
        }

        private bool DisplayAdvancedForm(bool force, bool playerNeedUpdate, bool extensionNeedUpdate)
        {
            var newVersion = false;
            if ((force || !Settings.ForgetMpdnVersion) && playerNeedUpdate)
            {
                new UpdateAvailableForm(Settings.MpdnVersionOnServer, Settings).ShowDialog(Gui.VideoBox);
                newVersion = true;
            }

            if ((force || !Settings.ForgetExtensionVersion) &&
                extensionNeedUpdate)
            {
                new ExtensionUpdateAvailableForm(Settings.ExtensionVersionOnServer, Settings).ShowDialog(
                    Gui.VideoBox);
                newVersion = true;
            }

            return newVersion;
        }

        private bool DisplaySimpleForm(bool force, bool playerNeedUpdate, bool extensionNeedUpdate)
        {
            if ((force || !Settings.ForgetMpdnVersion && !Settings.ForgetExtensionVersion)
                && playerNeedUpdate && extensionNeedUpdate)
            {
                new SimpleUpdateForm(SimpleUpdateForm.UpdateType.Both, Settings).ShowDialog(Gui.VideoBox);
                return true;
            }

            if ((force || !Settings.ForgetMpdnVersion) && playerNeedUpdate)
            {
                new SimpleUpdateForm(SimpleUpdateForm.UpdateType.Player, Settings).ShowDialog(Gui.VideoBox);
                return true;
            }

            if ((force || !Settings.ForgetExtensionVersion) && extensionNeedUpdate)
            {
                new SimpleUpdateForm(SimpleUpdateForm.UpdateType.Extensions, Settings).ShowDialog(Gui.VideoBox);
                return true;
            }
            return false;
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
            UseSimpleUpdate = RegistryHelper.IsPlayerInstalled();
        }
        
        public bool CheckForUpdate { get; set; }
        public Version MpdnVersionOnServer { get; set; }
        public ExtensionVersion ExtensionVersionOnServer { get; set; }
        public bool ForgetMpdnVersion { get; set; }
        public bool ForgetExtensionVersion { get; set; }
        public string LastMpdnReleaseChosen { get; set; }
        public bool UseSimpleUpdate { get; set; }
    }

    public class UpdateChecker
    {
        public static readonly string MpdnRepoUrl = "http://mpdn.zachsaw.com/";
        public static readonly string LatestFolderUrl = string.Format("{0}Latest/", MpdnRepoUrl);
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
           WebClientHelper.SetHeaders(WebClient);
        }

        private void DownloadStringCompleted(object sender, DownloadStringCompletedEventArgs e)
        {
            if (e.Error != null)
            {
                Trace.WriteLine(e.Error);
                return;
            }
            var changelog = e.Result;
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
                else if (serverVersion != null && !string.IsNullOrWhiteSpace(line))
                {
                    var extensionApiVersion = Version.GetExtensionApiVersion(line);
                    if (extensionApiVersion != -1)
                    {
                        serverVersion.ExtensionApiVersion = extensionApiVersion;
                    }
                    else
                    {
                        serverVersion.ChangelogLines.Add(line);
                    }
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
            try
            {
                var changelog = WebClient.DownloadString(ChangelogUrl);
                ParseChangelog(changelog);
            }
            catch (WebException e)
            {
               throw new InternetConnectivityException("No connection", e);
            }
           

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
            foreach (var line in Regex.Split(result.body, "\r\n|\r|\n").Where(line => !string.IsNullOrWhiteSpace(line)))
            {
                if (changelogStarted)
                {
                    version.ChangelogLines.Add(line);
                }
                else
                {
                    var extensionApiVersion = Version.GetExtensionApiVersion(line);
                    if (extensionApiVersion != -1)
                    {
                        version.ExtensionApiVersion = extensionApiVersion;
                    }
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