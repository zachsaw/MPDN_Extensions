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
using System.Linq;
using System.Net;
using System.Reflection;
using System.Text.RegularExpressions;
using System.Windows.Forms;
using CommonMark;
using Microsoft.Win32;
using Mpdn.Extensions.Framework.Controls;
using Newtonsoft.Json;

namespace Mpdn.Extensions.PlayerExtensions.UpdateChecker
{
    public static class WebClientHelper
    {
        public static void SetHeaders(WebClient client)
        {
            var version = ExtensionUpdateChecker.GetExtensionsVersion();

            client.Headers.Add("User-Agent",
                string.Format(
                    "Mozilla/5.0 (compatible; Windows NT {0}; MPDN/{1}; MPDN_Extensions/{2}; +http://mpdn.zachsaw.com/)",
                    Environment.OSVersion.Version, Application.ProductVersion, version));
        }
    }

    public static class ArchitectureHelper
    {
        public enum Architecture
        {
            x64,
            x86,
            AnyCPU
        }

        public static Architecture GetPlayerArtchitecture()
        {
            var appArch = Assembly.GetEntryAssembly().GetName().ProcessorArchitecture;
            Architecture arch;
            switch (appArch)
            {
                case ProcessorArchitecture.MSIL:
                    arch = Architecture.AnyCPU;
                    break;
                case ProcessorArchitecture.Amd64:
                    arch = Architecture.x64;
                    break;
                default:
                    arch = Architecture.x86;
                    break;
            }
            return arch;
        }
    }

    public static class RegistryHelper
    {
        public enum RegistryRoot
        {
            HKLM,
            HKCU
        }
        public static bool RegistryValueExists(RegistryRoot root, string subKey, string valueName)
        {
            RegistryKey registryKey;
            switch (root)
            {
                case RegistryRoot.HKLM:
                    registryKey = Registry.LocalMachine.OpenSubKey(subKey, false);
                    break;
                case RegistryRoot.HKCU:
                    registryKey = Registry.CurrentUser.OpenSubKey(subKey, false);
                    break;
                default:
                    throw new InvalidOperationException(
                        "parameter subKey must be either \"HKLM\" or \"HKCU\"");
            }

            return registryKey != null && registryKey.GetValue(valueName) != null;
        }

        public static bool IsPlayerInstalled()
        {
            return RegistryValueExists(RegistryRoot.HKLM,
                string.Format("SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\MediaPlayerDotNet_{0}",
                    ArchitectureHelper.GetPlayerArtchitecture())
                , "DisplayVersion");
        }
    }

    public static class ChangelogHelper
    {
        public static List<string> ParseMpdnChangelog(List<string> changelog)
        {
            var lines = new List<string> {"<h1>Changelog</h1>", "<div id='changelog'><ol>"};

            lines.AddRange(
                changelog.Select(line => string.IsNullOrWhiteSpace(line) ? null : string.Format("<li>{0}</li>", line)));
            lines.Add("</ol></div>");
            return lines;
        }

        public static List<string> LoadPreviousMpdnChangelog()
        {
            var html = new List<string>();
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
                    }
                    else if (string.IsNullOrWhiteSpace(line))
                    {
                        html.Add("</ol>");
                    }
                    else
                    {
                        html.Add(string.Format("<li>{0}</li>", line));
                    }
                }
            }
            return html;
        }

        public static List<string> ParseExtensionChangelog(List<string> changelog)
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

        public static List<string> LoadPreviousExtensionsChangelog()
        {
            var html = new List<string> {"<h1>Changelogs</h1>"};
            using (new HourGlass())
            {
                var webClient = new WebClient();
                WebClientHelper.SetHeaders(webClient);
                var releases =
                    JsonConvert.DeserializeObject<List<GitHubVersion>>(
                        webClient.DownloadString("https://api.github.com/repos/zachsaw/MPDN_Extensions/releases"));
                foreach (var gitHubVersion in releases)
                {
                    html.Add(string.Format("<h2>{0}</h2>", gitHubVersion.tag_name));
                    html.Add(CommonMarkConverter.Convert(gitHubVersion.body));
                }
            }
            return html;
        }
    }
}