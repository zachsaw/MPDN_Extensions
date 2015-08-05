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
using System.Net;
using System.Reflection;
using System.Windows.Forms;
using Microsoft.Win32;

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
                    throw new System.InvalidOperationException(
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
}