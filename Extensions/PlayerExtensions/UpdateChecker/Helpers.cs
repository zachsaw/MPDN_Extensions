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
        public static string GetPlayerArtchitecture()
        {
            var appArch = Assembly.GetEntryAssembly().GetName().ProcessorArchitecture;
            string arch;
            switch (appArch)
            {
                case ProcessorArchitecture.MSIL:
                    arch = "AnyCPU";
                    break;
                case ProcessorArchitecture.Amd64:
                    arch = "x64";
                    break;
                default:
                    arch = "x86";
                    break;
            }
            return arch;
        }
    }
}