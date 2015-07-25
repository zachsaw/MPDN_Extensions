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
using System.Text.RegularExpressions;

namespace Mpdn.Extensions.PlayerExtensions.UpdateChecker
{
    public class Version
    {
        private static readonly Regex s_VersionRegex = new Regex(@"([0-9]+)\.([0-9]+)\.([0-9]+)");

        public Version()
        {
            ChangelogLines = new List<string>();
        }

        public Version(string version) : this()
        {
            var matches = s_VersionRegex.Match(version);
            Major = uint.Parse(matches.Groups[1].Value);
            Minor = uint.Parse(matches.Groups[2].Value);
            Revision = uint.Parse(matches.Groups[3].Value);
        }

        public uint Major { get; set; }
        public uint Minor { get; set; }
        public uint Revision { get; set; }
        public List<string> ChangelogLines { get; set; }

        public static bool ContainsVersionString(string text)
        {
            return s_VersionRegex.IsMatch(text);
        }

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
                return false;
            if (v2 == null)
                return true;
            if (v1 == v2)
                return false;
            var iv1 = GetInteger(v1);
            var iv2 = GetInteger(v2);
            return iv1 < iv2;
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
            return Equals(ChangelogLines, other.ChangelogLines) && Revision == other.Revision && Minor == other.Minor && Major == other.Major;
        }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != this.GetType()) return false;
            return Equals((Version) obj);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                var hashCode = (ChangelogLines != null ? ChangelogLines.GetHashCode() : 0);
                hashCode = (hashCode*397) ^ (int) Revision;
                hashCode = (hashCode*397) ^ (int) Minor;
                hashCode = (hashCode*397) ^ (int) Major;
                return hashCode;
            }
        }

        public override string ToString()
        {
            return Major + "." + Minor + "." + Revision;
        }

        public virtual List<UpdateAvailableForm.SplitButtonToolStripItem> GenerateSplitButtonItemList()
        {
            var list = new List<UpdateAvailableForm.SplitButtonToolStripItem>
            {
                new UpdateAvailableForm.SplitButtonToolStripItem("x64 Installer",
                    string.Format("{0}MediaPlayerDotNet_x64_Installer.exe", UpdateChecker.WebsiteUrl)),
                new UpdateAvailableForm.SplitButtonToolStripItem("x86 Installer",
                    string.Format("{0}MediaPlayerDotNet_x86_Installer.exe", UpdateChecker.WebsiteUrl)),
                new UpdateAvailableForm.SplitButtonToolStripItem("AnyCPU Zip",
                    string.Format("{0}MediaPlayerDotNet_AnyCPU.zip", UpdateChecker.WebsiteUrl)),
                new UpdateAvailableForm.SplitButtonToolStripItem("x64 Zip",
                    string.Format("{0}MediaPlayerDotNet_x64.zip", UpdateChecker.WebsiteUrl)),
                new UpdateAvailableForm.SplitButtonToolStripItem("x86 Zip",
                    string.Format("{0}MediaPlayerDotNet_x86.zip", UpdateChecker.WebsiteUrl)),
                new UpdateAvailableForm.SplitButtonToolStripItem("Open in Browser",
                    UpdateChecker.WebsiteUrl, false)
            };
            return list;
        }
    }

    public class GitHubVersion
    {
        public string tag_name { get; set; }
        public string body { get; set; }
        public List<GitHubAsset> assets { get; set; }

        public class GitHubAsset
        {
            public string name { get; set; }
            public string browser_download_url { get; set; }
        }
    }

    public class ExtensionVersion : Version
    {
        public ExtensionVersion()
        {
        }

        public ExtensionVersion(string version)
            : base(version)
        {
        }

        public List<GitHubVersion.GitHubAsset> Files { get; set; }

        public override List<UpdateAvailableForm.SplitButtonToolStripItem> GenerateSplitButtonItemList()
        {
            var list =
                Files.Select(gitHubAsset => new UpdateAvailableForm.SplitButtonToolStripItem(gitHubAsset)).ToList();
            list.Add(new UpdateAvailableForm.SplitButtonToolStripItem("Open in Browser",
                "https://github.com/zachsaw/MPDN_Extensions/releases", false));
            return list;
        }
    }
}