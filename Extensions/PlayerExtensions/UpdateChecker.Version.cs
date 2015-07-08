using System.Collections.Generic;
using System.Text.RegularExpressions;

namespace Mpdn.Extensions.PlayerExtensions.UpdateChecker
{
    public class Version
    {
        private static readonly Regex s_VersionRegex = new Regex(@"([0-9]+)\.([0-9]+)\.([0-9]+)");

        public Version()
        {
        }

        public Version(string version)
        {
            var matches = s_VersionRegex.Match(version);
            Major = uint.Parse(matches.Groups[1].Value);
            Minor = uint.Parse(matches.Groups[2].Value);
            Revision = uint.Parse(matches.Groups[3].Value);
        }

        public uint Major { get; set; }
        public uint Minor { get; set; }
        public uint Revision { get; set; }
        public string Changelog { get; set; }

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
            var iv1 = GetInteger(this);
            var iv2 = GetInteger(other);
            return iv1 == iv2;
        }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != GetType()) return false;
            return Equals((Version) obj);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                var hashCode = (int) Major;
                hashCode = (hashCode*397) ^ (int) Minor;
                hashCode = (hashCode*397) ^ (int) Revision;
                return hashCode;
            }
        }

        public override string ToString()
        {
            return Major + "." + Minor + "." + Revision;
        }
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
        public ExtensionVersion()
        {
        }

        public ExtensionVersion(string version)
            : base(version)
        {
        }

        public List<GitHubVersion.GitHubAsset> Files { get; set; }
    }
}