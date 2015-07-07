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
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.RegularExpressions;
using System.Windows.Forms;

namespace Mpdn.Extensions.Framework
{
    public static class DisposeHelper
    {
        public static void Dispose(object obj)
        {
            var disposable = obj as IDisposable;
            if (disposable != null)
            {
                disposable.Dispose();
            }
        }

        public static void Dispose<T>(ref T obj) where T : class, IDisposable
        {
            if (obj != null)
            {
                obj.Dispose();
                obj = default(T);
            }
        }
    }

    public static class PathHelper
    {
        public static string GetDirectoryName(string path)
        {
            if (path == null)
            {
                throw new ArgumentNullException("path");
            }

            return Path.GetDirectoryName(path) ?? Path.GetPathRoot(path);
        }

        public static string GetExtension(string path)
        {
            if (path == null)
            {
                throw new ArgumentNullException("path");
            }

            var result = Path.GetExtension(path);
            if (result == null)
                throw new ArgumentException();

            return result;
        }

        public static string ExtensionsPath
        {
            get
            {
                return Path.Combine(GetDirectoryName(Assembly.GetAssembly(typeof (IPlayerExtension)).Location),
                    "Extensions");
            }
        }
    }

    public static class EnumHelpers
    {
        public static string ToDescription(this Enum en)
        {
            var type = en.GetType();
            var enumString = en.ToString();

            var memInfo = type.GetMember(enumString);

            if (memInfo.Length > 0)
            {
                var attrs = memInfo[0].GetCustomAttributes(typeof (DescriptionAttribute), false);

                if (attrs.Length > 0)
                {
                    return ((DescriptionAttribute) attrs[0]).Description;
                }
            }

            return enumString;
        }

        public static string[] GetDescriptions<T>()
        {
            return Enum.GetValues(typeof (T)).Cast<Enum>().Select(val => val.ToDescription()).ToArray();
        }
    }

    public static class RendererHelpers
    {
        public static TextureFormat GetTextureFormat(this RenderQuality quality)
        {
            switch (quality)
            {
                case RenderQuality.MaxPerformance:
                    return TextureFormat.Unorm8;
                case RenderQuality.Performance:
                    return TextureFormat.Float16;
                case RenderQuality.Quality:
                    return TextureFormat.Unorm16;
                case RenderQuality.MaxQuality:
                    return TextureFormat.Float32;
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        public static bool IsFullRange(this YuvColorimetric colorimetric)
        {
            return !IsLimitedRange(colorimetric);
        }

        public static bool IsLimitedRange(this YuvColorimetric colorimetric)
        {
            switch (colorimetric)
            {
                case YuvColorimetric.FullRangePc601: 
                case YuvColorimetric.FullRangePc709: 
                case YuvColorimetric.FullRangePc2020:
                    return false;
                case YuvColorimetric.ItuBt601: 
                case YuvColorimetric.ItuBt709: 
                case YuvColorimetric.ItuBt2020:
                    return true;
                default: 
                    throw new ArgumentOutOfRangeException();
            }
        }

        public static bool IsYuv(this FrameBufferInputFormat format)
        {
            return !IsRgb(format);
        }

        public static bool IsRgb(this FrameBufferInputFormat format)
        {
            switch (format)
            {
                case FrameBufferInputFormat.Rgb24:
                case FrameBufferInputFormat.Rgb32:
                case FrameBufferInputFormat.Rgb48:
                    return true;
            }

            return false;
        }

        public static int GetBitDepth(this FrameBufferInputFormat format)
        {
            switch (format)
            {
                case FrameBufferInputFormat.Nv12:
                case FrameBufferInputFormat.Yv12:
                case FrameBufferInputFormat.Yuy2:
                case FrameBufferInputFormat.Uyvy:
                case FrameBufferInputFormat.Yv24:
                case FrameBufferInputFormat.Ayuv:
                case FrameBufferInputFormat.Rgb24:
                case FrameBufferInputFormat.Rgb32:
                    return 8;
                case FrameBufferInputFormat.P010:
                case FrameBufferInputFormat.P210:
                case FrameBufferInputFormat.Y410:
                    return 10;
                case FrameBufferInputFormat.P016:
                case FrameBufferInputFormat.P216:
                case FrameBufferInputFormat.Y416:
                case FrameBufferInputFormat.Rgb48:
                    return 16;
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }
    }

    public static class StringHelpers
    {
        public static string SubstringIdx(this string self, int startIndex, int endIndex)
        {
            var length = endIndex - startIndex;
            if (length < 0)
            {
                length = 0;
            }
            return self.Substring(startIndex, length);
        }
    }

    public static class VersionHelpers
    {
        public static readonly Version ApplicationVersion = new Version(Application.ProductVersion);
        public static readonly Version ExtensionDllVersion;

        static VersionHelpers()
        {
            var asm = typeof(VersionHelpers).Assembly;
            var ver = FileVersionInfo.GetVersionInfo(asm.Location);
            ExtensionDllVersion = new Version(ver.ToString());
        }
        #region VersionObjects
        #region GithubObjects
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
        #endregion
        public class Version
        {
            public static readonly Regex VERSION_REGEX = new Regex(@"([0-9]+)\.([0-9]+)\.([0-9]+)");

            public Version()
            {
            }

            public Version(string version)
            {
                var matches = VERSION_REGEX.Match(version);
                Major = UInt32.Parse(matches.Groups[1].Value);
                Minor = UInt32.Parse(matches.Groups[2].Value);
                Revision = UInt32.Parse(matches.Groups[3].Value);
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
                return (int)(((v.Major & 0xFF) << 24) + (v.Minor << 12) + v.Revision);
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

     
        #endregion
    }


}
