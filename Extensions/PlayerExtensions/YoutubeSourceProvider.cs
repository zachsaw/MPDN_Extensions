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
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using DirectShowLib;
using Mpdn.Extensions.Framework;

namespace Mpdn.Extensions.PlayerExtensions
{
    public class YouTubeSourceProvider : PlayerExtension
    {
        private static readonly string[] s_SupportedSites =
        {
            "youtube.com",
            "vimeo.com",
            "dailymotion.com",
            "liveleak.com",
            "break.com",
            "metacafe.com",
            "veoh.com",
            "facebook.com",
            "ebaumsworld.com",
            "vkmag.com",
            "blip.tv",
            "godtube.com",
            "streetfire.net",
            "g4tv.com",
            "tcmag.com",
            "dailyhaha.com",
            "bofunk.com",
            "mediabom.tv",
            "tedxtalks.ted.com"
        };

        [ComImport, Guid("55C39876-FF76-4AB0-AAB0-0A46D535A26B")]
        private class YouTubeSourceFilter
        {
        }

        public override ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("D45CC2FA-2094-45D7-A035-B1A4F8C26F1C"),
                    Name = "YouTube Source Provider",
                    Description = "Provides MPDN with Custom YouTube Source Filter"
                };
            }
        }

        public override void Initialize()
        {
            Media.Loading += OnMediaLoading;
        }

        public override void Destroy()
        {
            Media.Loading -= OnMediaLoading;
        }

        private static void OnMediaLoading(object sender, MediaLoadingEventArgs e)
        {
            if (!IsYouTubeSource(e.Filename))
                return;

            e.CustomSourceFilter = delegate
            {
                try
                {
                    return (IBaseFilter) new YouTubeSourceFilter();
                }
                catch (Exception ex)
                {
                    // User may not have 3DYD YouTubeSource filter installed
                    Trace.WriteLine(ex);
                    return null;
                }
            };
        }

        private static bool IsYouTubeSource(string fileNameOrUri)
        {
            if (string.IsNullOrWhiteSpace(fileNameOrUri))
                return false;

            try
            {
                var uri = new Uri(fileNameOrUri);
                return uri.Scheme != Uri.UriSchemeFile &&
                       s_SupportedSites.Any(s => uri.Host.ToLowerInvariant().Contains(s));
            }
            catch (UriFormatException)
            {
                return false;
            }
        }
    }
}
