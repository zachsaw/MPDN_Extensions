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
using System.Runtime.InteropServices;
using System.Security;
using DirectShowLib;
using Mpdn.Extensions.Framework;

namespace Mpdn.Extensions.PlayerExtensions
{
    public class YouTubeSourceProvider : PlayerExtension
    {
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

        private class YouTubeSource : ICustomSourceFilter
        {
            [ComImport, Guid("55C39876-FF76-4AB0-AAB0-0A46D535A26B")]
            private class YouTubeSourceFilter { }

            [ComImport, Guid("171252A0-8820-4AFE-9DF8-5C92B2D66B04")]
            private class LavSplitter { }

            [ComImport, Guid("FF762ACC-13EC-463A-A29C-FD4B0CD3E019")]
            [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
            [SuppressUnmanagedCodeSecurity]
            public interface ISupportedSites
            {
                [PreserveSig]
                [return: MarshalAs(UnmanagedType.U4)]
                uint GetVersion();
                [PreserveSig]
                [return: MarshalAs(UnmanagedType.U4)]
                uint GetCount();
                [PreserveSig]
                [return: MarshalAs(UnmanagedType.BStr)]
                string GetName(uint index);
                [PreserveSig]
                int Test([MarshalAs(UnmanagedType.LPWStr)] string url, bool explicitly, [MarshalAs(UnmanagedType.BStr)] out string canonicalUrl);
            }

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

            public static bool IsSiteSupported(Uri uri)
            {
                var filter = (IBaseFilter) new YouTubeSourceFilter();
                try
                {
                    var sites = filter as ISupportedSites;
                    if (sites == null)
                        return s_SupportedSites.Any(s => uri.Host.ToLowerInvariant().Contains(s));

                    string result;
                    return sites.Test(uri.AbsoluteUri, true, out result) == 0;
                }
                finally
                {
                    Marshal.ReleaseComObject(filter);
                }
            }

            public YouTubeSource(IGraphBuilder graph, string filename)
            {
                m_Filter = (IBaseFilter) new YouTubeSourceFilter();
                m_Splitter = (IBaseFilter) new LavSplitter();

                var fileSourceFilter = (IFileSourceFilter) m_Filter;
                DsError.ThrowExceptionForHR(fileSourceFilter.Load(filename, null));

                DsError.ThrowExceptionForHR(graph.AddFilter(m_Filter, "3DYD YouTube Source"));
                DsError.ThrowExceptionForHR(graph.AddFilter(m_Splitter, "LAV Splitter"));

                var outpins = GetPins(m_Filter, "Output");
                try
                {
                    switch (outpins.Length)
                    {
                        case 0:
                            throw new Exception("3DYD YouTube Source outpins.Length = 0 - Not expected!");
                        case 1:
                            ConnectPins(graph, outpins[0], m_Splitter, "Input");
                            VideoOutputPin = DsFindPin.ByName(m_Splitter, "Video");
                            AudioOutputPin = DsFindPin.ByName(m_Splitter, "Audio");
                            break;
                        case 2:
                            m_Splitter2 = (IBaseFilter) new LavSplitter();
                            DsError.ThrowExceptionForHR(graph.AddFilter(m_Splitter2, "LAV Splitter 2"));
                            ConnectPins(graph, outpins[0], m_Splitter, "Input");
                            ConnectPins(graph, outpins[1], m_Splitter2, "Input");
                            // Assume the first "Output" pin is video and the second is audio
                            VideoOutputPin = DsFindPin.ByName(m_Splitter, "Video");
                            AudioOutputPin = DsFindPin.ByName(m_Splitter2, "Audio");
                            break;
                        default:
                            throw new NotSupportedException(
                                "Not supported: More than 2 output pins from 3DYD YouTube Source!");
                    }
                    SubtitleOutputPins = GetPins(m_Filter, "OutputSS");
                    ExtendedSeeking = (IAMExtendedSeeking) m_Splitter;
                    VideoStreamSelect = (IAMStreamSelect) m_Splitter;
                    AudioStreamSelect = (IAMStreamSelect) m_Splitter;
                    SubtitleStreamSelect = null;
                }
                finally
                {
                    foreach (var pin in outpins)
                    {
                        Marshal.ReleaseComObject(pin);
                    }
                }
            }

            public void Dispose()
            {
                if (m_Disposed)
                    return;

                m_Disposed = true;

                if (VideoOutputPin != null)
                {
                    Marshal.ReleaseComObject(VideoOutputPin);
                }
                if (AudioOutputPin != null)
                {
                    Marshal.ReleaseComObject(AudioOutputPin);
                }
                foreach (var pin in SubtitleOutputPins)
                {
                    Marshal.ReleaseComObject(pin);
                }
                if (m_Filter != null)
                {
                    Marshal.ReleaseComObject(m_Filter);
                }
                if (m_Splitter != null)
                {
                    Marshal.ReleaseComObject(m_Splitter);
                }
                if (m_Splitter2 != null)
                {
                    Marshal.ReleaseComObject(m_Splitter2);
                }
            }

            private static IPin[] GetPins(IBaseFilter filter, string vPinName)
            {
                var result = new List<IPin>();
                var ppPins = new IPin[1];
                IEnumPins ppEnum;
                DsError.ThrowExceptionForHR(filter.EnumPins(out ppEnum));
                try
                {
                    while (ppEnum.Next(1, ppPins, IntPtr.Zero) == 0)
                    {
                        PinInfo pInfo;
                        DsError.ThrowExceptionForHR(ppPins[0].QueryPinInfo(out pInfo));
                        if (pInfo.name == vPinName)
                        {
                            result.Add(ppPins[0]);
                        }
                        else
                        {
                            Marshal.ReleaseComObject(ppPins[0]);
                        }
                        DsUtils.FreePinInfo(pInfo);
                    }
                }
                finally
                {
                    Marshal.ReleaseComObject(ppEnum);
                }
                return result.ToArray();
            }

            private static void ConnectPins(IGraphBuilder graphBuilder, IPin pinOut, IBaseFilter toFilter,
                string toPinName)
            {
                IPin pinIn = null;
                try
                {
                    pinIn = GetPin(toFilter, toPinName);
                    DsError.ThrowExceptionForHR(graphBuilder.ConnectDirect(pinOut, pinIn, null));
                }
                finally
                {
                    if (pinIn != null)
                    {
                        Marshal.ReleaseComObject(pinIn);
                    }
                }
            }

            private static IPin GetPin(IBaseFilter filter, string pinName)
            {
                var pin = DsFindPin.ByName(filter, pinName);
                if (pin == null)
                {
                    throw new Exception("Failed to get DirectShow filter pin " + pinName);
                }
                return pin;
            }

            public IPin VideoOutputPin { get; private set; }
            public IPin AudioOutputPin { get; private set; }
            public IPin[] SubtitleOutputPins { get; private set; }
            public IAMExtendedSeeking ExtendedSeeking { get; private set; }
            public IAMStreamSelect VideoStreamSelect { get; private set; }
            public IAMStreamSelect AudioStreamSelect { get; private set; }
            public IAMStreamSelect SubtitleStreamSelect { get; private set; }

            private readonly IBaseFilter m_Filter;
            private readonly IBaseFilter m_Splitter;
            private readonly IBaseFilter m_Splitter2;
            private bool m_Disposed;
        }

        private static void OnMediaLoading(object sender, MediaLoadingEventArgs e)
        {
            var filename = e.Filename;
            if (!IsYouTubeSource(filename))
                return;
            if (e.CustomSourceFilter != null)
                return;

            e.CustomSourceFilter = graph =>
            {
                try
                {
                    return new YouTubeSource(graph, filename);
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

            Uri uri;
            return Uri.TryCreate(fileNameOrUri, UriKind.Absolute, out uri) &&
                    (uri.Scheme == Uri.UriSchemeHttp || uri.Scheme == Uri.UriSchemeHttps) &&
                    YouTubeSource.IsSiteSupported(uri);
        }
    }
}
