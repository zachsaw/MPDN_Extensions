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
using System.Runtime.InteropServices;
using DirectShowLib;
using Mpdn.Extensions.Framework;

namespace Mpdn.Extensions.PlayerExtensions
{
    public class RarFileSourceProvider : PlayerExtension
    {
        private RarFileSourceFilter m_RarFileSourceFilter;

        public override ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("8B61465F-D892-4C29-A99A-ED34C0426CCE"),
                    Name = "RAR File Source Provider",
                    Description = "Play media files from RAR archive"
                };
            }
        }

        public override void Initialize()
        {
            m_RarFileSourceFilter = new RarFileSourceFilter();

            Media.Loading += OnMediaLoading;
        }

        public override void Destroy()
        {
            DisposeHelper.Dispose(ref m_RarFileSourceFilter);

            Media.Loading -= OnMediaLoading;
        }

        private class RarFileSource : ICustomSourceFilter
        {
            [ComImport, Guid("171252A0-8820-4AFE-9DF8-5C92B2D66B04")]
            private class LavSplitter { }

            public RarFileSource(RarFileSourceFilter rarFileSourceFilter, IGraphBuilder graph, string filename)
            {
                m_Filter = rarFileSourceFilter.CreateInstance();
                m_Splitter = (IBaseFilter) new LavSplitter();

                var sourceFilter = (IFileSourceFilter) m_Filter;
                DsError.ThrowExceptionForHR(sourceFilter.Load(filename, null));

                DsError.ThrowExceptionForHR(graph.AddFilter(m_Filter, "RAR File Source Filter"));
                DsError.ThrowExceptionForHR(graph.AddFilter(m_Splitter, "LAV Splitter"));

                var outpins = GetPins(m_Filter, "Output");

                ConnectPins(graph, outpins[0], m_Splitter, "Input");
                VideoOutputPin = DsFindPin.ByName(m_Splitter, "Video");
                AudioOutputPin = DsFindPin.ByName(m_Splitter, "Audio");
                SubtitleOutputPins = GetPins(m_Splitter, "Subtitle");

                ExtendedSeeking = (IAMExtendedSeeking) m_Splitter;
                VideoStreamSelect = (IAMStreamSelect) m_Splitter;
                AudioStreamSelect = (IAMStreamSelect) m_Splitter;
                SubtitleStreamSelect = null;
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
            private bool m_Disposed;
        }

        private void OnMediaLoading(object sender, MediaLoadingEventArgs e)
        {
            var filename = e.Filename;
            if (!IsRarFile(filename))
                return;

            e.CustomSourceFilter = graph =>
            {
                try
                {
                    return new RarFileSource(m_RarFileSourceFilter, graph, filename);
                }
                catch (Exception ex)
                {
                    Trace.WriteLine(ex);
                    return null;
                }
            };
        }

        private static bool IsRarFile(string filename)
        {
            if (string.IsNullOrWhiteSpace(filename))
                return false;

            return PathHelper.GetExtension(filename).ToLowerInvariant() == ".rar";
        }

        public class RarFileSourceFilter : DynamicDirectShowFilter
        {
            private const string FILTER_CLSID = "8B61465F-D892-4C29-A99A-ED34C0426CCF";
            private static readonly Guid s_ClsId = new Guid(FILTER_CLSID);

            protected override string FilterName
            {
                get { return "RarFileSourceFilter"; }
            }

            protected override Guid FilterClsId
            {
                get { return s_ClsId; }
            }
        }
    }
}
