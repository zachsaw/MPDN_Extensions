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
using System.Runtime.InteropServices;
using DirectShowLib;
using Mpdn.Extensions.Framework;

namespace Mpdn.Extensions.PlayerExtensions
{
    public class RarFileSourceProvider : PlayerExtension
    {
        private const string RAR_FILE_EXT = ".rar";

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

            Player.RegisterMediaFileExtension(RAR_FILE_EXT);
            Media.Loading += OnMediaLoading;
        }

        public override void Destroy()
        {
            DisposeHelper.Dispose(ref m_RarFileSourceFilter);

            Player.UnregisterMediaFileExtension(RAR_FILE_EXT);
            Media.Loading -= OnMediaLoading;
        }

        private class RarFileSource : CustomSourceFilter
        {
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

            protected override void Dispose(bool disposing)
            {
                if (m_Disposed)
                    return;

                m_Disposed = true;

                if (m_Filter != null)
                {
                    Marshal.ReleaseComObject(m_Filter);
                }
                if (m_Splitter != null)
                {
                    Marshal.ReleaseComObject(m_Splitter);
                }
            }

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

            return PathHelper.GetExtension(filename).ToLowerInvariant() == RAR_FILE_EXT;
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
