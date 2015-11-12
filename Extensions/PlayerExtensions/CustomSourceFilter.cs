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
using System.Runtime.InteropServices;
using DirectShowLib;

namespace Mpdn.Extensions.PlayerExtensions
{
    public class CustomSourceFilter : ICustomSourceFilter
    {
        [ComImport, Guid("171252A0-8820-4AFE-9DF8-5C92B2D66B04")]
        protected class LavSplitter { }

        private bool m_Disposed;

        ~CustomSourceFilter()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
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
        }

        protected static IPin[] GetPins(IBaseFilter filter, string vPinName)
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

        protected static void ConnectPins(IGraphBuilder graphBuilder, IPin pinOut, IBaseFilter toFilter,
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

        protected static IPin GetPin(IBaseFilter filter, string pinName)
        {
            var pin = DsFindPin.ByName(filter, pinName);
            if (pin == null)
            {
                throw new Exception("Failed to get DirectShow filter pin " + pinName);
            }
            return pin;
        }

        public IPin VideoOutputPin { get; protected set; }
        public IPin AudioOutputPin { get; protected set; }
        public IPin[] SubtitleOutputPins { get; protected set; }
        public IAMExtendedSeeking ExtendedSeeking { get; protected set; }
        public IAMStreamSelect VideoStreamSelect { get; protected set; }
        public IAMStreamSelect AudioStreamSelect { get; protected set; }
        public IAMStreamSelect SubtitleStreamSelect { get; protected set; }
    }
}
