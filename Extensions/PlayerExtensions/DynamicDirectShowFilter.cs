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
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;
using DirectShowLib;
using Mpdn.Extensions.Framework;

namespace Mpdn.Extensions.PlayerExtensions
{
    public abstract class DynamicDirectShowFilter : IDisposable
    {
        protected abstract string FilterName { get; }
        protected abstract Guid FilterClsId { get; }

        private IntPtr m_FilterLib = IntPtr.Zero;

        public IBaseFilter CreateInstance()
        {
            return (IBaseFilter)CreateFilter();
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        private object CreateFilter()
        {
            bool succeeded = false;
            try
            {
                if (m_FilterLib == IntPtr.Zero)
                {
                    var path = Path.Combine(PathHelper.ExtensionsPath, "Libs", "Native", "DirectShowFilters",
                        FilterName, (Environment.Is64BitProcess ? "x64" : "x86"), FilterName + ".ax");
                    m_FilterLib = LoadLibrary(path);
                    if (m_FilterLib == IntPtr.Zero)
                    {
                        throw new Win32Exception();
                    }
                }

                IntPtr fnP = GetProcAddress(m_FilterLib, "DllGetClassObject");
                if (fnP == IntPtr.Zero)
                {
                    throw new Win32Exception();
                }

                var getClassObjectFn =
                    (DllGetClassObjectDelegate)
                        Marshal.GetDelegateForFunctionPointer(fnP, typeof(DllGetClassObjectDelegate));

                object pUnk;
                var iidIUnknown = new Guid("00000000-0000-0000-C000-000000000046");
                Marshal.ThrowExceptionForHR(getClassObjectFn(FilterClsId, iidIUnknown, out pUnk));

                var pCf = (IClassFactory)pUnk;
                if (pCf == null)
                {
                    throw new Win32Exception();
                }
                IntPtr resultPtr;
                pCf.CreateInstance(IntPtr.Zero, ref iidIUnknown, out resultPtr);
                if (resultPtr == IntPtr.Zero)
                {
                    throw new Win32Exception();
                }
                succeeded = true;
                return Marshal.GetObjectForIUnknown(resultPtr);
            }
            finally
            {
                if (!succeeded)
                {
                    Dispose(true);
                }
            }
        }

        ~DynamicDirectShowFilter()
        {
            Dispose(false);
        }

        protected void Dispose(bool disposing)
        {
            if (m_FilterLib == IntPtr.Zero)
                return;

            FreeLibrary(m_FilterLib);
            m_FilterLib = IntPtr.Zero;
        }

        [DllImport("kernel32", SetLastError = true, CharSet = CharSet.Auto)]
        private static extern IntPtr LoadLibrary(string lpFileName);

        [DllImport("kernel32", CharSet = CharSet.Ansi, ExactSpelling = true, SetLastError = true)]
        private static extern IntPtr GetProcAddress(IntPtr hModule, string procName);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool FreeLibrary(IntPtr hModule);

        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private delegate int DllGetClassObjectDelegate(
            [MarshalAs(UnmanagedType.LPStruct)] Guid rclsid,
            [MarshalAs(UnmanagedType.LPStruct)] Guid riid,
            [MarshalAs(UnmanagedType.IUnknown, IidParameterIndex = 1)] out object ppv);

        [ComImport]
        [Guid("00000001-0000-0000-C000-000000000046")]
        [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
        private interface IClassFactory
        {
            [PreserveSig]
            int CreateInstance(IntPtr pUnkOuter, ref Guid riid, out IntPtr ppvObject);

            [PreserveSig]
            int LockServer(bool fLock);
        }
    }
}
