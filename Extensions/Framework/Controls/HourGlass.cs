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
using System.Runtime.InteropServices;
using System.Windows.Forms;

namespace Mpdn.Extensions.Framework.Controls
{
    public class HourGlass : IDisposable
    {
        private const int SETCURSOR = 0x0020;

        [DllImport("user32.dll", CharSet = CharSet.Auto, SetLastError = false)]
        private static extern IntPtr SendMessage(IntPtr hWnd, int msg, int wParam, int lParam);

        public HourGlass()
        {
            Enabled = true;
        }

        public void Dispose()
        {
            Enabled = false;
        }

        public static bool Enabled
        {
            get { return Application.UseWaitCursor; }
            set
            {
                if (value == Application.UseWaitCursor)
                    return;

                Application.UseWaitCursor = value;
                var f = Form.ActiveForm;
                if (f != null)
                {
                    SendMessage(f.Handle, SETCURSOR, f.Handle.ToInt32(), 1);
                }
            }
        }
    }
}
