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
using System.Windows.Forms;
using Mpdn.RenderScript;

namespace Mpdn.Extensions.Framework
{
    public static class RenderScript
    {
        public static IRenderScriptUi Empty = new NullRenderScriptUi();

        public static bool IsEmpty(this IRenderScriptUi script)
        {
            return script is NullRenderScriptUi;
        }

        private class NullRenderScriptUi : IRenderScriptUi
        {
            public int Version
            {
                get { return Extension.InterfaceVersion; }
            }

            public ExtensionUiDescriptor Descriptor
            {
                get
                {
                    return new ExtensionUiDescriptor
                    {
                        Guid = Guid.Empty,
                        Name = "None",
                        Description = "Do not use render script"
                    };
                }
            }

            public IRenderScript CreateScript()
            {
                return null;
            }

            public void Initialize()
            {
            }

            public void Destroy()
            {
            }

            public bool HasConfigDialog()
            {
                return false;
            }

            public bool ShowConfigDialog(IWin32Window owner)
            {
                return false;
            }
        }
    }
}