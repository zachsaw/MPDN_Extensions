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

using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Forms;
using Mpdn.Extensions.Framework;

namespace Mpdn.Extensions.PlayerExtensions
{
    public class DynamicHotkeys : PlayerExtension
    {
        private IDictionary<Keys, Action> m_Actions = new Dictionary<Keys, Action>();

        public override void Initialize()
        {
            base.Initialize();
            Player.KeyDown += HotkeyRegister.OnKeyDown;
        }

        public override void Destroy()
        {
            Player.KeyDown -= HotkeyRegister.OnKeyDown;
            base.Destroy();
        }

        public override ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("29CBA419-591F-4CEB-9BC1-41D592F5F203"),
                    Name = "DynamicHotkeys",
                    Description = "Handles hotkeys.",
                    Copyright = ""
                };
            }
        }
    }
}
