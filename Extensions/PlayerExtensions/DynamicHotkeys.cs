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
using System.Linq;
using Mpdn.Extensions.Framework;

namespace Mpdn.Extensions.PlayerExtensions
{
    public class DynamicHotkeys : PlayerExtension
    {
        private static IList<Verb> s_Verbs = new List<Verb>();

        public override void Initialize()
        {
            base.Initialize();
            HotkeyRegister.HotkeysChanged += OnHotkeysChanged;
        }

        public override void Destroy()
        {
            HotkeyRegister.HotkeysChanged -= OnHotkeysChanged;
            base.Destroy();
        }

        public void OnHotkeysChanged(object sender, EventArgs e)
        {
            s_Verbs = HotkeyRegister.Hotkeys.Select(
                (hotkey, i) => new Verb(Category.Window, "Dynamic Hotkey " + i.ToString(), "", hotkey.Keys.ToString(), "", hotkey.Action)).ToList();
            LoadVerbs();
        }

        // TODO implement Verb hotkeys using HotkeyRegister instead of the other way around.
        public override IList<Verb> Verbs { get { return s_Verbs; } }

        public override ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("29CBA419-591F-4CEB-9BC1-41D592F5F203"),
                    Name = "DynamicHotkeys",
                    Description = "Allows scripts to dynamically add and remove hotkeys.",
                    Copyright = ""
                };
            }
        }
    }
}
