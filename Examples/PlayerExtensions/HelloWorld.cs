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
using System.Windows.Forms;

namespace Mpdn.Extensions.PlayerExtensions
{
    public class HelloWorld : IPlayerExtension
    {
        public ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("9714174F-B64D-43D8-BB16-52C5FEE2417B"),
                    Name = "Hello World",
                    Description = "Player Extension Example",
                    Copyright = "Copyright Example © 2014-2015. All rights reserved."
                };
            }
        }

        public void Initialize()
        {
            PlayerControl.KeyDown += PlayerKeyDown;
        }

        public void Destroy()
        {
            PlayerControl.KeyDown -= PlayerKeyDown;
        }

        public IList<Verb> Verbs
        {
            get
            {
                return new[]
                {
                    new Verb(Category.Help, string.Empty, "Hello World", "Ctrl+Shift+H", "Say hello world", HelloWorldClick),
                    new Verb(Category.Help, "My subcategory", "Another Hello World", "Ctrl+Shift+Y", "Say hello world too", HelloWorld2Click)
                };
            }
        }
        
        public bool HasConfigDialog()
        {
            return false;
        }

        public bool ShowConfigDialog(IWin32Window owner)
        {
            return false;
        }

        private void HelloWorldClick()
        {
            MessageBox.Show(PlayerControl.VideoPanel, "Hello World!");
        }

        private void HelloWorld2Click()
        {
            MessageBox.Show(PlayerControl.VideoPanel, "Hello World Too!");
        }

        private void PlayerKeyDown(object sender, PlayerControlEventArgs<KeyEventArgs> e)
        {
            switch (e.InputArgs.KeyData)
            {
                case Keys.Control | Keys.Shift | Keys.H:
                    HelloWorldClick();
                    break;
                case Keys.Control | Keys.Shift | Keys.Y:
                    HelloWorld2Click();
                    break;
            }
        }
    }
}
