using System;
using System.Collections.Generic;
using System.Windows.Forms;

namespace Mpdn.PlayerExtensions.Example
{
    public class HelloWorld : IPlayerExtension
    {
        private IPlayerControl m_PlayerControl;

        public ExtensionDescriptor Descriptor
        {
            get
            {
                return new ExtensionDescriptor
                {
                    Guid = new Guid("9714174F-B64D-43D8-BB16-52C5FEE2417B"),
                    Name = "Hello World",
                    Description = "Player Extension Example",
                    Copyright = "Copyright Example © 2014. All rights reserved."
                };
            }
        }

        public void Initialize(IPlayerControl playerControl)
        {
            m_PlayerControl = playerControl;
            m_PlayerControl.KeyDown += PlayerKeyDown;
        }

        public void Destroy()
        {
            m_PlayerControl.KeyDown -= PlayerKeyDown;
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

        private void HelloWorldClick()
        {
            MessageBox.Show(m_PlayerControl.Form, "Hello World!");
        }

        private void HelloWorld2Click()
        {
            MessageBox.Show(m_PlayerControl.Form, "Hello World Too!");
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
