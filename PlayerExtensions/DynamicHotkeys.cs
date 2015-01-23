using System;
using System.Collections.Generic;
using System.Linq;

namespace Mpdn.PlayerExtensions
{
    public class DynamicHotkeys : PlayerExtension
    {
        protected static IList<Verb> m_Verbs = new List<Verb>();
        protected static Action Reload;

        public override IList<Verb> Verbs { get { return m_Verbs; } }

        public override void Initialize()
        {
            Reload = LoadVerbs;
            base.Initialize();
        }

        public static void RegisterHotkey(string caption, string shortcut, Action action, string hint = "")
        {
            m_Verbs.Add(new Verb(Category.Window, "Dynamic Hotkeys", caption, shortcut, hint, action));
            Reload();
        }

        public static void RemoveHotkey(string caption)
        {
            m_Verbs = m_Verbs.Where(v => v.Caption != caption).ToList();
        }

        protected override PlayerExtensionDescriptor ScriptDescriptor
        {
            get
            {
                return new PlayerExtensionDescriptor
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
