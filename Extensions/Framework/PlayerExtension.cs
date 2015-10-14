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
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Text.RegularExpressions;
using System.Windows.Forms;
using Mpdn.Extensions.Framework.Config;

namespace Mpdn.Extensions.Framework
{
    public abstract class PlayerExtension : PlayerExtension<NoSettings> { }

    public abstract class PlayerExtension<TSettings> : PlayerExtension<TSettings, ScriptConfigDialog<TSettings>>
        where TSettings : class, new()
    { }

    public abstract class PlayerExtension<TSettings, TDialog> : ExtensionUi<Config.Internal.PlayerExtensions, TSettings, TDialog>, IPlayerExtension
        where TSettings : class, new()
        where TDialog : ScriptConfigDialog<TSettings>, new()
    {
        #region Implementation

        public virtual IList<Verb> Verbs
        {
            get { return new Verb[0]; }
        }

        public override void Initialize()
        {
            base.Initialize();

            Player.KeyDown += PlayerKeyDown;
            LoadVerbs();
        }

        public override void Destroy()
        {
            Player.KeyDown -= PlayerKeyDown;

            base.Destroy();
        }

        private readonly IDictionary<Keys, Action> m_Actions = new Dictionary<Keys, Action>();

        protected void LoadVerbs()
        {
            m_Actions.Clear();
            foreach (var verb in Verbs)
            {
                var shortcut = DecodeKeyString(verb.ShortcutDisplayStr);
                if (shortcut == Keys.None)
                    continue;

                m_Actions.Remove(shortcut); // Prevent duplicates FIFO.
                m_Actions.Add(shortcut, verb.Action);
            }
        }

        private void PlayerKeyDown(object sender, PlayerControlEventArgs<KeyEventArgs> e)
        {
            Action action;
            if (m_Actions.TryGetValue(e.InputArgs.KeyData, out action))
            {
                action();
            }
        }

        protected static bool TryDecodeKeyString(string keyString, out Keys keys)
        {
            keys = Keys.None;
            if (string.IsNullOrWhiteSpace(keyString))
                return false;

            keyString = keyString.ToLower().Trim();
            var keyWords = Regex.Split(keyString, @"\W+");
            var specialKeys = AddSpecialKeys(keyString);
            keyString = string.Join(", ",
                keyWords.Concat(specialKeys).Where(k => !string.IsNullOrWhiteSpace(k)).Select(DecodeKeyWord).ToArray());

            return (Enum.TryParse(keyString, true, out keys));
        }

        private static IEnumerable<string> AddSpecialKeys(string keyString)
        {
            var specialKeys = new List<string>();
            if (keyString.Length >= 1)
            {
                var lastChar = keyString[keyString.Length - 1];
                switch (lastChar)
                {
                    case '+':
                    case '-':
                        specialKeys.Add(new string(lastChar, 1));
                        break;
                }
            }
            return specialKeys;
        }

        private static Keys DecodeKeyString(string keyString)
        {
            Keys keys;
            return TryDecodeKeyString(keyString, out keys) ? keys : Keys.None;
        }

        private static string DecodeKeyWord(string keyWord)
        {
            switch (keyWord)
            {
                case "ctrl":
                    return "Control";
                case "0":
                    return "D0";
                case "1":
                    return "D1";
                case "2":
                    return "D2";
                case "3":
                    return "D3";
                case "4":
                    return "D4";
                case "5":
                    return "D5";
                case "6":
                    return "D6";
                case "7":
                    return "D7";
                case "8":
                    return "D8";
                case "9":
                    return "D9";
                case "+":
                    return "Add";
                case "-":
                    return "Subtract";
                default:
                    return keyWord;
            }
        }

        #endregion
    }


    public class AboutExtensions : PlayerExtension
    {
        public override ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("AB0556BA-E743-48FD-8D3E-CDCAFD66E637"),
                    Name = "About MPDN Extensions",
                    Description = "View the about box of MPDN Extensions"
                };
            }
        }

        public override IList<Verb> Verbs
        {
            get
            {
                return new[]
                {
                    new Verb(Category.Help, string.Empty, "About MPDN Extensions...", string.Empty, string.Empty, ShowAboutBox)
                };
            }
        }

        private static void ShowAboutBox()
        {
            // TODO custom form for about box
            var version = FileVersionInfo.GetVersionInfo(typeof (AboutExtensions).Assembly.Location).FileVersion;
            MessageBox.Show(Gui.VideoBox,
                string.Format("MPDN Extensions version {0}\r\n\r\nLicense: Open Source LGPLv3", version),
                "About MPDN Extensions", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }
    }
}
