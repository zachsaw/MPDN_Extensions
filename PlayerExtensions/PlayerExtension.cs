﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using System.Windows.Forms;

namespace Mpdn.PlayerExtensions
{
    public abstract class PlayerExtension : IPlayerExtension
    {
        private readonly IDictionary<Keys, Action> m_Actions = new Dictionary<Keys, Action>();

        protected IPlayerControl PlayerControl { get; private set; }

        public abstract ExtensionDescriptor Descriptor { get; }

        public abstract IList<Verb> Verbs { get; }

        #region Implementation

        public void Initialize(IPlayerControl playerControl)
        {
            PlayerControl = playerControl;
            PlayerControl.KeyDown += PlayerKeyDown;

            LoadVerbs();
            Initialize();
        }

        public virtual void Initialize()
        {
        }

        public void LoadVerbs()
        {
            foreach (var verb in Verbs)
            {
                var shortcut = DecodeKeyString(verb.ShortcutDisplayStr);
                m_Actions.Add(shortcut, verb.Action);
            }
        }

        public virtual void Destroy()
        {
            PlayerControl.KeyDown -= PlayerKeyDown;
        }

        public virtual bool ShowConfigDialog(IWin32Window owner)
        {
            throw new NotImplementedException();
        }

        private void PlayerKeyDown(object sender, PlayerControlEventArgs<KeyEventArgs> e)
        {
            Action action;
            if (m_Actions.TryGetValue(e.InputArgs.KeyData, out action))
            {
                action();
            }
        }

        private static Keys DecodeKeyString(String keyString)
        {
            var keyWords = Regex.Split(keyString, @"\W+");
            keyString = String.Join(", ", keyWords.Select(DecodeKeyWord).ToArray());

            Keys keys;
            if (Enum.TryParse(keyString, true, out keys))
                return keys;

            throw new ArgumentException("Can't convert string to keys.");
        }

        private static String DecodeKeyWord(String keyWord)
        {
            switch (keyWord.ToLower())
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
                default: 
                    return keyWord;
            }
        }

        #endregion
    }
}