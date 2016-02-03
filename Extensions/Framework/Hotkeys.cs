using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using System.Windows.Forms;

namespace Mpdn.Extensions.Framework
{
    public struct Hotkey
    {
        public Keys Keys;
        public Action Action;

        public Hotkey(Keys keys, Action action)
        {
            Keys = keys;
            Action = action;
        }
    }

    public static class HotkeyRegister
    {
        public static event EventHandler HotkeysChanged;

        private static readonly Dictionary<Guid, List<Hotkey>> s_Hotkeys = new Dictionary<Guid, List<Hotkey>>();

        public static IEnumerable<Hotkey> Hotkeys
        {
            get { return s_Hotkeys.SelectMany(x => x.Value); }
        }

        public static void RegisterHotkey(Guid guid, string hotkey, Action action)
        {
            Keys keys;
            if (!HotkeyHelper.TryDecodeKeyString(hotkey, out keys))
                return;

            List<Hotkey> list;
            if (!s_Hotkeys.TryGetValue(guid, out list))
                s_Hotkeys.Add(guid, list = new List<Hotkey>());

            list.Add(new Hotkey(keys, action));
            OnHotkeysChanged();
        }

        public static void DeregisterHotkey(Guid guid)
        {
            s_Hotkeys.Remove(guid);
            OnHotkeysChanged();
        }

        public static void OnHotkeysChanged()
        {
            if (HotkeysChanged != null)
                HotkeysChanged(null, EventArgs.Empty);
        }
    }

    public static class HotkeyHelper
    {
        public static Keys SafeDecodeKeyString(string keyString)
        {
            Keys keys;
            return TryDecodeKeyString(keyString, out keys) ? keys : Keys.None;
        }

        public static bool TryDecodeKeyString(string keyString, out Keys keys)
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

    }
}
