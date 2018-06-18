using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using System.Windows.Forms;

namespace Mpdn.Extensions.Framework
{
    public class ConcurrentSet<T> : IEnumerable<T>
    {
        private ConcurrentDictionary<T, bool> m_Dictionary;

        private static KeyValuePair<T, bool> CreateEntry(T item)
        {
            return new KeyValuePair<T, bool>(item, true);
        }

        public ConcurrentSet(IEnumerable<T> items)
        {
            m_Dictionary = new ConcurrentDictionary<T, bool>(items.Select(CreateEntry));
        }

        public bool TryAdd(T item)
        {
            return m_Dictionary.TryAdd(item, true);
        }

        public bool TryRemove(T item)
        {
            bool _;
            return m_Dictionary.TryRemove(item, out _);
        }

        public IEnumerator<T> GetEnumerator()
        {
            return m_Dictionary.ToArray().Select(x => x.Key).GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }

    public static class HotkeyRegister
    {
        public static EventHandler<PlayerControlEventArgs<KeyEventArgs>> OnKeyDown = OnKeyDownInternal;

        private static readonly ConcurrentDictionary<Keys, ConcurrentSet<Entry>> s_Hotkeys =
            new ConcurrentDictionary<Keys, ConcurrentSet<Entry>>();

        public static IDisposable AddOrUpdateHotkey(string keyString, Action action)
        {
            Keys keys;
            if (!HotkeyHelper.TryDecodeKeyString(keyString, out keys))
                return null;

            var entry = new Entry(action);
            s_Hotkeys.AddOrUpdate(keys,
                (_) => new ConcurrentSet<Entry>(new[] { entry }),
                (_, set) => { set.TryAdd(entry); return set; });

            return entry.GetReference();
        }

        private static void OnKeyDownInternal(object sender, PlayerControlEventArgs<KeyEventArgs> e)
        {
            ConcurrentSet<Entry> hotkeys;
            if (s_Hotkeys.TryGetValue(e.InputArgs.KeyData, out hotkeys))
            {
                foreach (var hotkey in hotkeys)
                    if (!hotkey.DoAction())
                        hotkeys.TryRemove(hotkey);
                e.Handled = true;
            }
        }

        private struct Entry
        {
            private readonly Reference m_Reference;
            private readonly Action m_Action;

            public Entry(Action action)
            {
                m_Reference = new Reference();
                m_Action = action;
            }

            public bool DoAction()
            {
                if (m_Reference.Disposed)
                    return false;

                m_Action();
                return true;
            }

            public IDisposable GetReference() { return m_Reference; }

            private sealed class Reference : IDisposable
            {
                public bool Disposed { get; private set; }

                ~Reference()
                {
                    Dispose();
                    GC.SuppressFinalize(this);
                }

                public void Dispose()
                {
                    Disposed = true;
                }
            }
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
