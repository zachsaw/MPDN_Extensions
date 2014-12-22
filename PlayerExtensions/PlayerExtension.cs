using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Windows.Forms;

namespace Mpdn.PlayerExtensions
{
    public abstract class PlayerExtension : IPlayerExtension
    {
        protected IPlayerControl PlayerControl { get; private set; }
        private IDictionary<Keys, Action> Actions;

        public abstract ExtensionDescriptor Descriptor { get; }

        public abstract IList<Verb> Verbs { get; }

        #region Implementation

        public virtual void Initialize(IPlayerControl playerControl)
        {
            PlayerControl = playerControl;
            PlayerControl.KeyDown += PlayerKeyDown;

            Actions = new Dictionary<Keys, Action>();
            foreach (var verb in Verbs) {
                var keys = DecodeKeyString(verb.ShortcutDisplayStr);
                Actions.Add(keys, verb.Action);
            }
        }

        public virtual void Destroy()
        {
            PlayerControl.KeyDown -= PlayerKeyDown;
        }

        private void PlayerKeyDown(object sender, PlayerKeyEventArgs e)
        {
            Action action;
            if (Actions.TryGetValue(e.Key.KeyData, out action))
                action.Invoke();
        }

        private Keys DecodeKeyString(String keyString)
        {
            var keyWords = Regex.Split(keyString, @"\W+");
            keyString = String.Join(", ", keyWords.Select(DecodeKeyWord).ToArray());

            Keys keys;
            if (Enum.TryParse<Keys>(keyString, true, out keys))
                return keys;
            else
                throw new ArgumentException("Can't convert string to keys.");
        }

        private String DecodeKeyWord(String keyWord)
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
                default: return keyWord;
            }
        }

        #endregion
    }
}

