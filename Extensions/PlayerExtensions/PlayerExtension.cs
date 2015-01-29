﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using System.Windows.Forms;
using Mpdn.PlayerExtensions.Config;

namespace Mpdn.PlayerExtensions
{
    public class ScriptConfigDialog<TSettings> : Form
    where TSettings : class, new()
    {
        protected TSettings Settings { get; private set; }

        public virtual void Setup(TSettings settings)
        {
            Settings = settings;

            LoadSettings();
        }

        protected virtual void LoadSettings()
        {
            // This needs to be overriden
            throw new NotImplementedException();
        }

        protected virtual void SaveSettings()
        {
            // This needs to be overriden
            throw new NotImplementedException();
        }

        protected override void OnFormClosed(FormClosedEventArgs e)
        {
            base.OnFormClosed(e);

            if (DialogResult != DialogResult.OK)
                return;

            SaveSettings();
        }
    }

    public abstract class PlayerExtension : PlayerExtension<object> { }

    public abstract class PlayerExtension<TSettings> : PlayerExtension<TSettings, ScriptConfigDialog<TSettings>>
        where TSettings : class, new()
    { }

    public abstract class PlayerExtension<TSettings, TDialog> : ExtensionUi<TSettings, TDialog>, IPlayerExtension
        where TSettings : class, new()
        where TDialog : ScriptConfigDialog<TSettings>, new()
    {
        public abstract IList<Verb> Verbs { get; }

        #region Implementation

        public override void Initialize()
        {
            base.Initialize();

            PlayerControl.KeyDown += PlayerKeyDown;
            LoadVerbs();
        }

        public override void Destroy()
        {
            PlayerControl.KeyDown -= PlayerKeyDown;

            base.Destroy();
        }

        private readonly IDictionary<Keys, Action> m_Actions = new Dictionary<Keys, Action>();

        protected void LoadVerbs()
        {
            foreach (var verb in Verbs)
            {
                var shortcut = DecodeKeyString(verb.ShortcutDisplayStr);
                m_Actions.Remove(shortcut); //Prevent duplicates FIFO.
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

    public abstract class ExtensionUi<TSettings, TDialog> : IExtensionUi
        where TSettings : class, new()
        where TDialog : ScriptConfigDialog<TSettings>, new()
    {
        protected virtual string ConfigFileName { get { return this.GetType().Name; } }

        public abstract ExtensionUiDescriptor Descriptor { get; }

        #region Implementation

        protected Config ScriptConfig { get; private set; }

        protected TSettings Settings
        {
            get { return ScriptConfig == null ? new TSettings() : ScriptConfig.Config; }
        }

        public bool HasConfigDialog()
        {
            return !(typeof(TDialog).IsAssignableFrom(typeof(ScriptConfigDialog<TSettings>)));
        }

        public virtual void Initialize()
        {
            ScriptConfig = new Config(ConfigFileName);
        }

        public virtual void Destroy()
        {
            ScriptConfig.Save();
        }

        public virtual bool ShowConfigDialog(IWin32Window owner)
        {
            using (var dialog = new TDialog())
            {
                dialog.Setup(ScriptConfig.Config);
                if (dialog.ShowDialog(owner) != DialogResult.OK)
                    return false;

                ScriptConfig.Save();
                return true;
            }
        }

        #endregion

        #region ScriptSettings Class

        public class Config : ScriptSettings<TSettings>
        {
            private readonly string m_ConfigName;

            public Config(string configName)
            {
                m_ConfigName = configName;
                Load();
            }

            public Config(TSettings settings)
                : base(settings)
            {
            }

            protected override string ScriptConfigFileName
            {
                get { return string.Format("{0}.config", m_ConfigName); }
            }
        }

        #endregion
    }
}