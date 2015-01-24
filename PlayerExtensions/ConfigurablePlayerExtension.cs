﻿using System;
using System.Windows.Forms;
﻿using Mpdn.PlayerExtensions.Config;

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

    public abstract class ConfigurablePlayerExtension<TSettings, TDialog> : PlayerExtension
        where TSettings : class, new()
        where TDialog : ScriptConfigDialog<TSettings>, new()
    {
        protected Config ScriptConfig { get; private set; }

        protected abstract string ConfigFileName { get; }

        protected TSettings Settings
        {
            get { return ScriptConfig == null ? new TSettings() : ScriptConfig.Config; }
        }

        #region Implementation

        public override ExtensionDescriptor Descriptor
        {
            get
            {
                return new ExtensionDescriptor
                {
                    HasConfigDialog = true,
                    Copyright = ScriptDescriptor.Copyright,
                    Description = ScriptDescriptor.Description,
                    Guid = ScriptDescriptor.Guid,
                    Name = ScriptDescriptor.Name
                };
            }
        }

        public override void Initialize()
        {
            base.Initialize();

            ScriptConfig = new Config(ConfigFileName);
        }

        public override bool ShowConfigDialog(IWin32Window owner)
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