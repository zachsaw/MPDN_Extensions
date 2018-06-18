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
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Mpdn.Extensions.Framework.Config
{
    public class NoSettings
    {
    }

    public interface IScriptConfigDialog<in TSettings> : IDisposable
        where TSettings : class, new()
    {
        void Setup(TSettings settings);
        DialogResult ShowDialog(IWin32Window owner);
    }

    public class ScriptConfigDialog<TSettings> : Form, IScriptConfigDialog<TSettings>
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
            throw new NotImplementedException("Loadsettings undefined (should be overriden).");
        }

        protected virtual void SaveSettings()
        {
            // This needs to be overriden
            throw new NotImplementedException("SaveSettings undefined (should be overriden).");
        }

        // Checks if DialogResult is OK, and if so saves the settings. Remember to set the DialogResult.
        protected override void OnFormClosed(FormClosedEventArgs e)
        {
            base.OnFormClosed(e);

            if (DialogResult != DialogResult.OK)
                return;

            SaveSettings();
        }
    }

    public static class SettingsPrefetcher<TExtensionClass>
    {
        private static ConcurrentDictionary<string, IScriptSettings<object>> m_Settings = new ConcurrentDictionary<string, IScriptSettings<object>>();

        public static IScriptSettings<TSettings> Load<TSettings>(string configFileName)
            where TSettings : class, new()
        {
            return (IScriptSettings<TSettings>) m_Settings.GetOrAdd(configFileName, fname => new PersistentConfig<TExtensionClass, TSettings>(fname));
        }

        public static IScriptSettings<TSettings> Remove<TSettings>(string configFileName)
            where TSettings : class, new()
        {
            IScriptSettings<object> result;
            m_Settings.TryRemove(configFileName, out result);
            return (IScriptSettings<TSettings>)result;
        }

        public static Task Prefetch<TSettings>(string configFileName)
            where TSettings : class, new()
        {
            return Task.Run(() =>
            {
                try
                {
                    Load<TSettings>(configFileName);
                }
                catch { /* Ignore errors */ }
            });
        }
    }

    public abstract class ExtensionUi<TExtensionClass, TSettings, TDialog> : IExtensionUi
        where TSettings : class, new()
        where TDialog : IScriptConfigDialog<TSettings>, new()
    {
        public event EventHandler<EventArgs> SettingsChanged;

        public int Version
        {
            get { return Extension.InterfaceVersion; }
        }

        protected virtual string ConfigFileName
        {
            get { return GetType().Name; }
        }

        public abstract ExtensionUiDescriptor Descriptor { get; }

        public bool SaveToString(out string result)
        {
            return ScriptConfig.SaveToString(out result);
        }

        public bool LoadFromString(string input)
        {
            return ScriptConfig.LoadFromString(input);
        }

        #region Implementation

        private Lazy<IScriptSettings<TSettings>> m_LoadScriptConfig;

        private IScriptSettings<TSettings> ScriptConfig
        {
            get { return m_LoadScriptConfig.Value; }
            set { LoadConfigLazy(() => value); }
        }

        private void LoadConfigLazy(Func<IScriptSettings<TSettings>> load)
        {
            m_LoadScriptConfig = new Lazy<IScriptSettings<TSettings>>(load);
        }

        protected ExtensionUi()
        {
            LoadConfigLazy(() => new MemConfig<TSettings>(new TSettings()));
            SettingsPrefetcher<TExtensionClass>.Prefetch<TSettings>(ConfigFileName);
        }

        public TSettings Settings
        {
            get { return ScriptConfig.Config; }
            set { ScriptConfig = new MemConfig<TSettings>(value); }
        }

        public bool HasConfigDialog()
        {
            return !(typeof (TDialog).IsAssignableFrom(typeof (ScriptConfigDialog<TSettings>)));
        }

        public virtual void Initialize()
        {
            LoadConfigLazy(() => SettingsPrefetcher<TExtensionClass>.Load<TSettings>(ConfigFileName));
        }

        public virtual void Destroy()
        {
            ScriptConfig.Save();
            SettingsPrefetcher<TExtensionClass>.Remove<TSettings>(ConfigFileName);
        }

        public virtual bool ShowConfigDialog(IWin32Window owner)
        {
            using (var dialog = new TDialog())
            {
                dialog.Setup(ScriptConfig.Config);
                if (dialog.ShowDialog(owner) == DialogResult.OK)
                {
                    RaiseSettingsChanged();
                    return true;
                }

                return false;
            }
        }

        protected virtual void RaiseSettingsChanged()
        {
            if (SettingsChanged != null)
                SettingsChanged(this, new EventArgs());
        }

        #endregion
    }

    public static class ExtensionUi
    {
        public static T CreateNew<T>(this T scriptUi)
            where T: IExtensionUi
        {
            var constructor = scriptUi.GetType().GetConstructor(Type.EmptyTypes);
            if (constructor == null)
            {
                throw new EntryPointNotFoundException("ExtensionUi must implement parameter-less constructor");
            }

            return (T) constructor.Invoke(new object[0]);
        }

        public static IExtensionUi Identity = new IdentityExtensionUi();

        private class IdentityExtensionUi : IExtensionUi
        {
            public void Initialize()
            {
            }

            public void Destroy()
            {
            }

            public bool HasConfigDialog()
            {
                return false;
            }

            public bool ShowConfigDialog(IWin32Window owner)
            {
                return false;
            }

            public int Version { get { return 1; } }
            public ExtensionUiDescriptor Descriptor
            {
                get
                {
                    return new ExtensionUiDescriptor
                    {
                        Guid = Guid.Empty,
                        Name = "None",
                        Description = "Do nothing"
                    };
                }
            }
        }
    }
}