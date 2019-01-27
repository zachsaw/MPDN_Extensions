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
using System.Diagnostics;
using System.Windows.Forms;

namespace Mpdn.Extensions.Framework.Config
{
    public sealed class PersistentConfig<TExtensionClass, TSettings> : ScriptSettingsBase<TSettings>
        where TSettings : class, new()
    {
        private readonly string m_ConfigName;

        public PersistentConfig(string configName)
        {
            m_ConfigName = configName;
            if (Load())
                return;

#if DEBUG
            Trace.WriteLine(string.Format("Load Settings Failed!\r\n\r\n{0}", LastException));
            try
            {
                GuiThread.DoAsync(() =>
                MessageBox.Show(Gui.VideoBox,
                    "WARNING: Script settings have failed to load. This will cause user's config to be deleted (or become corrupted)",
                    "Load Settings Failed", MessageBoxButtons.OK, MessageBoxIcon.Warning));
            } catch { /* Ignore errors */ }
#endif
        }

        protected override IConfigProvider<TSettings> CreateConfigProvider()
        {
            return new PersistentConfigProvider<TSettings>(ConfigFilePath);
        }

        private string ConfigFilePath
        {
            get
            {
                string classFolder = typeof (TExtensionClass).Name;
                if (classFolder.StartsWith("I")) classFolder = classFolder.Substring(1);
                if (!classFolder.EndsWith("s")) classFolder += "s";

                return AppPath.GetUserDataFilePath(ScriptConfigFileName, classFolder);
            }
        }

        private string ScriptConfigFileName
        {
            get { return string.Format("{0}.config", m_ConfigName); }
        }
    }

    public sealed class MemConfig<TSettings> : ScriptSettingsBase<TSettings>
        where TSettings : class, new()
    {
        public MemConfig()
        {
            Settings.Configuration = new TSettings();
        }

        public MemConfig(TSettings settings)
        {
            if (settings == null)
            {
                throw new ArgumentNullException("settings");
            }
            Settings.Configuration = settings;
        }

        protected override IConfigProvider<TSettings> CreateConfigProvider()
        {
            return new MemConfigProvider<TSettings>();
        }

        public override bool Load()
        {
            return true;
        }

        public override bool Save()
        {
            return true;
        }
    }
}
