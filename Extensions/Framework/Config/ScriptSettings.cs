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

namespace Mpdn.Extensions.Framework.Config
{
    public sealed class PersistentConfig<TExtensionClass, TSettings> : ScriptSettingsBase<TSettings>
        where TSettings : class, new()
    {
        private readonly string m_ConfigName;

        public PersistentConfig(string configName)
        {
            m_ConfigName = configName;
            Load();
        }

        public string ConfigFilePath
        {
            get { return AppPath.GetUserDataFilePath(ScriptConfigFileName, typeof (TExtensionClass).Name); }
        }

        private string ScriptConfigFileName
        {
            get { return string.Format("{0}.config", m_ConfigName); }
        }

        protected override IConfigProvider<TSettings> CreateConfigProvider()
        {
            return new PersistentConfigProvider<TSettings>(ConfigFilePath);
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
