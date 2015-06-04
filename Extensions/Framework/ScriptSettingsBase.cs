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

namespace Mpdn.Extensions.Framework
{
    public abstract class ScriptSettingsBase<TSettings> : IScriptSettings<TSettings> where TSettings : class, new()
    {
        private Exception m_LastException;
        private IConfigProvider<TSettings> m_ScriptSettings;

        protected IConfigProvider<TSettings> Settings
        {
            get { return m_ScriptSettings ?? (m_ScriptSettings = CreateConfigProvider()); }
        }

        protected abstract IConfigProvider<TSettings> CreateConfigProvider();

        public TSettings Config
        {
            get
            {
                Debug.Assert(Settings.Configuration != null);
                return Settings.Configuration;
            }
        }

        public Exception LastException
        {
            get { return m_LastException; }
        }

        public virtual bool Load()
        {
            return Settings.Load(out m_LastException);
        }

        public virtual bool Save()
        {
            return Settings.Save(out m_LastException);
        }

        public virtual bool LoadFromString(string input)
        {
            return Settings.Load(out m_LastException);
        }

        public virtual bool SaveToString(out string output)
        {
            return Settings.SaveToString(out output, out m_LastException);
        }
    }
}
