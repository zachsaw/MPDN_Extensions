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
using System.IO;
using YAXLib;

namespace Mpdn.Extensions.Framework
{
    public abstract class ScriptSettingsBase<TFolderName, TSettings> where TSettings : class, new()
    {
        private readonly bool m_InMemory;

        private Exception m_LastException;
        private ConfigProvider<TSettings> m_ScriptSettings;

        protected ScriptSettingsBase()
        {
            m_InMemory = false;
        }

        protected ScriptSettingsBase(TSettings settings)
        {
            m_InMemory = true;
            Settings.Configuration = settings ?? new TSettings();
        }

        private ConfigProvider<TSettings> Settings
        {
            get { return m_ScriptSettings ?? (m_ScriptSettings = new ConfigProvider<TSettings>(ConfigFilePath)); }
        }

        protected abstract string ScriptConfigFileName { get; }

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

        public string ConfigFilePath
        {
            get
            {
                return AppPath.GetUserDataFilePath(ScriptConfigFileName, typeof (TFolderName).Name);
            }
        }

        public bool Load()
        {
            if (m_InMemory)
            {
                Settings.Configuration = new TSettings();
                return true;
            }

            if (Settings.Load(out m_LastException))
                return true;

            Settings.Configuration = new TSettings();
            return false;
        }

        public bool Save()
        {
            if (m_InMemory)
                return true;

            try
            {
                Settings.Save();
            }
            catch (Exception ex)
            {
                m_LastException = ex;
                return false;
            }

            return true;
        }

        public bool SaveToString(out string settings)
        {
            settings = string.Empty;
            return true;
        }

        public void Destroy()
        {
            if (m_InMemory)
                return;

            try
            {
                File.Delete(ConfigFilePath);
            }
            catch
            {
            }
        }
    }

    public class ConfigProvider<T> where T : class, new()
    {
        private readonly string m_FileName;
        private T m_Configuration;

        public ConfigProvider(string fileName)
        {
            var path = PathHelper.GetDirectoryName(fileName);
            if (string.IsNullOrWhiteSpace(path) || string.IsNullOrWhiteSpace(fileName))
            {
                throw new ArgumentException("fileName is invalid", fileName);
            }

            m_FileName = fileName;
        }

        public T Configuration
        {
            get { return m_Configuration; }
            set { m_Configuration = value; }
        }

        public bool Load(out Exception loadException)
        {
            loadException = null;
            if (string.IsNullOrWhiteSpace(m_FileName) || !File.Exists(m_FileName))
            {
                // No config file yet
                m_Configuration = new T();
                return true;
            }

            var serializer = CreateSerializer();

            try
            {
                object obj;
                using (var reader = new StreamReader(m_FileName))
                {
                    obj = serializer.Deserialize(reader);
                }

                if (obj == null)
                {
                    throw new YAXException("Failed to deserialize from config file");
                }

                m_Configuration = (T)obj;
                return true;
            }
            catch (Exception ex)
            {
                loadException = ex;
                m_Configuration = new T();
            }
            return false;
        }

        public bool LoadFromString(string input, out Exception loadException)
        {
            loadException = null;

            var serializer = CreateSerializer();

            try
            {
                var obj = serializer.Deserialize(input);
                if (obj == null)
                {
                    throw new YAXException("Failed to deserialize from string");
                }

                m_Configuration = (T)obj;
                return true;
            }
            catch (Exception ex)
            {
                loadException = ex;
                m_Configuration = new T();
            }
            return false;
        }

        public void Save()
        {
            if (m_Configuration == null)
            {
                m_Configuration = new T();
            }

            var path = PathHelper.GetDirectoryName(m_FileName);
            Directory.CreateDirectory(path);

            var serializer = CreateSerializer();
            using (var writer = new StreamWriter(m_FileName))
            {
                serializer.Serialize(m_Configuration, writer);
                writer.Flush();
            }
        }

        public string SaveToString()
        {
            var serializer = CreateSerializer();
            return serializer.Serialize(m_Configuration);
        }

        private static YAXSerializer CreateSerializer()
        {
            return new YAXSerializer(typeof (T), YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Ignore,
                YAXSerializationOptions.DontSerializeCyclingReferences |
                YAXSerializationOptions.DontSerializePropertiesWithNoSetter |
                YAXSerializationOptions.SerializeNullObjects);
        }
    }
}
