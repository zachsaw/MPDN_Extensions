using System;
using System.Diagnostics;
using System.IO;
using YAXLib;

namespace Mpdn.RenderScript
{
    namespace Config
    {
        public abstract class ScriptSettings<TSettings> where TSettings : class, new()
        {
            private readonly bool m_InMemory;

            private Exception m_LastException;
            private ConfigProvider<TSettings> m_ScriptSettings;

            protected ScriptSettings()
            {
                m_InMemory = false;
            }

            protected ScriptSettings(TSettings settings)
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
                    const string rootNs = "MediaPlayerDotNet";
                    var localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
                    var extension = Environment.Is64BitProcess ? "64" : "32";
                    var appConfigFolder = Path.Combine(localAppData, rootNs,
                        string.Format("RenderScripts.{0}", extension));
                    return Path.Combine(appConfigFolder, ScriptConfigFileName);
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

        public static class PathEx
        {
            public static string GetDirectoryName(string path)
            {
                if (path == null)
                {
                    throw new ArgumentNullException("path");
                }

                return Path.GetDirectoryName(path) ?? Path.GetPathRoot(path);
            }
        }

        public class ConfigProvider<T> where T : class, new()
        {
            private readonly string m_FileName;
            private T m_Configuration;

            public ConfigProvider(string fileName)
            {
                var path = PathEx.GetDirectoryName(fileName);
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

                    m_Configuration = (T) obj;
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

                var path = PathEx.GetDirectoryName(m_FileName);
                Directory.CreateDirectory(path);

                var serializer = CreateSerializer();

                using (var writer = new StreamWriter(m_FileName))
                {
                    serializer.Serialize(m_Configuration, writer);
                    writer.Flush();
                }
            }

            private static YAXSerializer CreateSerializer()
            {
                return new YAXSerializer(typeof (T), YAXExceptionHandlingPolicies.ThrowErrorsOnly,
                    YAXExceptionTypes.Warning);
            }
        }
    }
}