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
using System.IO;
using YAXLib;

namespace Mpdn.Extensions.Framework
{
    public class PersistentConfigProvider<T> : ConfigProviderBase<T>
        where T : class, new()
    {
        private readonly string m_FileName;

        public PersistentConfigProvider(string fileName)
        {
            var path = PathHelper.GetDirectoryName(fileName);
            if (string.IsNullOrWhiteSpace(path) || string.IsNullOrWhiteSpace(fileName))
            {
                throw new ArgumentException("fileName is invalid", fileName);
            }

            m_FileName = fileName;
        }

        public override bool Load(out Exception loadException)
        {
            loadException = null;
            if (string.IsNullOrWhiteSpace(m_FileName) || !File.Exists(m_FileName))
            {
                // No config file yet
                Configuration = new T();
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

                Configuration = (T) obj;
                return true;
            }
            catch (Exception ex)
            {
                loadException = ex;
                Configuration = new T();
            }
            return false;
        }

        public override bool Save(out Exception saveException)
        {
            saveException = null;
            try
            {
                if (Configuration == null)
                {
                    Configuration = new T();
                }

                var path = PathHelper.GetDirectoryName(m_FileName);
                Directory.CreateDirectory(path);

                var serializer = CreateSerializer();
                using (var writer = new StreamWriter(m_FileName))
                {
                    serializer.Serialize(Configuration, writer);
                    writer.Flush();
                }
                return true;
            }
            catch (Exception ex)
            {
                saveException = ex;
            }
            return false;
        }
    }

    public class MemConfigProvider<T> : ConfigProviderBase<T>
        where T : class, new()
    {
        public override bool Load(out Exception loadException)
        {
            loadException = null;
            Configuration = new T();
            return true;
        }

        public override bool Save(out Exception saveException)
        {
            saveException = null;
            return true;
        }
    }
}
