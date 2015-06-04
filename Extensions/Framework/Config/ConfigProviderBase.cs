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

//css_reference System.Xml.Linq
using System;
using YAXLib;

namespace Mpdn.Extensions.Framework.Config
{
    public abstract class ConfigProviderBase<T> : IConfigProvider<T>
        where T : class, new()
    {
        public T Configuration { get; set; }
        public abstract bool Load(out Exception loadException);
        public abstract bool Save(out Exception saveException);

        public virtual bool LoadFromString(string input, out Exception loadException)
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

        public virtual bool SaveToString(out string output, out Exception saveException)
        {
            output = string.Empty;
            saveException = null;
            try
            {
                var serializer = CreateSerializer();
                output = serializer.Serialize(Configuration);
                return true;
            }
            catch (Exception ex)
            {
                saveException = ex;
            }
            return false;
        }

        protected static YAXSerializer CreateSerializer()
        {
            return new YAXSerializer(typeof (T), YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Ignore,
                YAXSerializationOptions.DontSerializeCyclingReferences |
                YAXSerializationOptions.DontSerializePropertiesWithNoSetter |
                YAXSerializationOptions.SerializeNullObjects);
        }
    }
}