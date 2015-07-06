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
using System.Runtime.Serialization;

namespace Mpdn.Extensions.Framework.Exceptions
{
    public class MpdnScriptEngineException : Exception
    {
        public MpdnScriptEngineException()
        {
        }

        public MpdnScriptEngineException(string message)
            : base(message)
        {
        }

        public MpdnScriptEngineException(string message, Exception innerException)
            : base(message, innerException)
        {
        }

        protected MpdnScriptEngineException(SerializationInfo info, StreamingContext context)
            : base(info, context)
        {
        }
    }
}
