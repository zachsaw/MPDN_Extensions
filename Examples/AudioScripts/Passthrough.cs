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
using Mpdn.Extensions.Framework;

namespace Mpdn.Examples.AudioScripts
{
    public class Passthrough : Extensions.Framework.AudioScript
    {
        public override ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("137018B2-E203-4571-B7C1-31FC2C48CD60"),
                    Name = "Passthrough",
                    Description = "Simple audio passthrough example"
                };
            }
        }

        public override bool Process()
        {
            AudioHelpers.CopySample(Audio.Input, Audio.Output); // passthrough from input to output
            return true; // true = we handled the audio processing
        }
    }
}
