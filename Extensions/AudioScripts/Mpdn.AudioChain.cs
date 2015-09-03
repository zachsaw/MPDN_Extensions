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

namespace Mpdn.Extensions.AudioScripts
{
    public class AudioChainSettings
    {
        public AudioChainSettings()
        {
        }
    }

    public class AudioChain : AudioScript<AudioChainSettings, AudioChainConfigDialog>
    {
        public override ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("92E3FD70-C663-4C94-9B17-2EC10D9EA8CC"),
                    Name = "AudioChain",
                    Description = "Audio script chaining"
                };
            }
        }

        protected override void Process(float[,] samples, short channels, int sampleCount)
        {
            throw new NotImplementedException();
        }
    }
}
