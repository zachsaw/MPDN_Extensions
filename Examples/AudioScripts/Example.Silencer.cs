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
using Mpdn.Extensions.Framework.AudioChain;

namespace Mpdn.Examples.AudioScripts
{
    namespace Example
    {
        public class Silencer : AudioFilter
        {
            private const int CHANNEL_TO_SILENT = 0;

            protected override bool CpuOnly
            {
                get { return true; }
            }

            protected override void Process(float[,] samples, short channels, int sampleCount)
            {
                for (int i = 0; i < sampleCount; i++)
                {
                    samples[CHANNEL_TO_SILENT, i] = 0;
                }
            }
        }

        public class SilencerUi : AudioChainUi<StaticAudioChain<Silencer>>
        {
            public override ExtensionUiDescriptor Descriptor
            {
                get
                {
                    return new ExtensionUiDescriptor
                    {
                        Guid = new Guid("93C32135-7F75-44EE-91C9-6BDF905D76BB"),
                        Name = "Silencer",
                        Description = "Simple audio silencer example"
                    };
                }
            }

            public override string Category
            {
                get { return "Volume"; }
            }
        }
    }
}