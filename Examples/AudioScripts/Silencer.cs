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
using DirectShowLib;
using Mpdn.Extensions.Framework;

namespace Mpdn.Examples.AudioScripts
{
    public class Silencer : Extensions.Framework.AudioScript
    {
        private const int CHANNEL_TO_SILENT = 0;

        private short m_Channels;
        private int m_BytesPerSample;

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

        public override void OnGetMediaType(WaveFormatExtensible format)
        {
            m_Channels = format.nChannels;
            m_BytesPerSample = format.wBitsPerSample/8;
        }

        public override bool Process()
        {
            AudioHelpers.CopySample(Audio.Input, Audio.Output); // passthrough from input to output

            IntPtr samples;
            Audio.Output.GetPointer(out samples);
            SilenceSamples(samples, Audio.Output.GetActualDataLength(), CHANNEL_TO_SILENT); // Silence output

            return true; // true = we handled the audio processing
        }

        private unsafe void SilenceSamples(IntPtr samples, int cb, int channel)
        {
            var pb = (byte*) samples;
            var bytesPerSample = m_BytesPerSample;
            var currentChannel = 0;

            while (cb > 0)
            {
                --cb;
                switch (bytesPerSample)
                {
                    case 1:
                        if (currentChannel == channel)
                        {
                            // 8 bit sound uses 0..255 representing -128..127
                            *pb = 128; // 128 means 0
                        }
                        break;
                    case 2:
                        if (currentChannel == channel)
                        {
                            // 16 bit sound uses 16 bits properly (0 means 0)
                            var psi = (short*) pb;
                            *psi = 0;
                        }
                        ++pb; // nudge it on another 8 bits here to get a 16 bit step
                        --cb; // and nudge the count too.
                        break;
                }
                ++pb; // move on 8 bits to next sound sample
                currentChannel = (currentChannel + 1)%m_Channels;
            }
        }
    }
}
