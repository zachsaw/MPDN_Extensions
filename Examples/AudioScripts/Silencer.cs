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
    public class Silencer : Extensions.Framework.AudioScript
    {
        private const int CHANNEL_TO_SILENT = 0;

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

        protected override bool SupportBitStreaming
        {
            get { return false; }
        }

        protected override AudioSampleFormat[] SupportedSampleFormats
        {
            get { return new[] {AudioSampleFormat.Pcm8, AudioSampleFormat.Pcm16, AudioSampleFormat.Float, }; }
        }

        protected override void Process(IntPtr samples, int length)
        {
            SilenceSamples(samples, Audio.Output.GetActualDataLength(), CHANNEL_TO_SILENT); // Silence output
        }

        private static unsafe void SilenceSamples(IntPtr samples, int cb, int channel)
        {
            var pb = (byte*) samples;
            var format = Audio.OutputFormat;
            var channels = format.nChannels;
            var sampleFormat = format.SampleFormat();
            var currentChannel = 0;

            while (cb > 0)
            {
                switch (sampleFormat)
                {
                    case AudioSampleFormat.Pcm8:
                        if (currentChannel == channel)
                        {
                            // 8 bit sound uses 0..255 representing -128..127
                            *pb = 128; // 128 means 0
                        }
                        pb++;
                        cb--;
                        break;

                    case AudioSampleFormat.Pcm16:
                        if (currentChannel == channel)
                        {
                            // 16 bit sound uses 16 bits properly (0 means 0)
                            var psi = (short*) pb;
                            *psi = 0;
                        }

                        pb += 2;
                        cb -= 2;
                        break;

                    case AudioSampleFormat.Float:
                        if (currentChannel == channel)
                        {
                            var psi = (float*) pb;
                            *psi = 0;
                        }

                        pb += 4;
                        cb -= 4;
                        break;
                }
                currentChannel = (currentChannel + 1)%channels;
            }
        }
    }
}
