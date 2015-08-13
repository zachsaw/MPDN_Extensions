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
    public class Gargle : Extensions.Framework.AudioScript
    {
        private const int GARGLE_RATE = 10;
        private const int SHAPE = 0; // 0=Triangle, 1=Sqaure

        private int m_Phase;

        public override ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("A27971B2-F625-4AC8-9AC5-5B448AB77BB6"),
                    Name = "Gargle",
                    Description = "Simple audio gargle example (translated from Windows SDK)"
                };
            }
        }

        protected override void Process(float[,] samples)
        {
//            GargleSamples(samples, Audio.Output.GetActualDataLength(), SHAPE == 0); // gargle output
        }

        private unsafe void GargleSamples(IntPtr samples, int cb, bool triangle)
        {
            var pb = (byte*) samples;
            var format = Audio.OutputFormat;
            var channels = format.nChannels;
            var samplesPerSec = format.nSamplesPerSec;
            var sampleFormat = format.SampleFormat();

            // We know how many samples per sec and how
            // many channels so we can calculate the modulation period in samples.
            //
            int period = (samplesPerSec*channels)/GARGLE_RATE;

            while (cb > 0)
            {
                // If m_Shape is 0 (triangle) then we multiply by a triangular waveform
                // that runs 0..Period/2..0..Period/2..0... else by a square one that
                // is either 0 or Period/2 (same maximum as the triangle) or zero.
                //
                {
                    // m_Phase is the number of samples from the start of the period.
                    // We keep this running from one call to the next,
                    // but if the period changes so as to make this more
                    // than Period then we reset to 0 with a bang.  This may cause
                    // an audible click or pop (but, hey! it's only a sample!)
                    //
                    ++m_Phase;

                    if (m_Phase > period)
                        m_Phase = 0;

                    int m = m_Phase; // m is what we modulate with

                    if (triangle)
                    {
                        // Triangle
                        if (m > period/2) m = period - m; // handle downslope
                    }
                    else
                    {
                        // Square wave
                        if (m <= period/2) m = period/2;
                        else m = 0;
                    }

                    switch (sampleFormat)
                    {
                        case AudioSampleFormat.Pcm8:
                            // 8 bit sound uses 0..255 representing -128..127
                            // Any overflow, even by 1, would sound very bad.
                            // so we clip paranoically after modulating.
                            // I think it should never clip by more than 1
                            //
                            {
                                int i = *pb - 128; // sound sample, zero based

                                i = (i*m*2)/period; // modulate
                                if (i > 127) i = 127; // clip
                                if (i < -128) i = -128;

                                *pb = (byte) (i + 128); // reset zero offset to 128
                            }
                            pb++;
                            cb--;
                            break;

                        case AudioSampleFormat.Pcm16:
                            // 16 bit sound uses 16 bits properly (0 means 0)
                            // We still clip paranoically
                            //
                            {
                                var psi = (short*) pb;
                                int i = *psi; // in a register, we might hope

                                i = (i*m*2)/period; // modulate
                                if (i > 32767) i = 32767; // clip
                                if (i < -32768) i = -32768;

                                *psi = (short) i;
                            }
                            pb += 2;
                            cb -= 2;
                            break;

                        case AudioSampleFormat.Float:
                            {
                                var psi = (float*) pb;
                                float i = *psi;

                                i = (i*m*2)/period; // modulate
                                if (i > 1.0f) i = 1.0f; // clip
                                if (i < -1.0f) i = -1.0f;

                                *psi = i;
                            }
                            pb += 4;
                            cb -= 4;
                            break;
                    }
                }
            }
        }
    }
}
