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
        public class Gargle : AudioFilter
        {
            private const int GARGLE_RATE = 5;
            private const int SHAPE = 0; // 0=Triangle, 1=Sqaure

            private int m_Phase;

            protected override bool CpuOnly
            {
                get { return true; }
            }

            protected override void Process(float[,] samples, short channels, int sampleCount)
            {
                // Note: This runs on CPU only (in .NET) but it can just as easily be ported to run on OpenCL
                GargleSamples(samples, SHAPE == 0);
            }

            private void GargleSamples(float[,] samples, bool triangle)
            {
                int period = Input.Format.nSamplesPerSec/GARGLE_RATE;

                var channels = samples.GetLength(0);
                var length = samples.GetLength(1);
                for (int i = 0; i < length; i++)
                {
                    // m_Phase is the number of samples from the start of the period.
                    // We keep this running from one call to the next,
                    // but if the period changes so as to make this more
                    // than Period then we reset to 0 with a bang.  This may cause
                    // an audible click or pop (but, hey! it's only a sample!)
                    //
                    m_Phase++;

                    if (m_Phase > period)
                        m_Phase = 0;

                    var m = m_Phase; // m is what we modulate with
                    for (int c = 0; c < channels; c++)
                    {
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
                        var v = samples[c, i];
                        // Note: No clipping required - the framework clips it to [-1.0f..1.0f] for us
                        samples[c, i] = (v*m*2)/period;
                    }
                }
            }
        }

        public class GargleUi : AudioChainUi<StaticAudioChain<Gargle>>
        {
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

            public override string Category
            {
                get { return "Effect"; }
            }
        }
    }
}