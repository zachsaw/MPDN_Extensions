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

using System;
using Mpdn.Extensions.Framework.AudioChain;

namespace Mpdn.Extensions.AudioScripts.Shiandow
{
    public class Crossfeed : CpuAudioFilter
    {
        protected override void Process(float[,] samples, short channels, int sampleCount)
        {
            if (channels != 2)
                return;

            Apply(samples, Output.Format.nSamplesPerSec);
        }
        
        const int crossfeedOrder = 16;
        float[,] crossfeed = new float[2, crossfeedOrder];

        const int echoOrder = 12;
        float[,] echo = new float[2, echoOrder];

        public void Apply(float[,] samples, int frequency)
        {
            int length = samples.GetLength(1);

            float crossfeedDelay = 450f;
            float echoDelay = 1200f;// crossfeedDelay * (float) (Math.PI / Math.Sqrt(2.0));

            float crossfeedAttenuation = -3.5f;
            float echoAttenuation = -8.0f;

            float volume = (float) Math.Pow(10, crossfeedAttenuation / 10);
            float a = (1000000 * (crossfeedOrder - 1)) / (crossfeedDelay * frequency);
            float strength = (2 - a) / (2 + a);

            for (int i = 0; i < length; i++)
            {
                float L = samples[0, i];
                float R = samples[1, i];

                float prevL = crossfeed[0, 0];
                float prevR = crossfeed[1, 0];

                crossfeed[0, 0] = L;
                crossfeed[1, 0] = R;

                // Bilinear transform of Laplace transform ((a + s)^-k) of gamma distribution (x^(k-1) e^(-ax))
                for (int j = 1; j < crossfeedOrder; j++)
                {
                    L = (1 - strength) * (L + prevL) / 2.0f + strength * crossfeed[0, j];
                    R = (1 - strength) * (R + prevR) / 2.0f + strength * crossfeed[1, j];

                    prevL = crossfeed[0, j];
                    prevR = crossfeed[1, j];

                    crossfeed[0, j] = L;
                    crossfeed[1, j] = R;
                }

                samples[0, i] = (samples[0, i] + volume * R) / (1 + volume);
                samples[1, i] = (samples[1, i] + volume * L) / (1 + volume);
            }

            /*
            //float spread = ((crossfeedDelay / crossfeedOrder) * frequency) / 1000000;
            //float strength = (float) Math.Exp(-1 / spread);
            float strength = crossfeedDelay * frequency / (crossfeedOrder * 1000000 + crossfeedDelay * frequency); // Correct delay (at 0 hz)

            for (int i = 0; i < length; i++)
            {
                float L = samples[0, i];
                float R = samples[1, i];

                // Zero-Pole mapping ((1 - e^-a z^-1)^-k) of Laplace transform ((a + s)^-k) of gamma distribution (x^(k-1) e^(-ax))
                for (int j = 0; j < crossfeedOrder; j++)
                {
                    L = (1 - strength) * L + strength * crossfeed[0, j];
                    R = (1 - strength) * R + strength * crossfeed[1, j];

                    crossfeed[0, j] = L;
                    crossfeed[1, j] = R;
                }

                samples[0, i] = (samples[0, i] + volume * R) / (1 + volume);
                samples[1, i] = (samples[1, i] + volume * L) / (1 + volume);
            }*/

            volume = (float) Math.Pow(10, echoAttenuation / 10);
            a = (1000000 * (echoOrder - 1)) / (echoDelay * frequency);
            strength = (2 - a) / (2 + a);

            for (int i = 0; i < length; i++)
            {
                /*float L = (samples[0, i] + volume * echo[1, echoOrder - 1]);// / (1 + volume);
                float R = (samples[1, i] + volume * echo[0, echoOrder - 1]);// / (1 + volume);

                float prevL = echo[0, 0];
                float prevR = echo[1, 0];

                echo[0, 0] = L;
                echo[1, 0] = R;

                samples[0, i] = L * (1 - volume);
                samples[1, i] = R * (1 - volume);*/

                float L = samples[0, i];
                float R = samples[1, i];

                float prevL = echo[0, 0];
                float prevR = echo[1, 0];

                echo[0, 0] = L;
                echo[1, 0] = R;

                // Bilinear transform of Laplace transform ((a + s)^-k) of gamma distribution (x^(k-1) e^(-ax))
                for (int j = 1; j < echoOrder; j++)
                {
                    L = (1 - strength) * (L + prevL) / 2.0f + strength * echo[0, j];
                    R = (1 - strength) * (R + prevR) / 2.0f + strength * echo[1, j];

                    prevL = echo[0, j];
                    prevR = echo[1, j];

                    echo[0, j] = L;
                    echo[1, j] = R;
                }

                samples[0, i] = (samples[0, i] + volume * R) / (1 + volume);
                samples[1, i] = (samples[1, i] + volume * L) / (1 + volume);
            }

            /*
            float spread = ((echoDelay / echoOrder) * frequency) / 1000000;
            strength = (float)Math.Exp(-1 / spread);
            //strength = echoDelay * frequency / (echoOrder * 1000000 + echoDelay * frequency); // Correct delay (at 0 hz)

            for (int i = 0; i < length; i++)
            {
                float L = (samples[0, i] + volume * echo[1, echoOrder - 1]);// / (1 + volume);
                float R = (samples[1, i] + volume * echo[0, echoOrder - 1]);// / (1 + volume);

                samples[0, i] = L * (1 - volume);
                samples[1, i] = R * (1 - volume);

                // Zero-Pole mapping ((1 - e^-a z^-1)^-k) of Laplace transform ((a + s)^-k) of gamma distribution (x^(k-1) e^(-ax))
                for (int j = 0; j < echoOrder; j++)
                {
                    L = (1 - strength) * L + strength * echo[0, j];
                    R = (1 - strength) * R + strength * echo[1, j];

                    echo[0, j] = L;
                    echo[1, j] = R;
                }
            }*/
        }
    }

    public class CrossfeedUi : AudioChainUi<Crossfeed>
    {
        public override string Category { get { return "Crossfeed"; } }

        public override ExtensionUiDescriptor Descriptor {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("C820B4A5-6721-4B08-BED5-BB9E1BBDE816"),
                    Name = "Crossfeed",
                    Description = "Adds crossfeed to audio.",
                    Copyright = "By Shiandow (2017)"
                };
            }
        }
    }
}
