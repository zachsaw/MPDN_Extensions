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
    public class Crossfeed : CpuAudioProcess
    {
        protected override void Process(IAudioDescription input, float[,] samples, short channels, int sampleCount)
        {
            if (channels != 2)
                return;

            Apply(samples, input.Format.nSamplesPerSec);
        }

        #region Effects

        private interface IEffect
        {
            void Render(float[] input, float[] output = null);
        }

        private class BilinearGammaEffect : IEffect
        {
            private readonly int m_Order;
            private readonly float m_strength;
            private readonly float m_volume;

            private readonly bool m_Cross;
            private readonly bool m_PreserveMid;

            private float m_prevL = 0.0f;
            private float m_prevR = 0.0f;
            private float[,] m_cache;

            public BilinearGammaEffect(int frequency, float delay, float volume, int order, bool cross = true, bool preserveMid = false)
            {
                m_Cross = cross;
                m_Order = order;
                m_volume = volume;
                m_PreserveMid = preserveMid;

                float a = m_Order / (delay * frequency);
                m_strength = (2 - a) / (2 + a);

                m_cache = new float[2, m_Order];
            }

            public void Render(float[] input, float[] output = null)
            {
                output = output ?? input;
                float L = input[0];
                float R = input[1];

                for (int j = 0; j < m_Order; j++)
                {
                    L = (1 - m_strength) * (L + m_prevL) / 2.0f + m_strength * m_cache[0, j];
                    R = (1 - m_strength) * (R + m_prevR) / 2.0f + m_strength * m_cache[1, j];

                    m_prevL = m_cache[0, j];
                    m_prevR = m_cache[1, j];

                    m_cache[0, j] = L;
                    m_cache[1, j] = R;
                }

                m_prevL = input[0];
                m_prevR = input[1];

                if (m_PreserveMid)
                {
                    output[0] = (output[0] + m_volume * (m_Cross ? R - L : L - R)/2.0f) / (1 + m_volume);
                    output[1] = (output[1] + m_volume * (m_Cross ? L - R : R - L)/2.0f) / (1 + m_volume);
                }
                else
                {
                    output[0] = (output[0] + m_volume * (m_Cross ? R : L)) / (1 + m_volume);
                    output[1] = (output[1] + m_volume * (m_Cross ? L : R)) / (1 + m_volume);
                }
            }
        }

        private class ZeroPoleGammaEffect : IEffect
        {
            private int m_Order;
            private float m_strength;
            private float m_volume;

            private float[,] m_cache;

            private bool m_Echo;

            public ZeroPoleGammaEffect(int frequency, float delay, float volume, int order, bool echo = false)
            {
                m_Echo = echo;
                m_Order = order;
                m_volume = volume;

                float a = m_Order / (delay * frequency);

                float spread = ((delay / m_Order) * frequency) / 1000000;
                m_strength = (float)Math.Exp(-1 / spread);

                m_cache = new float[2, m_Order];
            }
        
            public void Render(float[] input, float[] output = null)
            {
                float L, R;
                output = output ?? input;

                if (m_Echo)
                {
                    L = (1 - m_volume) * input[0] + m_volume * m_cache[1, m_Order - 1];
                    R = (1 - m_volume) * input[1] + m_volume * m_cache[0, m_Order - 1];

                    output[0] = (1 - m_volume) * output[0] + m_volume * m_cache[1, m_Order - 1];
                    output[1] = (1 - m_volume) * output[1] + m_volume * m_cache[0, m_Order - 1];
                }
                else
                {
                    L = input[0];
                    R = input[1];
                }

                // Zero-Pole mapping ((1 - e^-a z^-1)^-k) of Laplace transform ((a + s)^-k) of gamma distribution (x^(k-1) e^(-ax))
                for (int j = 0; j < m_Order; j++)
                {
                    L = (1 - m_strength) * L + m_strength * m_cache[0, j];
                    R = (1 - m_strength) * R + m_strength * m_cache[1, j];

                    m_cache[0, j] = L;
                    m_cache[1, j] = R;
                }

                if (!m_Echo)
                {
                    output[0] = (output[0] + m_volume * R) / (1 + m_volume);
                    output[1] = (output[1] + m_volume * L) / (1 + m_volume);
                }
            }
        }

        #endregion

        private IEffect crossfeed;
        private IEffect acoustics;

        public void Apply(float[,] samples, int frequency)
        {
            if (crossfeed == null || acoustics == null)
            {
                double angle = 45 * (Math.PI / 180); // w.r.t. bisector
                double head = 0.175;
                double v = 340.0;

                float crossfeedDelay = (float)((Math.Sin(angle) + angle) * (head / 2) / v);
                float acousticsDelay =  (float)(Math.PI * (head / 2) / v);

                int crossfeedOrder = 16;
                int acousticsOrder = 4;

                //float crossfeedVolume = (float)(1.0 * (1 + Math.Cos(angle)) / 2);
                float xfeed = 0.25f;
                float crossfeedVolume = (float)(((1 + xfeed) - (1 - xfeed) * Math.Cos(angle)) / ((1 + xfeed) + (1 - xfeed) * Math.Cos(angle)));
                float acousticsVolume = 0.01f;

                //crossfeedVolume /= (1 + acousticsVolume);
                //acousticsVolume /= (1 + crossfeedVolume);

                crossfeed = new BilinearGammaEffect(frequency, crossfeedDelay, crossfeedVolume, crossfeedOrder);
                acoustics = new BilinearGammaEffect(frequency, acousticsDelay, acousticsVolume, acousticsOrder);
            }

            int length = samples.GetLength(1);
            for (int i = 0; i < length; i++)
            {
                float[] input = new float[2];
                float[] output = new float[2];
                input[0] = samples[0, i];
                input[1] = samples[1, i];
                output[0] = samples[0, i];
                output[1] = samples[1, i];

                crossfeed.Render(input, output);
                //acoustics.Render(output, output);

                samples[0, i] = output[0];
                samples[1, i] = output[1];
            }
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
