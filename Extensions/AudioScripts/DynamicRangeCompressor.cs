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

// Ported from NAudio SimpleCompressor for MPDN
// https://github.com/naudio/NAudio/blob/master/NAudio/Dsp/SimpleCompressor.cs
// NAudio SimpleCompress is based on SimpleComp v1.10 © 2006, ChunkWare Music Software, OPEN-SOURCE

using System;
using System.Diagnostics;
using Mpdn.Extensions.Framework;

namespace Mpdn.Extensions.AudioScripts
{
    public class DynamicRangeCompressorSettings
    {
        public float ThresholdDb { get; set; }
        public float Ratio { get; set; }
        public float MakeupGainDb { get; set; }
        public int AttackMs { get; set; }
        public int ReleaseMs { get; set; }

        public DynamicRangeCompressorSettings()
        {
            ThresholdDb = -15;
            Ratio = 3;
            MakeupGainDb = 3;
            AttackMs = 200; // 0.2s
            ReleaseMs = 1000; // 1s
        }
    }

    public class DynamicRangeCompressor : AudioScript<DynamicRangeCompressorSettings, DynamicRangeCompressorConfigDialog>
    {
        // DC offset to prevent denormal
        protected const float DC_OFFSET = 1.0e-25f;

        private float m_EnvdB = DC_OFFSET;
        private EnvelopeDetector m_Attack;
        private EnvelopeDetector m_Release;
        private int m_SampleRate;
        private int m_AttackMs;
        private int m_ReleaseMs;

        public override ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("C29B883B-AF2B-45F7-9FDA-1F8A67004E2E"),
                    Name = "DynamicRangeCompressor",
                    Description = "Performs simple DRC",
                    Copyright = "Adapted from NAudio SimpleCompressor by Zachs"
                };
            }
        }

        protected override bool CpuOnly
        {
            get { return true; }
        }

        public override void OnNewSegment(long startTime, long endTime, double rate)
        {
            base.OnNewSegment(startTime, endTime, rate);

            m_EnvdB = DC_OFFSET;
        }

        protected override void Process(float[,] samples, short channels, int sampleCount)
        {
            var sampleRate = Audio.InputFormat.nSamplesPerSec;
            if (m_SampleRate != sampleRate || m_AttackMs != Settings.AttackMs || m_ReleaseMs != Settings.ReleaseMs)
            {
                m_Attack = new EnvelopeDetector(Settings.AttackMs, sampleRate);
                m_Release = new EnvelopeDetector(Settings.ReleaseMs, sampleRate);
                m_SampleRate = sampleRate;
                m_AttackMs = Settings.AttackMs;
                m_ReleaseMs = Settings.ReleaseMs;
                m_EnvdB = DC_OFFSET;
            }

            Compress(samples, Settings.ThresholdDb, Settings.Ratio, Settings.MakeupGainDb);
        }

        private void Compress(float[,] samples, float thresholdDb, float ratio, float makeupGainDb)
        {
            var channels = samples.GetLength(0);
            var length = samples.GetLength(1);
            for (int i = 0; i < length; i++)
            {
                var allChMax = float.MinValue;
                for (int c = 0; c < channels; c++)
                {
                    var s = Math.Abs(samples[c, i]);
                    allChMax = Math.Max(s, allChMax);
                }

                allChMax += DC_OFFSET;
                var allChMaxdB = Decibels.FromLinear(allChMax);

                var overdB = allChMaxdB - thresholdDb;
                if (overdB < 0)
                    overdB = 0;

                // attack/release
                overdB += DC_OFFSET;
                Run(overdB, ref m_EnvdB);
                overdB = m_EnvdB - DC_OFFSET;

                var gr = -overdB*(ratio - 1.0f);
                gr = Decibels.ToLinear(gr)*Decibels.ToLinear(makeupGainDb);

                for (int c = 0; c < channels; c++)
                {
                    samples[c, i] *= gr;
                }
            }
        }

        private void Run(float inValue, ref float state)
        {
            // assumes that:
            // positive delta = attack
            // negative delta = release
            // good for linear & log values
            if (inValue > state)
            {
                m_Attack.Run(inValue, ref state); // attack
            }
            else
            {
                m_Release.Run(inValue, ref state); // release
            }
        }
    }

    class EnvelopeDetector
    {
        private float m_SampleRate;
        private float m_TimeConstantMs;
        private float m_Coef;

        public EnvelopeDetector() : this(1.0f, 44100.0f)
        {
        }

        public EnvelopeDetector(float timeConstantMs, float sampleRate)
        {
            Debug.Assert(sampleRate > 0.0);
            Debug.Assert(timeConstantMs > 0.0);
            m_SampleRate = sampleRate;
            m_TimeConstantMs = timeConstantMs;
            SetCoef();
        }

        public float TimeConstant
        {
            get
            {
                return m_TimeConstantMs;
            }
            set
            {
                Debug.Assert(value > 0.0);
                m_TimeConstantMs = value;
                SetCoef();
            }
        }

        public float SampleRate
        {
            get
            {
                return m_SampleRate;
            }
            set
            {
                Debug.Assert(value > 0.0);
                m_SampleRate = value;
                SetCoef();
            }
        }

        public void Run(float inValue, ref float state)
        {
            state = inValue + m_Coef * (state - inValue);
        }

        private void SetCoef()
        {
            m_Coef = (float) Math.Exp(-1.0 / (0.001 * m_TimeConstantMs * m_SampleRate));
        }
    }
}
