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
using Cudafy;
using Mpdn.Extensions.Framework;

namespace Mpdn.Extensions.AudioScripts
{
    public class DynamicRangeCompressorSettings
    {
        public float ThresholddB { get; set; }
        public float Ratio { get; set; }
        public float MakeupGaindB { get; set; }
        public int AttackMs { get; set; }
        public int ReleaseMs { get; set; }

        public DynamicRangeCompressorSettings()
        {
            ThresholddB = -15;
            Ratio = 3;
            MakeupGaindB = 3;
            AttackMs = 200; // 0.2s
            ReleaseMs = 1000; // 1s
        }
    }

    public class DynamicRangeCompressor :
        AudioScript<DynamicRangeCompressorSettings, DynamicRangeCompressorConfigDialog>
    {
        // DC offset to prevent denormal
        protected const float DC_OFFSET = 1.0e-25f;

        private float m_EnvdB = DC_OFFSET;
        private EnvelopeDetector m_Attack;
        private EnvelopeDetector m_Release;
        private int m_SampleRate;
        private int m_AttackMs;
        private int m_ReleaseMs;
        private int m_SampleCount;

        private float[] m_DevOverdBs;

        public override ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("C29B883B-AF2B-45F7-9FDA-1F8A67004E2E"),
                    Name = "DynamicRangeCompressor",
                    Description = "Performs simple DRC with OpenCL",
                    Copyright = "Adapted from NAudio SimpleCompressor by Zachs"
                };
            }
        }

        protected override void OnLoadAudioKernel()
        {
            Gpu.LoadAudioKernel(typeof (Decibels), typeof (DynamicRangeCompressorKernel));
        }

        public override void OnNewSegment(long startTime, long endTime, double rate)
        {
            base.OnNewSegment(startTime, endTime, rate);

            m_EnvdB = DC_OFFSET;
        }

        public override void OnMediaClosed()
        {
            DisposeGpuResources();
            base.OnMediaClosed();
        }

        private void DisposeGpuResources()
        {
            if (m_DevOverdBs == null)
                return;

            Gpu.Free(m_DevOverdBs);
            m_DevOverdBs = null;
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

            Compress(samples, sampleCount, Settings.ThresholddB, Settings.Ratio, Settings.MakeupGaindB);
        }

        private void Compress(float[,] samples, int sampleCount, float thresholddB, float ratio, float makeupGaindB)
        {
            const int threadCount = 512;

            var makeupGainLin = Decibels.ToLinear(makeupGaindB);
            if (m_SampleCount != sampleCount)
            {
                DisposeGpuResources();
                m_DevOverdBs = Gpu.Allocate<float>(sampleCount);
            }
            var devOverdBs = m_DevOverdBs;
            Gpu.Launch(threadCount, 1).GetOverDecibels(samples, thresholddB, devOverdBs);
            var overdBs = new float[sampleCount];
            Gpu.CopyFromDevice(devOverdBs, overdBs);

            // This bit is serial, can't be done on GPU
            for (int i = 0; i < sampleCount; i++)
            {
                // attack/release
                var overdB = overdBs[i];
                // assumes that:
                // positive delta = attack
                // negative delta = release
                // good for linear & log values
                if (overdB > m_EnvdB)
                {
                    m_Attack.Run(overdB, ref m_EnvdB); // attack
                }
                else
                {
                    m_Release.Run(overdB, ref m_EnvdB); // release
                }
                overdBs[i] = m_EnvdB;
            }

            Gpu.CopyToDevice(overdBs, devOverdBs);
            Gpu.Launch(threadCount, 1).ApplyGains(samples, devOverdBs, ratio, makeupGainLin);
        }
    }

    internal class EnvelopeDetector
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
            get { return m_TimeConstantMs; }
            set
            {
                Debug.Assert(value > 0.0);
                m_TimeConstantMs = value;
                SetCoef();
            }
        }

        public float SampleRate
        {
            get { return m_SampleRate; }
            set
            {
                Debug.Assert(value > 0.0);
                m_SampleRate = value;
                SetCoef();
            }
        }

        public void Run(float inValue, ref float state)
        {
            state = inValue + m_Coef*(state - inValue);
        }

        private void SetCoef()
        {
            m_Coef = (float) Math.Exp(-1.0/(0.001*m_TimeConstantMs*m_SampleRate));
        }
    }

    public static class DynamicRangeCompressorKernel
    {
        // DC offset to prevent denormal
        private const float DC_OFFSET = 1.0e-25f;

        [Cudafy]
        public static void GetOverDecibels(GThread thread, float[,] samples, float thresholddB, float[] overdBs)
        {
            var channels = samples.GetLength(0);
            var sampleCount = samples.GetLength(1);

            int tid = thread.blockIdx.x;
            while (tid < sampleCount)
            {
                var allChMax = float.MinValue;
                for (int c = 0; c < channels; c++)
                {
                    var s = samples[c, tid];
                    s = s > 0 ? s : -s; // fabs(s)
                    allChMax = GMath.Max(s, allChMax);
                }

                allChMax += DC_OFFSET;
                var allChMaxdB = Decibels.FromLinear(allChMax);

                var overdB = GMath.Max(allChMaxdB - thresholddB, 0);
                overdB += DC_OFFSET;

                overdBs[tid] = overdB;

                tid += thread.gridDim.x;
            }
        }

        [Cudafy]
        public static void ApplyGains(GThread thread, float[,] samples, float[] overdBs, float ratio,
            float makeupGainLin)
        {
            var channels = samples.GetLength(0);
            var sampleCount = samples.GetLength(1);

            int tid = thread.blockIdx.x;
            while (tid < sampleCount)
            {
                var overdB = overdBs[tid] - DC_OFFSET;

                var gr = -overdB*(ratio - 1.0f);
                gr = Decibels.ToLinear(gr)*makeupGainLin;

                for (int c = 0; c < channels; c++)
                {
                    var s = samples[c, tid];
                    s *= gr;
                    samples[c, tid] = s;
                }

                tid += thread.gridDim.x;
            }
        }
    }
}
