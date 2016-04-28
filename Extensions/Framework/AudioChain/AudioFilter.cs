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
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using Cudafy.Host;
using DirectShowLib;
using Mpdn.Extensions.Framework.Filter;

namespace Mpdn.Extensions.Framework.AudioChain
{
    public interface IAudioFilter : IFilter<IAudioOutput>
    { }

    public abstract class AudioFilter : PinFilter<IAudioOutput>, IAudioFilter
    {
        protected abstract void Process(float[,] samples, short channels, int sampleCount);

        protected override void Initialize()
        {
            try
            {
                OnLoadAudioKernel();
            }
            catch (Exception ex)
            {
                Trace.WriteLine(ex);
            }
        }

        protected override void Dispose(bool disposing)
        {
            base.Dispose(disposing);
            try
            {
                m_SampleFormat = AudioSampleFormat.Unknown;
                DisposeGpuResources();
            }
            catch (Exception ex)
            {
                Trace.WriteLine(ex);
            }
        }

        protected override IAudioOutput DefineOutput()
        {
            return new AudioOutput(OutputFormat, Pin.Output.MediaSample);
        }

        protected virtual WaveFormatExtensible OutputFormat
        {
            get { return Pin.Output.Format; }
        }

        protected override void Render(IList<IAudioOutput> inputs)
        {
            if (inputs.Count != 1)
                throw new ArgumentException("Incorrect number of inputs.");
            if (!Process(inputs.First(), Output))
                AudioHelpers.CopySample(inputs.First().Sample, Output.Sample, true);
        }

        private bool Process(AudioSampleFormat sampleFormat, IntPtr samples, short channels, int length, IMediaSample output)
        {
            UpdateSampleFormat(sampleFormat);
            return m_ProcessFunc(samples, channels, length, output);
        }

        public virtual bool Process(IAudioOutput input, IAudioOutput output)
        {
            if (input.Format.IsBitStreaming())
                return false;

            // WARNING: We assume input and output formats are the same

            // passthrough from input to output
            AudioHelpers.CopySample(input.Sample, output.Sample, false);

            IntPtr samples;
            input.Sample.GetPointer(out samples);
            var format = input.Format;
            var bytesPerSample = format.wBitsPerSample / 8;
            var length = input.Sample.GetActualDataLength() / bytesPerSample;
            var channels = format.nChannels;
            var sampleFormat = format.SampleFormat();

            return Process(sampleFormat, samples, channels, length, output.Sample);
        }

        #region Audio Rendering

        private AudioSampleFormat m_SampleFormat = AudioSampleFormat.Unknown;
        private Func<IntPtr, short, int, IMediaSample, bool> m_ProcessFunc;

        private object m_DevInputSamples;
        private object m_DevOutputSamples;
        private float[,] m_DevNormSamples;
        private int m_Length;

        protected IAudioOutput Input { get { return Pin.Output; } }

        protected GPGPU Gpu { get { return AudioProc.Gpu; } }

        protected virtual bool CpuOnly { get { return false; } }

        protected virtual void OnLoadAudioKernel() { }

        private void UpdateSampleFormat(AudioSampleFormat sampleFormat)
        {
            if (m_SampleFormat != AudioSampleFormat.Unknown && (m_SampleFormat == sampleFormat))
                return;

            m_ProcessFunc = CpuOnly ? GetProcessCpuFunc(sampleFormat) : GetProcessFunc(sampleFormat);
            m_SampleFormat = sampleFormat;

            DisposeGpuResources();
        }

        private void UpdateGpuResources(int length)
        {
            if (m_Length == length)
                return;

            DisposeGpuResources();
            m_Length = length;
        }

        private void DisposeGpuResources()
        {
            DisposeDevInputSamples();
            DisposeDevOutputSamples();
            DisposeDevNormSamples();
            m_Length = 0;
        }

        private void DisposeDevNormSamples()
        {
            if (m_DevNormSamples == null)
                return;

            Gpu.Free(m_DevNormSamples);
            m_DevNormSamples = null;
        }

        private void DisposeDevOutputSamples()
        {
            if (m_DevOutputSamples == null)
                return;

            Gpu.Free(m_DevOutputSamples);
            m_DevOutputSamples = null;
        }

        private void DisposeDevInputSamples()
        {
            if (m_DevInputSamples == null)
                return;

            Gpu.Free(m_DevInputSamples);
            m_DevInputSamples = null;
        }

        private T[] GetDevInputSamples<T>(int length) where T : struct
        {
            return (T[])(m_DevInputSamples ?? (m_DevInputSamples = Gpu.Allocate<T>(length)));
        }

        private T[] GetDevOutputSamples<T>(int length) where T : struct
        {
            return (T[])(m_DevOutputSamples ?? (m_DevOutputSamples = Gpu.Allocate<T>(length)));
        }

        private float[,] GetDevNormSamples(int channels, int sampleCount)
        {
            return m_DevNormSamples ?? (m_DevNormSamples = Gpu.Allocate<float>(channels, sampleCount));
        }

        private Func<IntPtr, short, int, IMediaSample, bool> GetProcessFunc(AudioSampleFormat sampleFormat)
        {
            switch (sampleFormat)
            {
                case AudioSampleFormat.Float:
                    return ProcessInternal<float>;
                case AudioSampleFormat.Double:
                    return ProcessInternal<double>;
                case AudioSampleFormat.Pcm8:
                    return ProcessInternal<byte>;
                case AudioSampleFormat.Pcm16:
                    return ProcessInternal<short>;
                case AudioSampleFormat.Pcm24:
                    return ProcessInternal<AudioKernels.Int24>;
                case AudioSampleFormat.Pcm32:
                    return ProcessInternal<int>;
                default:
                    throw new ArgumentOutOfRangeException("sampleFormat");
            }
        }

        private bool ProcessInternal<T>(IntPtr samples, short channels, int length, IMediaSample output) where T : struct
        {
            UpdateGpuResources(length);

            var sampleCount = length / channels;
            try
            {
                var devInputSamples = GetDevInputSamples<T>(length);
                var devInputResult = GetDevNormSamples(channels, sampleCount);
                var devOutputResult = GetDevOutputSamples<T>(length);
                Gpu.CopyToDevice(samples, 0, devInputSamples, 0, length);
                Gpu.Launch(AudioProc.THREAD_COUNT, 1, string.Format("GetSamples{0}", typeof(T).Name),
                    devInputSamples, devInputResult);
                Process(devInputResult, channels, sampleCount);
                output.GetPointer(out samples);
                Gpu.Launch(AudioProc.THREAD_COUNT, 1, string.Format("PutSamples{0}", typeof(T).Name),
                    devInputResult, devOutputResult);
                Gpu.CopyFromDevice(devOutputResult, 0, samples, 0, length);
            }
            catch (Exception ex)
            {
                Trace.WriteLine(ex);
                return false;
            }
            return true;
        }

        private Func<IntPtr, short, int, IMediaSample, bool> GetProcessCpuFunc(AudioSampleFormat sampleFormat)
        {
            switch (sampleFormat)
            {
                case AudioSampleFormat.Float:
                    return ProcessCpuInternal<float>;
                case AudioSampleFormat.Double:
                    return ProcessCpuInternal<double>;
                case AudioSampleFormat.Pcm8:
                    return ProcessCpuInternal<byte>;
                case AudioSampleFormat.Pcm16:
                    return ProcessCpuInternal<short>;
                case AudioSampleFormat.Pcm24:
                    return ProcessCpuInternal<AudioKernels.Int24>;
                case AudioSampleFormat.Pcm32:
                    return ProcessCpuInternal<int>;
                default:
                    throw new ArgumentOutOfRangeException("sampleFormat");
            }
        }

        private bool ProcessCpuInternal<T>(IntPtr samples, short channels, int length, IMediaSample output) where T : struct
        {
            UpdateGpuResources(length);

            var inputSamples = GetInputSamplesCpu<T>(samples, channels, length);
            if (inputSamples == null)
                return false;

            var sampleCount = length / channels;
            Process(inputSamples, channels, sampleCount);

            output.GetPointer(out samples);
            return PutOutputSamplesCpu<T>(inputSamples, samples);
        }

        private bool PutOutputSamplesCpu<T>(float[,] samples, IntPtr output) where T : struct
        {
            var sampleCount = samples.GetLength(1);
            var channels = samples.GetLength(0);
            var length = sampleCount * channels;
            try
            {
                var devSamples = GetDevNormSamples(channels, sampleCount);
                var devOutput = GetDevOutputSamples<T>(length);
                Gpu.CopyToDevice(samples, devSamples);
                Gpu.Launch(AudioProc.THREAD_COUNT, 1, string.Format("PutSamples{0}", typeof(T).Name),
                    devSamples, devOutput);
                Gpu.CopyFromDevice(devOutput, 0, output, 0, length);
            }
            catch (Exception ex)
            {
                Trace.WriteLine(ex.Message);
                return false;
            }
            return true;
        }

        private float[,] GetInputSamplesCpu<T>(IntPtr samples, int channels, int length) where T : struct
        {
            var result = new float[channels, length / channels];
            try
            {
                var devSamples = GetDevInputSamples<T>(length);
                var devOutput = GetDevNormSamples(channels, length / channels);
                Gpu.CopyToDevice(samples, 0, devSamples, 0, length);
                Gpu.Launch(AudioProc.THREAD_COUNT, 1, string.Format("GetSamples{0}", typeof(T).Name),
                    devSamples, devOutput);
                Gpu.CopyFromDevice(devOutput, result);
            }
            catch (Exception ex)
            {
                Trace.WriteLine(ex.Message);
                return null;
            }
            return result;
        }

        #endregion
    }

    public interface IAudioOutput : IFilterOutput
    {
        WaveFormatExtensible Format { get; }
        IMediaSample MediaSample { get; }
        IMediaSample Sample { get; }
    }

    public class AudioOutput : FilterOutput, IAudioOutput
    {
        private IMediaSample m_Sample;

        public AudioOutput(WaveFormatExtensible format, IMediaSample mediaSample)
        {
            if (format == null) throw new ArgumentNullException("format");
            if (mediaSample == null) throw new ArgumentNullException("mediaSample");

            MediaSample = mediaSample;
            Format = format;
        }

        public WaveFormatExtensible Format { get; private set; }

        public IMediaSample MediaSample { get; private set; }

        public IMediaSample Sample
        {
            get { return m_Sample; }
        }

        public override void Allocate()
        {
            m_Sample = new MediaSample(MediaSample);
        }

        public override void Deallocate()
        {
            DisposeHelper.Dispose(ref m_Sample);
        }
    }

    public sealed class MediaSample : IMediaSample, IDisposable
    {
        private readonly int m_Size;
        private readonly IntPtr m_Buffer;

        private int m_ActualDataLength;
        private bool m_IsSyncPoint;
        private bool m_IsPreroll;
        private bool m_IsDiscontinuity;
        private long m_TimeStart;
        private long m_TimeEnd;
        private long m_MediaTimeStart;
        private long m_MediaTimeEnd;
        private bool m_HasTime;
        private bool m_HasMediaTime;
        private readonly AMMediaType m_MediaType;

        private bool m_Disposed;

        public MediaSample(IMediaSample sample)
        {
            m_Size = sample.GetSize();
            m_ActualDataLength = sample.GetActualDataLength();
            m_IsSyncPoint = sample.IsSyncPoint() == 0;
            m_IsPreroll = sample.IsPreroll() == 0;
            m_IsDiscontinuity = sample.IsDiscontinuity() == 0;
            m_HasTime = sample.GetTime(out m_TimeStart, out m_TimeEnd) == 0;
            m_HasMediaTime = sample.GetMediaTime(out m_MediaTimeStart, out m_MediaTimeEnd) == 0;
            m_Buffer = Marshal.AllocCoTaskMem(m_Size);
            // Copy the media type
            AMMediaType mediaType;
            if (sample.GetMediaType(out mediaType) == 0)
            {
                m_MediaType = mediaType;
            }
        }

        public int GetPointer(out IntPtr ppBuffer)
        {
            ppBuffer = m_Buffer;
            return 0;
        }

        public int GetSize()
        {
            return m_Size;
        }

        public int GetTime(out long pTimeStart, out long pTimeEnd)
        {
            pTimeStart = m_TimeStart;
            pTimeEnd = m_TimeEnd;
            return !m_HasTime ? -2147220919 /* VFW_E_SAMPLE_TIME_NOT_SET */ : 0;
        }

        public int SetTime(DsLong pTimeStart, DsLong pTimeEnd)
        {
            m_TimeStart = pTimeStart.ToInt64();
            m_TimeEnd = pTimeEnd.ToInt64();
            m_HasTime = true;
            return 0;
        }

        public int IsSyncPoint()
        {
            return m_IsSyncPoint ? 0 : 1;
        }

        public int SetSyncPoint(bool bIsSyncPoint)
        {
            m_IsSyncPoint = bIsSyncPoint;
            return 0;
        }

        public int IsPreroll()
        {
            return m_IsPreroll ? 0 : 1;
        }

        public int SetPreroll(bool bIsPreroll)
        {
            m_IsPreroll = bIsPreroll;
            return 0;
        }

        public int GetActualDataLength()
        {
            return m_ActualDataLength;
        }

        public int SetActualDataLength(int len)
        {
            m_ActualDataLength = len;
            return 0;
        }

        public int GetMediaType(out AMMediaType mediaType)
        {
            mediaType = m_MediaType;
            return m_MediaType == null ? 1 : 0;
        }

        public int SetMediaType(AMMediaType pMediaType)
        {
            return 1; // not supported
        }

        public int IsDiscontinuity()
        {
            return m_IsDiscontinuity ? 0 : 1;
        }

        public int SetDiscontinuity(bool bDiscontinuity)
        {
            m_IsDiscontinuity = bDiscontinuity;
            return 0;
        }

        public int GetMediaTime(out long pTimeStart, out long pTimeEnd)
        {
            pTimeStart = m_MediaTimeStart;
            pTimeEnd = m_MediaTimeEnd;
            return !m_HasMediaTime ? -2147220911 : 0;
        }

        public int SetMediaTime(DsLong pTimeStart, DsLong pTimeEnd)
        {
            m_MediaTimeStart = pTimeStart.ToInt64();
            m_MediaTimeEnd = pTimeEnd.ToInt64();
            m_HasMediaTime = true;
            return 0;
        }

        public void Dispose()
        {
            if (m_Disposed)
                return;

            Marshal.FreeCoTaskMem(m_Buffer);
            if (m_MediaType != null)
            {
                DsUtils.FreeAMMediaType(m_MediaType);
            }
            m_Disposed = true;
        }
    }
}
