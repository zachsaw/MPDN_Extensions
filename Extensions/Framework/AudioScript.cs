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
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using Cudafy;
using Cudafy.Host;
using DirectShowLib;
using Mpdn.AudioScript;
using Mpdn.Extensions.Framework.Config;
using Mpdn.OpenCl;
using Mpdn.RenderScript;

namespace Mpdn.Extensions.Framework
{
    public interface IAudioChain : IAudioScript
    {
        bool Process(AudioParam input, AudioParam output);
    }

    public struct AudioParam
    {
        public WaveFormatExtensible Format;
        public IMediaSample Sample;

        public AudioParam(WaveFormatExtensible format, IMediaSample sample)
        {
            Format = format;
            Sample = sample;
        }
    }

    public sealed class MediaSample : IMediaSample, IDisposable
    {
        private readonly IMediaSample m_Sample;
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

        private bool m_Disposed;

        public MediaSample(IMediaSample sample)
        {
            m_Sample = sample;
            m_Size = sample.GetSize();
            m_ActualDataLength = sample.GetActualDataLength();
            m_IsSyncPoint = sample.IsSyncPoint() == 0;
            m_IsPreroll = sample.IsPreroll() == 0;
            m_IsDiscontinuity = sample.IsDiscontinuity() == 0;
            sample.GetTime(out m_TimeStart, out m_TimeEnd);
            sample.GetMediaTime(out m_MediaTimeStart, out m_MediaTimeEnd);
            m_Buffer = Marshal.AllocCoTaskMem(m_Size);
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
            return 0;
        }

        public int SetTime(DsLong pTimeStart, DsLong pTimeEnd)
        {
            m_TimeStart = pTimeStart.ToInt64();
            m_TimeEnd = pTimeEnd.ToInt64();
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

        public int GetMediaType(out AMMediaType ppMediaType)
        {
            return m_Sample.GetMediaType(out ppMediaType);
        }

        public int SetMediaType(AMMediaType pMediaType)
        {
            throw new InvalidOperationException("Changing media type not supported");
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
            return 0;
        }

        public int SetMediaTime(DsLong pTimeStart, DsLong pTimeEnd)
        {
            m_MediaTimeStart = pTimeStart.ToInt64();
            m_MediaTimeEnd = pTimeEnd.ToInt64();
            return 0;
        }

        public void Dispose()
        {
            if (m_Disposed)
                return;

            Marshal.FreeCoTaskMem(m_Buffer);
            m_Disposed = true;
        }
    }

    public abstract class AudioScript : AudioScript<NoSettings> { }

    public abstract class AudioScript<TSettings> : AudioScript<TSettings, ScriptConfigDialog<TSettings>>
        where TSettings : class, new()
    { }

    public abstract class AudioScript<TSettings, TDialog> : ExtensionUi<Config.Internal.AudioScripts, TSettings, TDialog>, IAudioChain
        where TSettings : class, new()
        where TDialog : ScriptConfigDialog<TSettings>, new()
    {
        private const int THREAD_COUNT = 512;

        private GPGPU m_Gpu;

        private AudioSampleFormat m_SampleFormat = AudioSampleFormat.Unknown;
        private Func<IntPtr, short, int, IMediaSample, bool> m_ProcessFunc;

        private object m_DevInputSamples;
        private object m_DevOutputSamples;
        private float[,] m_DevNormSamples;
        private int m_Length;

        protected abstract void Process(float[,] samples, short channels, int sampleCount);

        #region Implementation

        protected GPGPU Gpu { get { return m_Gpu; } }

        protected virtual bool CpuOnly { get { return false; } }

        protected virtual void OnLoadAudioKernel() { }

        public virtual bool Process()
        {
            return Process(new AudioParam(Audio.InputFormat, Audio.Input), new AudioParam(Audio.OutputFormat, Audio.Output));
        }

        private bool Process(AudioSampleFormat sampleFormat, IntPtr samples, short channels, int length, IMediaSample output)
        {
            UpdateSampleFormat(sampleFormat);
            return m_ProcessFunc(samples, channels, length, output);
        }

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
            return (T[]) (m_DevInputSamples ?? (m_DevInputSamples = Gpu.Allocate<T>(length)));
        }

        private T[] GetDevOutputSamples<T>(int length) where T : struct
        {
            return (T[]) (m_DevOutputSamples ?? (m_DevOutputSamples = Gpu.Allocate<T>(length)));
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

            var gpu = m_Gpu;
            var sampleCount = length/channels;
            try
            {
                var devInputSamples = GetDevInputSamples<T>(length);
                var devInputResult = GetDevNormSamples(channels, sampleCount);
                var devOutputResult = GetDevOutputSamples<T>(length);
                gpu.CopyToDevice(samples, 0, devInputSamples, 0, length);
                gpu.Launch(THREAD_COUNT, 1, string.Format("GetSamples{0}", typeof(T).Name),
                    devInputSamples, devInputResult);
                Process(devInputResult, channels, sampleCount);
                output.GetPointer(out samples);
                gpu.Launch(THREAD_COUNT, 1, string.Format("PutSamples{0}", typeof(T).Name),
                    devInputResult, devOutputResult);
                gpu.CopyFromDevice(devOutputResult, 0, samples, 0, length);
            }
            catch (Exception ex)
            {
                Trace.WriteLine(ex.Message);
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
            var length = sampleCount*channels;
            var gpu = m_Gpu;
            try
            {
                var devSamples = GetDevNormSamples(channels, sampleCount);
                var devOutput = GetDevOutputSamples<T>(length);
                gpu.CopyToDevice(samples, devSamples);
                gpu.Launch(THREAD_COUNT, 1, string.Format("PutSamples{0}", typeof(T).Name),
                    devSamples, devOutput);
                gpu.CopyFromDevice(devOutput, 0, output, 0, length);
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
            var gpu = m_Gpu;
            var result = new float[channels, length / channels];
            try
            {
                var devSamples = GetDevInputSamples<T>(length);
                var devOutput = GetDevNormSamples(channels, length / channels);
                gpu.CopyToDevice(samples, 0, devSamples, 0, length);
                gpu.Launch(THREAD_COUNT, 1, string.Format("GetSamples{0}", typeof(T).Name),
                    devSamples, devOutput);
                gpu.CopyFromDevice(devOutput, result);
            }
            catch (Exception ex)
            {
                Trace.WriteLine(ex.Message);
                return null;
            }
            return result;
        }

        private void Cleanup()
        {
            try
            {
                m_SampleFormat = AudioSampleFormat.Unknown;

                if (m_Gpu == null)
                    return;

                DisposeGpuResources();

                m_Gpu.FreeAll();
                m_Gpu.HostFreeAll();
                m_Gpu.DestroyStreams();
                m_Gpu.UnloadModules();
                DisposeHelper.Dispose(ref m_Gpu);
            }
            catch (Exception ex)
            {
                Trace.WriteLine(ex.Message);
            }
        }

        public virtual void OnMediaClosed()
        {
            Cleanup();
        }

        public virtual bool Process(AudioParam input, AudioParam output)
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

        public virtual void OnGetMediaType(WaveFormatExtensible format)
        {
            try
            {
                var devices = CudafyHost.GetDeviceProperties(eGPUType.OpenCL).ToArray();
                var device = devices.FirstOrDefault(d => d.Integrated && d.PlatformName.Contains("Intel(R)")); // use Intel iGPU if possible
                if (device == null || IsInUseForVideoRendering(device))
                {
                    // Fallback to CPU (prefer AMD Accelerated Parallel Processing first as it is faster)
                    const string cpuId = " CPU ";
                    device = devices.FirstOrDefault(d => d.Name.Contains(cpuId) && d.PlatformName.Contains("AMD"));
                    if (device == null)
                    {
                        // Settle for any CPU OpenCL device (most likely Intel OpenCL) as the last resort
                        device = devices.FirstOrDefault(d => d.Name.Contains(cpuId));
                        if (device == null)
                        {
                            throw new OpenClException("Error: CPU OpenCL drivers not installed");
                        }
                    }
                }
                m_Gpu = CudafyHost.GetDevice(eGPUType.OpenCL, device.DeviceId);
                m_Gpu.LoadAudioKernel(typeof(AudioKernels));
                OnLoadAudioKernel();
            }
            catch (Exception ex)
            {
                Trace.WriteLine(ex.Message);
            }
            
            // Provides script a chance to change the output format
        }

        private static bool IsInUseForVideoRendering(GPGPUProperties device)
        {
            var name1 = device.Name.Trim();
            var name2 = Renderer.Dx9GpuInfo.Details.Description.Trim();
            return name1.Contains(name2) || name2.Contains(name1);
        }

        public virtual void OnNewSegment(long startTime, long endTime, double rate)
        {
        }

        #endregion
    }
}
