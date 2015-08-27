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
using Cudafy;
using Cudafy.Host;
using DirectShowLib;
using Mpdn.AudioScript;
using Mpdn.Extensions.Framework.Config;
using Mpdn.OpenCl;
using Mpdn.RenderScript;

namespace Mpdn.Extensions.Framework
{
    public abstract class AudioScript : AudioScript<NoSettings> { }

    public abstract class AudioScript<TSettings> : AudioScript<TSettings, ScriptConfigDialog<TSettings>>
        where TSettings : class, new()
    { }

    public abstract class AudioScript<TSettings, TDialog> : ExtensionUi<Config.Internal.AudioScripts, TSettings, TDialog>, IAudioScript
        where TSettings : class, new()
        where TDialog : ScriptConfigDialog<TSettings>, new()
    {
        private const int THREAD_COUNT = 512;

        private GPGPU m_Gpu;

        private AudioSampleFormat m_SampleFormat = AudioSampleFormat.Unknown;
        private Func<IntPtr, short, int, IMediaSample, bool> m_ProcessFunc;

        protected abstract void Process(float[,] samples, short channels, int sampleCount);

        #region Implementation

        protected GPGPU Gpu { get { return m_Gpu; } }

        protected virtual bool CpuOnly { get { return false; } }

        protected virtual void OnLoadAudioKernel() { }

        public virtual bool Process()
        {
            if (Audio.InputFormat.IsBitStreaming())
                return false;

            // WARNING: We assume input and output formats are the same

            var input = Audio.Input;
            var output = Audio.Output;

            // passthrough from input to output
            AudioHelpers.CopySample(input, output, false);

            IntPtr samples;
            input.GetPointer(out samples);
            var format = Audio.InputFormat;
            var bytesPerSample = format.wBitsPerSample/8;
            var length = input.GetActualDataLength()/bytesPerSample;
            var channels = format.nChannels;
            var sampleFormat = format.SampleFormat();

            return Process(sampleFormat, samples, channels, length, output);
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
            var gpu = m_Gpu;
            var sampleCount = length/channels;
            var result = new float[channels, sampleCount];
            try
            {
                var devInputSamples = gpu.Allocate<T>(length);
                var devInputResult = gpu.Allocate(result);
                var devOutputResult = gpu.Allocate<T>(length);
                try
                {
                    gpu.CopyToDevice(samples, 0, devInputSamples, 0, length);
                    gpu.Launch(THREAD_COUNT, 1, string.Format("GetSamples{0}", typeof(T).Name),
                        devInputSamples, devInputResult);
                    Process(devInputResult, channels, sampleCount);
                    output.GetPointer(out samples);
                    gpu.Launch(THREAD_COUNT, 1, string.Format("PutSamples{0}", typeof(T).Name),
                        devInputResult, devOutputResult);
                    gpu.CopyFromDevice(devOutputResult, 0, samples, 0, length);
                }
                finally
                {
                    gpu.Free(devInputSamples);
                    gpu.Free(devInputResult);
                    gpu.Free(devOutputResult);
                }
            }
            catch (Exception ex)
            {
                gpu.FreeAll();
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
                var devSamples = gpu.Allocate<float>(channels, sampleCount);
                var devOutput = gpu.Allocate<T>(length);
                try
                {
                    gpu.CopyToDevice(samples, devSamples);
                    gpu.Launch(THREAD_COUNT, 1, string.Format("PutSamples{0}", typeof(T).Name),
                        devSamples, devOutput);
                    gpu.CopyFromDevice(devOutput, 0, output, 0, length);
                }
                finally
                {
                    gpu.Free(devSamples);
                    gpu.Free(devOutput);
                }
            }
            catch (Exception ex)
            {
                gpu.FreeAll();
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
                var devSamples = gpu.Allocate<T>(length);
                var devOutput = gpu.Allocate(result);
                try
                {
                    gpu.CopyToDevice(samples, 0, devSamples, 0, length);
                    gpu.Launch(THREAD_COUNT, 1, string.Format("GetSamples{0}", typeof(T).Name),
                        devSamples, devOutput);
                    gpu.CopyFromDevice(devOutput, result);
                }
                finally
                {
                    gpu.Free(devSamples);
                    gpu.Free(devOutput);
                }
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
