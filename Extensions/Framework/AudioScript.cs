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

        private Func<IntPtr, int, int, float[,]> m_GetInputSamplesFunc;
        private Func<float[,], IntPtr, bool> m_PutOutputSamplesFunc;
        private AudioSampleFormat m_SampleFormat = AudioSampleFormat.Unknown;

        protected abstract void Process(float[,] samples);

        #region Implementation

        public bool Process()
        {
            if (Audio.InputFormat.IsBitStreaming())
                return false;

            // WARNING: We assume input and output formats are the same

            var input = Audio.Input;
            var output = Audio.Output;

            // passthrough from input to output
            AudioHelpers.CopySample(input, output);

            IntPtr samples;
            output.GetPointer(out samples);
            var outputFormat = Audio.InputFormat;
            var bytesPerSample = outputFormat.wBitsPerSample/8;
            var length = output.GetActualDataLength() / bytesPerSample;
            var channels = outputFormat.nChannels;

            var sampleFormat = outputFormat.SampleFormat();
            if (m_SampleFormat == AudioSampleFormat.Unknown || (m_SampleFormat != sampleFormat))
            {
                m_GetInputSamplesFunc = GetInputSamplesFunc(sampleFormat);
                m_PutOutputSamplesFunc = PutOutputSamplesFunc(sampleFormat);
                m_SampleFormat = sampleFormat;
            }

            var inputSamples = m_GetInputSamplesFunc(samples, channels, length);
            if (inputSamples == null)
                return false;

            Process(inputSamples);

            return m_PutOutputSamplesFunc(inputSamples, samples);
        }

        private Func<IntPtr, int, int, float[,]> GetInputSamplesFunc(AudioSampleFormat sampleFormat)
        {
            switch (sampleFormat)
            {
                case AudioSampleFormat.Float:
                    return GetInputSamples<float>;
                case AudioSampleFormat.Double:
                    return GetInputSamples<double>;
                case AudioSampleFormat.Pcm8:
                    return GetInputSamples<byte>;
                case AudioSampleFormat.Pcm16:
                    return GetInputSamples<short>;
                case AudioSampleFormat.Pcm24:
                    return GetInputSamples<AudioKernels.UInt24>;
                case AudioSampleFormat.Pcm32:
                    return GetInputSamples<int>;
                default:
                    throw new ArgumentOutOfRangeException("sampleFormat");
            }
        }

        private Func<float[,], IntPtr, bool> PutOutputSamplesFunc(AudioSampleFormat sampleFormat)
        {
            switch (sampleFormat)
            {
                case AudioSampleFormat.Float:
                    return PutOutputSamples<float>;
                case AudioSampleFormat.Double:
                    return PutOutputSamples<double>;
                case AudioSampleFormat.Pcm8:
                    return PutOutputSamples<byte>;
                case AudioSampleFormat.Pcm16:
                    return PutOutputSamples<short>;
                case AudioSampleFormat.Pcm24:
                    return PutOutputSamples<AudioKernels.UInt24>;
                case AudioSampleFormat.Pcm32:
                    return PutOutputSamples<int>;
                default:
                    throw new ArgumentOutOfRangeException("sampleFormat");
            }
        }

        private bool PutOutputSamples<T>(float[,] samples, IntPtr output) where T : struct
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
                    gpu.Launch(Math.Min(THREAD_COUNT, length), 1, string.Format("PutSamples{0}", typeof (T).Name),
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
                Trace.WriteLine(ex.Message);
                return false;
            }
            return true;
        }

        private float[,] GetInputSamples<T>(IntPtr samples, int channels, int length) where T : struct
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
                    gpu.Launch(Math.Min(THREAD_COUNT, length), 1, string.Format("GetSamples{0}", typeof (T).Name),
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
                m_Gpu.UnloadModule(AudioKernels.KernelModule);
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
                var index = devices.TakeWhile(d => d.Integrated).Count();
                if (index >= devices.Count())
                {
                    Trace.WriteLine("GPGPU Warning: Unable to use CPU as GPGPU!");
                    index = 0;
                }
                m_Gpu = CudafyHost.GetDevice(eGPUType.OpenCL, index);
                m_Gpu.LoadModule(AudioKernels.KernelModule);
            }
            catch (Exception ex)
            {
                Trace.WriteLine(ex.Message);
            }
            
            // Provides script a chance to change the output format
        }

        public virtual void OnNewSegment(long startTime, long endTime, double rate)
        {
        }

        #endregion
    }
}
