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
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using DirectShowLib;

namespace Mpdn.Extensions.Framework
{
    public enum AudioSampleFormat
    {
        Unknown,
        Float,
        Double,
        Pcm8,
        Pcm16,
        Pcm24,
        Pcm32
    }

    public static class AudioHelpers
    {
        private const int S_OK = 0;

        private const short WAVE_FORMAT_PCM = 1;
        private const short WAVE_FORMAT_IEEE_FLOAT = 3;
        private const short WAVE_FORMAT_EXTENSIBLE = unchecked((short)0xFFFE);

        [DllImport("kernel32.dll", EntryPoint = "CopyMemory", SetLastError = false)]
        public static extern void CopyMemory(IntPtr dest, IntPtr src, int count);

        public static void CopySample(IMediaSample src, IMediaSample dest, bool copySamples)
        {
            var sourceSize = src.GetActualDataLength();

            if (copySamples)
            {
                IntPtr sourceBuffer;
                src.GetPointer(out sourceBuffer);

                IntPtr destBuffer;
                dest.GetPointer(out destBuffer);

                CopyMemory(destBuffer, sourceBuffer, sourceSize);
            }

            // Copy the sample times
            long start, end;

            if (src.GetTime(out start, out end) == S_OK)
            {
                dest.SetTime(start, end);
            }

            if (src.GetMediaTime(out start, out end) == S_OK)
            {
                dest.SetMediaTime(start, end);
            }

            // Copy the media type
            AMMediaType mediaType;
            src.GetMediaType(out mediaType);
            dest.SetMediaType(mediaType);
            DsUtils.FreeAMMediaType(mediaType);

            dest.SetSyncPoint(src.IsSyncPoint() == S_OK);
            dest.SetPreroll(src.IsPreroll() == S_OK);
            dest.SetDiscontinuity(src.IsDiscontinuity() == S_OK);

            // Copy the actual data length
            dest.SetActualDataLength(sourceSize);
        }

        public static AudioSampleFormat SampleFormat(this WaveFormatExtensible format)
        {
            return GetSampleFormat(format);
        }

        public static AudioSampleFormat GetSampleFormat(WaveFormatExtensible format)
        {
            if (format.nSamplesPerSec == 0)
                return AudioSampleFormat.Unknown;

            switch (format.wFormatTag)
            {
                case WAVE_FORMAT_IEEE_FLOAT:
                    switch (format.wBitsPerSample)
                    {
                        case 32: return AudioSampleFormat.Float;
                        case 64: return AudioSampleFormat.Double;
                    }
                    break;
                case WAVE_FORMAT_PCM:
                    switch (format.wBitsPerSample)
                    {
                        case 8: return AudioSampleFormat.Pcm8;
                        case 16: return AudioSampleFormat.Pcm16;
                        case 24: return AudioSampleFormat.Pcm24;
                        case 32: return AudioSampleFormat.Pcm32;
                    }
                    break;
                case WAVE_FORMAT_EXTENSIBLE:
                    if (format.SubFormat == MediaSubType.IEEE_FLOAT)
                    {
                        switch (format.wBitsPerSample)
                        {
                            case 32: return AudioSampleFormat.Float;
                            case 64: return AudioSampleFormat.Double;
                        }
                    }
                    else if (format.SubFormat == MediaSubType.PCM)
                    {
                        switch (format.wBitsPerSample)
                        {
                            case 8: return AudioSampleFormat.Pcm8;
                            case 16: return AudioSampleFormat.Pcm16;
                            case 24: return AudioSampleFormat.Pcm24;
                            case 32: return AudioSampleFormat.Pcm32;
                        }
                    }
                    break;
            }

            return AudioSampleFormat.Unknown;
        }

        public static bool IsBitStreaming(this WaveFormatExtensible format)
        {
            return IsBitstreaming(format);
        }

        public static bool IsBitstreaming(WaveFormatExtensible format)
        {
            return GetSampleFormat(format) == AudioSampleFormat.Unknown;
        }

        public static void LoadAudioKernel(this GPGPU gpu, Type type, params Type[] types)
        {
            gpu.LoadModule(GetCudafyModule(type, types), false);
        }

        private static CudafyModule GetCudafyModule(Type type, params Type[] types)
        {
            var joined = string.Join(",", type.ToString(), string.Join(",", types.Select(t => t.ToString()))).TrimEnd(',');
            var filename = string.Format("{0}.cdfy", Path.Combine(AudioKernelCacheRoot, joined));
            var km = CudafyModule.TryDeserialize(filename);
            if (km != null && km.TryVerifyChecksums())
                return km;

            var ts = new Type[types.Length + 1];
            ts[0] = type;
            Array.Copy(types, 0, ts, 1, types.Length);
            km = CudafyTranslator.Cudafy(eArchitecture.OpenCL, ts);
            Directory.CreateDirectory(AudioKernelCacheRoot);
            km.Serialize(filename);

            return km;
        }

        private static string AudioKernelCacheRoot
        {
            get { return AppPath.GetUserDataDir("CudafyCache"); }
        }
    }

    public static class Decibels
    {
        // 20 / ln( 10 )
        private const float LOG_2_DB = 8.6858896380650365530225783783321f;

        // ln( 10 ) / 20
        private const float DB_2_LOG = 0.11512925464970228420089957273422f;

        /// <summary>
        /// linear to dB conversion
        /// </summary>
        /// <param name="lin">linear value</param>
        /// <returns>decibel value</returns>
        [Cudafy]
        public static float FromLinear(float lin)
        {
            return GMath.Log(lin)*LOG_2_DB;
        }

        /// <summary>
        /// dB to linear conversion
        /// </summary>
        /// <param name="dB">decibel value</param>
        /// <returns>linear value</returns>
        [Cudafy]
        public static float ToLinear(float dB)
        {
            return GMath.Exp(dB*DB_2_LOG);
        }
    }
}