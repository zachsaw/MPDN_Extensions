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
using System.Runtime.InteropServices;
using Cudafy;
using Cudafy.Translator;

namespace Mpdn.Extensions.Framework
{
    public static class AudioKernels
    {
        [Cudafy]
        public static void GetSamplesByte(GThread thread, byte[] samples, float[,] output)
        {
            var channels = output.GetLength(0);
            var sampleCount = output.GetLength(1);
            const float mid = 128;

            int tid = thread.blockIdx.x;
            while (tid < sampleCount)
            {
                for (int i = 0; i < channels; i++)
                {
                    output[i, tid] = (samples[(tid * channels) + i] / mid) - 1.0f;
                }
                tid += thread.gridDim.x;
            }
        }

        [Cudafy]
        public static void PutSamplesByte(GThread thread, float[,] samples, byte[] output)
        {
            var channels = samples.GetLength(0);
            var sampleCount = samples.GetLength(1);
            const float mid = 128;

            int tid = thread.blockIdx.x;
            while (tid < sampleCount)
            {
                for (int i = 0; i < channels; i++)
                {
                    var f = Math.Max(-1.0f, Math.Min(1.0f, samples[i, tid]));
                    output[(tid * channels) + i] = (byte)((f + 1.0f) * mid);
                }
                tid += thread.gridDim.x;
            }
        }

        [Cudafy]
        public static void GetSamplesInt16(GThread thread, short[] samples, float[,] output)
        {
            var channels = output.GetLength(0);
            var sampleCount = output.GetLength(1);
            const float mid = -short.MinValue;
            const float min = -short.MaxValue;

            int tid = thread.blockIdx.x;
            while (tid < sampleCount)
            {
                for (int i = 0; i < channels; i++)
                {
                    output[i, tid] = ((samples[(tid * channels) + i] - min) / mid) - 1.0f;
                }
                tid += thread.gridDim.x;
            }
        }

        [Cudafy]
        public static void PutSamplesInt16(GThread thread, float[,] samples, short[] output)
        {
            var channels = samples.GetLength(0);
            var sampleCount = samples.GetLength(1);
            const float mid = -short.MinValue;
            const float min = -short.MaxValue;

            int tid = thread.blockIdx.x;
            while (tid < sampleCount)
            {
                for (int i = 0; i < channels; i++)
                {
                    var f = Math.Max(-1.0f, Math.Min(1.0f, samples[i, tid]));
                    output[(tid * channels) + i] = (short)((f + 1.0f) * mid + min);
                }
                tid += thread.gridDim.x;
            }
        }

        // ReSharper disable InconsistentNaming
        public struct int24
        {
            public const int MinValue = -8388608;
            public const int MaxValue = 8388607;
        }
        // ReSharper restore InconsistentNaming

        [Cudafy]
        [StructLayout(LayoutKind.Sequential)]
        public struct Int24
        {
            public byte B0;
            public byte B1;
            public byte B2;
        }

        [Cudafy]
        public static void GetSamplesInt24(GThread thread, Int24[] samples, float[,] output)
        {
            var channels = output.GetLength(0);
            var sampleCount = output.GetLength(1);
            const float mid = -int24.MinValue;
            const float min = -int24.MaxValue;

            int tid = thread.blockIdx.x;
            while (tid < sampleCount)
            {
                for (int i = 0; i < channels; i++)
                {
                    var index = (tid * channels) + i;
                    var b0 = samples[index].B0;
                    var b1 = samples[index].B1;
                    var b2 = samples[index].B2;
                    var v = b0 | (b1 << 8) | (b2 << 16);
                    if ((v & 0x800000) != 0)
                    {
                        v |= ~0xffffff;
                    }
                    output[i, tid] = ((v - min) / mid) - 1.0f;
                }
                tid += thread.gridDim.x;
            }
        }

        [Cudafy]
        public static void PutSamplesInt24(GThread thread, float[,] samples, Int24[] output)
        {
            var channels = samples.GetLength(0);
            var sampleCount = samples.GetLength(1);
            const float mid = -int24.MinValue;
            const float min = -int24.MaxValue;

            int tid = thread.blockIdx.x;
            while (tid < sampleCount)
            {
                for (int i = 0; i < channels; i++)
                {
                    var f = Math.Max(-1.0f, Math.Min(1.0f, samples[i, tid]));
                    var val = (int)((f + 1.0f) * mid + min);
                    var index = (tid * channels) + i;
                    output[index].B0 = (byte)(val & 0xFF);
                    output[index].B1 = (byte)(val >> 8);
                    output[index].B2 = (byte)(val >> 16);
                }
                tid += thread.gridDim.x;
            }
        }

        [Cudafy]
        public static void GetSamplesInt32(GThread thread, int[] samples, float[,] output)
        {
            var channels = output.GetLength(0);
            var sampleCount = output.GetLength(1);
            const float mid = -((float)int.MinValue);
            const float min = -int.MaxValue;

            int tid = thread.blockIdx.x;
            while (tid < sampleCount)
            {
                for (int i = 0; i < channels; i++)
                {
                    output[i, tid] = ((samples[(tid * channels) + i] - min) / mid) - 1.0f;
                }
                tid += thread.gridDim.x;
            }
        }

        [Cudafy]
        public static void PutSamplesInt32(GThread thread, float[,] samples, int[] output)
        {
            var channels = samples.GetLength(0);
            var sampleCount = samples.GetLength(1);
            const float mid = -((float)int.MinValue);
            const float min = -int.MaxValue;

            int tid = thread.blockIdx.x;
            while (tid < sampleCount)
            {
                for (int i = 0; i < channels; i++)
                {
                    var f = Math.Max(-1.0f, Math.Min(1.0f, samples[i, tid]));
                    output[(tid * channels) + i] = (int)((f + 1.0f) * mid + min);
                }
                tid += thread.gridDim.x;
            }
        }

        [Cudafy]
        public static void GetSamplesSingle(GThread thread, float[] samples, float[,] output)
        {
            var channels = output.GetLength(0);
            var sampleCount = output.GetLength(1);

            int tid = thread.blockIdx.x;
            while (tid < sampleCount)
            {
                for (int i = 0; i < channels; i++)
                {
                    output[i, tid] = samples[(tid * channels) + i];
                }
                tid += thread.gridDim.x;
            }
        }

        [Cudafy]
        public static void PutSamplesSingle(GThread thread, float[,] samples, float[] output)
        {
            var channels = samples.GetLength(0);
            var sampleCount = samples.GetLength(1);

            int tid = thread.blockIdx.x;
            while (tid < sampleCount)
            {
                for (int i = 0; i < channels; i++)
                {
                    var f = Math.Max(-1.0f, Math.Min(1.0f, samples[i, tid]));
                    output[(tid * channels) + i] = f;
                }
                tid += thread.gridDim.x;
            }
        }

        [Cudafy]
        public static void GetSamplesDouble(GThread thread, double[] samples, float[,] output)
        {
            var channels = output.GetLength(0);
            var sampleCount = output.GetLength(1);

            int tid = thread.blockIdx.x;
            while (tid < sampleCount)
            {
                for (int i = 0; i < channels; i++)
                {
                    output[i, tid] = (float)samples[(tid * channels) + i];
                }
                tid += thread.gridDim.x;
            }
        }

        [Cudafy]
        public static void PutSamplesDouble(GThread thread, float[,] samples, double[] output)
        {
            var channels = samples.GetLength(0);
            var sampleCount = samples.GetLength(1);

            int tid = thread.blockIdx.x;
            while (tid < sampleCount)
            {
                for (int i = 0; i < channels; i++)
                {
                    var f = Math.Max(-1.0f, Math.Min(1.0f, samples[i, tid]));
                    output[(tid * channels) + i] = f;
                }
                tid += thread.gridDim.x;
            }
        }

        private static CudafyModule s_KernelModule;

        public static CudafyModule KernelModule
        {
            get
            {
                return s_KernelModule ??
                       (s_KernelModule = CudafyTranslator.Cudafy(eArchitecture.OpenCL, typeof(AudioKernels)));
            }
        }
    }
}
