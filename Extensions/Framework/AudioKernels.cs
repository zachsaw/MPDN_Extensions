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
using System.Runtime.InteropServices;
using Cudafy;

namespace Mpdn.Extensions.Framework
{
    public static class AudioKernels
    {
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
        private static float Clip(float val)
        {
            return GMath.Max(-1.0f, GMath.Min(1.0f, val));
        }

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
                    var f = Clip(samples[i, tid]);
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
                    var f = Clip(samples[i, tid]);
                    output[(tid * channels) + i] = (short)((f + 1.0f) * mid + min);
                }
                tid += thread.gridDim.x;
            }
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
                    var f = Clip(samples[i, tid]);
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
                    var f = Clip(samples[i, tid]);
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
                    var f = Clip(samples[i, tid]);
                    output[(tid * channels) + i] = f;
                }
                tid += thread.gridDim.x;
            }
        }

        [Cudafy]
        private static unsafe float ConvertDoubleToFloat(ulong d)
        {
            ulong sign;
            ulong exponent;
            ulong mantissa;

            // IEEE binary64 format
            sign = (d >> 63) & 1; // 1
            exponent = (d >> 52) & 0x7FF; // 11
            mantissa = d & 0x000FFFFFFFFFFFFFul; // 52
            exponent -= 1023;

            // IEEE binary32 format
            exponent += 127; // rebase
            exponent &= 0xFF;
            mantissa >>= (52 - 23); // left justify

            var result = (uint) (mantissa | (exponent << 23) | (sign << 31));
            return *(float*) &result;
        }

        [Cudafy]
        private static unsafe ulong ConvertFloatToDouble(float f)
        {
            uint d = *(uint*) &f;

            ulong sign;
            ulong exponent;
            ulong mantissa;

            // IEEE binary32 format
            sign = (d >> 31) & 1; // 1
            exponent = (d >> 23) & 0xFF; // 8
            mantissa = d & 0x7FFFFF; // 23
            exponent += 1023;

            // IEEE binary64 format
            exponent -= 127; // rebase
            exponent &= 0x7FF;
            mantissa <<= (52 - 23); // right justify

            return mantissa | (exponent << 52) | (sign << 63);
        }

        [Cudafy]
        public static void GetSamplesDouble(GThread thread, ulong[] samples, float[,] output)
        {
            // Warning: Untested (LAV Audio Decoder doesn't support output of double format)

            var channels = output.GetLength(0);
            var sampleCount = output.GetLength(1);

            int tid = thread.blockIdx.x;
            while (tid < sampleCount)
            {
                for (int i = 0; i < channels; i++)
                {
                    output[i, tid] = ConvertDoubleToFloat(samples[(tid * channels) + i]);
                }
                tid += thread.gridDim.x;
            }
        }

        [Cudafy]
        public static void PutSamplesDouble(GThread thread, float[,] samples, ulong[] output)
        {
            // Warning: Untested (LAV Audio Decoder doesn't support output of double format)

            var channels = samples.GetLength(0);
            var sampleCount = samples.GetLength(1);

            int tid = thread.blockIdx.x;
            while (tid < sampleCount)
            {
                for (int i = 0; i < channels; i++)
                {
                    var f = Clip(samples[i, tid]);
                    output[(tid * channels) + i] = ConvertFloatToDouble(f);
                }
                tid += thread.gridDim.x;
            }
        }
    }
}
