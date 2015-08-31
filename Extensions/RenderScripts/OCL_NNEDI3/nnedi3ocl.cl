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

// The original kernel was created by SEt:
// http://forum.doom9.org/showthread.php?t=169766

// modifications by madshi (for use in madVR):
// (1) use image objects instead of buffers
// (2) hard coded 8x4 instead of 8x6
// (3) only one kernel for both x and y upscaling
// (4) padding + mirroring built into the main kernel
// (5) flexible image channel handling

// further modded and optimized by Zachs for use in MPDN

//#define EXTRA_CHECKS

constant sampler_t Sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#if !defined(cl_nv_pragma_unroll) && (__OPENCL_VERSION__ >= 110)
#define FASTLOAD
#endif

#define LOAD_INPUTS(t) \
    float16 _t = *(local float16*) in[j]; \
    float8 t[8] = { _t.s01234567, _t.s12345678, _t.s23456789, _t.s3456789A, \
                    _t.s456789AB, _t.s56789ABC, _t.s6789ABCD, _t.s789ABCDE };                                  

#define LOAD_WEIGHTS(w1, w2) \
    float _w[16]; float w1[8]; float w2[8]; \
    *((float16*) _w) = *(__global const float16*) weights; \
    *((float8*) w1) = *(float8*) _w; *((float8*) w2) = *(((float8*) _w)+1); \

float8 nnedi3process(__local float (* restrict in)[74], __global const float* restrict weights, uint nnst)
{
    float8 sum = 0, sumsq = 0;
    
    for (uint j = 0; j < 4; j++)
    {
#ifdef FASTLOAD
        LOAD_INPUTS(p)
        #pragma unroll
        for (uint i = 0; i < 8; i++)
        {
            float8 pix = p[i];
            sum += pix;
            sumsq += pix*pix;
        }
#else
        float8 pix = (float8)(0, in[j][0], in[j][1], in[j][2], in[j][3], in[j][4], in[j][5], in[j][6]);
        #pragma unroll
        for (uint i = 0; i < 8; i++)
        {
            pix = (float8)(pix.s1234567, in[j][i + 7]);
            sum += pix;
            sumsq += pix*pix;
        }
#endif
    }
    
    float8 mstd0, mstd1, mstd2, mstd3 = 0;
    mstd0 = sum / 32.0f;
    mstd1 = sumsq / 32.0f - mstd0 * mstd0;
    mstd2 = mstd1 > FLT_EPSILON ? rsqrt(mstd1) : 0;
    mstd1 = mstd1 * mstd2;

    float8 vsum = 0, wsum = 0;
    for (uint k = 0; k < nnst; k++)
    {
        float8 sum1 = 0;
        float8 sum2 = 0;
        #pragma unroll 2
        for (uint j = 0; j < 4; j++)
        {
#ifdef FASTLOAD
            LOAD_INPUTS(p)
            LOAD_WEIGHTS(w1, w2)
            weights += 16;
            #pragma unroll
            for (uint i = 0; i < 8; i++)
            {
                float8 pix = p[i];
                sum1 += pix * w1[i];
                sum2 += pix * w2[i];
            }
#else
            float w[16];
            *((float16*) w) = *(__global const float16*) weights;
            weights += 16;
            float8 pix = (float8)(0, in[j][0], in[j][1], in[j][2], in[j][3], in[j][4], in[j][5], in[j][6]);
            #pragma unroll
            for (uint i = 0; i < 8; i++)
            {
                pix = (float8) (pix.s1234567, in[j][i + 7]);
                sum1 += pix * w[i];
                sum2 += pix * w[i + 8];
            }
#endif
        }
        float2 w = *(__global const float2*) weights;
        weights += 4;
#ifdef EXTRA_CHECKS
        sum1 = exp(clamp(sum1 * mstd2 + w.s0, -80.0f, +80.0f));
#else
        sum1 = exp(sum1 * mstd2 + w.s0);
#endif
        sum2 = sum2 * mstd2 + w.s1;
        wsum += sum1;
        vsum += sum1 * (sum2 / (1.0f + fabs(sum2)));
    }
#ifdef EXTRA_CHECKS
    mstd3 += wsum > 1e-10f ? 5.0f * vsum / wsum * mstd1 + mstd0 : mstd0;
#else
    mstd3 += mstd0 + 5.0f * vsum / wsum * mstd1;
#endif
    return mstd3;
}

#define OFFSET 0
#define GetIX(x) get_group_id(0) * 64 + (x) - 3
#define GetIY(y) get_group_id(1) * 8 + (y) - 1 - OFFSET
#define GET_(x, y, swapXy) read_imagef(srcImg, Sampler, (swapXy) ? ((int2) (y, x)) : ((int2) (x, y)))

#ifdef CHROMA_U
#define GET(x, y, swapXy) GET_(x, y, swapXy).s1
#else
#ifdef CHROMA_V
#define GET(x, y, swapXy) GET_(x, y, swapXy).s2
#else
#define GET(x, y, swapXy) GET_(x, y, swapXy).s0
#endif
#endif

__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void nnedi3(__read_only image2d_t srcImg, __write_only image2d_t dstImg, 
            __global float* restrict weights, uint nnst,
            uint srcWidth, uint srcHeight, uint swapXy)
{
    __local float input[11][74];
    uint xy = get_local_id(1) * 8 + get_local_id(0);
    if (xy < 63)
    {
        uint x = (xy % 9) * 8;
        uint y = xy / 9;
        for (uint i = x; i < x + 8; i++)
        {
            input[y][i] = GET(GetIX(i), GetIY(y), swapXy);
        }
        if (y < 4)
        {
            for (uint i = x; i < x + 8; i++)
            {
                input[y + 7][i] = GET(GetIX(i), GetIY(y) + 7, swapXy);
            }
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    float8 mstd3 = 
        nnedi3process(
            (__local float(*)[74]) &input[get_local_id(1)][get_local_id(0) * 8], 
            (__global const float *restrict) weights, nnst);

    uint y = get_group_id(1) * 16 + get_local_id(1) * 2;
    if (y >= srcHeight * 2)
        return;

    uint x = get_group_id(0) * 64 + get_local_id(0) * 8;
    for (uint i = 0; i < 8; i++)
    {
        if (x + i >= srcWidth)
            continue;
        
        write_imagef(dstImg, 
                     swapXy 
                        ? (int2) (y +     OFFSET, x + i) 
                        : (int2) (x + i, y +     OFFSET), 
                        input[get_local_id(1) + 1 + OFFSET][get_local_id(0) * 8 + 3 + i]);

        write_imagef(dstImg, 
                     swapXy
                        ? (int2) (y + 1 - OFFSET, x + i) 
                        : (int2) (x + i, y + 1 - OFFSET), 
                        ((float*) &mstd3)[i]);
    }
}
