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

// Ported to SM5.0 by Shiandow for use in MPDN
// with further modifications and optimizations by Zach Saw

//#define EXTRA_CHECKS

#define WT 8
#define HT 4

// Declarations

Texture2D inputTexture : register(t0);

cbuffer args : register(b0)
{
    float outWidth;
    float outHeight;
    float counter;
    float timeStamp;
};

cbuffer size0 : register(b1)
{
    float width;
    float height;
    float ppx;
    float ppy;
};

#ifdef BUF_STRUCTURED

struct Weights
{
    float4 val[WT];
};

StructuredBuffer<Weights> w1 : register(t1);
StructuredBuffer<Weights> w2 : register(t2);
StructuredBuffer<float2>  w  : register(t3);

#define W1(nns, wt) w1[nns].val[wt]
#define W2(nns, wt) w2[nns].val[wt]
#define WS(nns)     w[nns]

#else

cbuffer weights1 : register(b2)
{
    float4 w1[nns][WT];
}

cbuffer weights2 : register(b3)
{
    float4 w2[nns][WT];
}

cbuffer weights3 : register(b4)
{
    float2 w[nns];
}

#define W1(nns, wt) w1[nns][wt]
#define W2(nns, wt) w2[nns][wt]
#define WS(nns)     w[nns]

#endif

SamplerState ss;

struct PS_IN
{
    float4 Position   : SV_POSITION;
    float2 Texture    : TEXCOORD0;
};

/* Handle Input */
#define Get_(x,y) (inputTexture.Sample(ss,tex+float2(ppx*(x),ppy*(y))))

#ifdef CHROMA_U
#define Get(x,y) (Get_(x,y)[1])
#define GetResult(x) (float4(1,x,1,1))
#else
#ifdef CHROMA_V
#define Get(x,y) (Get_(x,y)[2])
#define GetResult(x) (float4(1,1,x,1))
#else
#define Get(x,y) (Get_(x,y)[0])
#define GetResult(x) (float4(x,1,1,1))
#endif
#endif

#define EPSILON_F32 1.19209290e-07

/* Main code */
float4 main( PS_IN In ) : SV_TARGET
{
    float2 tex = In.Texture;

    float4 t[WT];
    
    float sum = 0;
    float sumsq = 0;
    [unroll] for (int i = 0; i<WT; i++)
#ifdef VECTOR_DOT
    {
        [unroll] for (int j = 0; j<HT; j++)
            t[i][j] = Get(i-3, j-1);
        float4 pix = t[i];
        sum += dot(pix, 1);
        sumsq += dot(pix, pix);
    }
#else
    [unroll] for (int j = 0; j<HT; j++)
    {
        float pix = t[i][j] = Get(i-3, j-1);
        sum += pix;
        sumsq += pix*pix;
    }
#endif
    
    float4 mstd = 0;
    mstd[0] = sum / (WT*HT);
    mstd[1] = sumsq / (WT*HT) - mstd[0] * mstd[0];
    mstd[1] = (mstd[1] <= EPSILON_FL32) ? 0.0 : sqrt(mstd[1]);
    mstd[2] = (mstd[1] > 0) ? (1.0 / mstd[1]) : 0.0;

    float vsum = 0;
    float wsum = 0;
#ifdef UNROLLED
    [unroll]
#else
    [loop] [fastopt]
#endif
    for (int n = 0; n<nns; n++)
    {
        float2 sum = 0;
#ifdef LOOP_INNER
        [loop] [fastopt]
#else
        [unroll]
#endif
        for (int i = 0; i<WT; i++)
        {
#ifdef VECTOR_DOT
            float4 pix = t[i];
            sum[0] += dot(pix, W1(n, i));
            sum[1] += dot(pix, W2(n, i));
#else
            [unroll] for (int j = 0; j<HT; j++)
            {
                float pix = t[i][j];
                sum[0] += pix*W1(n, i)[j];
                sum[1] += pix*W2(n, i)[j];
            }
#endif
        }
        sum[0] = sum[0]*mstd[2] + WS(n)[0];
        sum[1] = sum[1]*mstd[2] + WS(n)[1];
#ifdef EXTRA_CHECKS
        sum[0] = exp(clamp(sum[0], -80.0, 80.0));
#else
        sum[0] = exp(sum[0]);
#endif
        vsum += sum[0]*(sum[1]/(1+abs(sum[1])));
        wsum += sum[0];
    }

#ifdef EXTRA_CHECKS
    return GetResult(saturate(mstd[0] + (wsum > 1e-10 ? (5*vsum/wsum)*mstd[1] : 0.0)));
#else
    return GetResult(saturate(mstd[0] + (5*vsum/wsum)*mstd[1]));
#endif
}