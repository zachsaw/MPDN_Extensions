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

//#define EXTRA_CHECKS

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

cbuffer weights1 : register(b2)
{
	float4 w1[nns][8];
}

cbuffer weights2 : register(b3)
{
	float4 w2[nns][8];
}

cbuffer weights3 : register(b4)
{
	float2 w[nns];
}

SamplerState ss;

struct PS_IN
{
	float4 Position   : SV_POSITION;
	float2 Texture    : TEXCOORD0;
};

/* Handle Input */
#define Get(x,y) (inputTexture.Sample(ss,tex+float2((x)/width,(y)/height))[0])

/* Main code */
float4 main( PS_IN In ) : SV_TARGET
{
	float2 tex = In.Texture;

    float4 t[8];
    
	float sum = 0;
	float sumsq = 0;
	[unroll] for (int i = 0; i<8; i++)
	[unroll] for (int j = 0; j<4; j++) {
		float tt = t[i][j] = Get(i-3, j-1);
		sum += tt;
		sumsq += tt*tt;
	}

	float4 mstd = 0;
	mstd[0] = sum / 32.0;
	mstd[1] = sumsq / 32.0 - mstd[0] * mstd[0];
	mstd[1] = (mstd[1] <= 1.19209290e-07) ? 0.0 : sqrt(mstd[1]);
	mstd[2] = (mstd[1] > 0) ? (1.0 / mstd[1]) : 0.0;

	float vsum = 0;
	float wsum = 0;
#if nns == 16
    [unroll]
#else
	[loop] 
#endif

    for (int n = 0; n<nns; n++) {
		float2 sum = {0, 0}; 
#if nns == 128
		[loop] 
#else
        [unroll] 
#endif    
        for (int i = 0; i<8; i++)
        [unroll]
        for (int j = 0; j<4; j++) {
			float tt = t[i][j];
			sum[0] += tt*w1[n][i][j];
			sum[1] += tt*w2[n][i][j];
		}
		sum[0] = sum[0]*mstd[2] + w[n][0];
		sum[1] = sum[1]*mstd[2] + w[n][1];
#ifdef EXTRA_CHECKS
		sum[0] = exp(clamp(sum[0], -80.0, 80.0));
#else
        sum[0] = exp(sum[0]);
#endif
		vsum += sum[0]*(sum[1]/(1+abs(sum[1])));
		wsum += sum[0];
	}

#ifdef EXTRA_CHECKS
	return mstd[0] + (wsum > 1e-10 ? (5*vsum/wsum)*mstd[1] : 0.0);
#else
    return mstd[0] + (5*vsum/wsum)*mstd[1];
#endif
}