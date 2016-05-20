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
// -- Main parameters --
#define strength (args0[0])
#define softness (args0[1])

// -- Misc --
sampler s0    : register(s0);
sampler sDiff : register(s1);
sampler sOriginal : register(s2);
float4 size1  : register(c2); // Original size
float4 sizeOutput : register(c3);
float4 args0  : register(c4);

// -- Edge detection options -- 
#define acuity 6.0
#define radius 0.5
#define power 1.0

// -- Skip threshold --
#define threshold 1
#define skip (1 == 0)
// #define skip (c0.a < threshold/255.0)

// -- Size handling --
#define originalSize size1

#define width  (sizeOutput[0])
#define height (sizeOutput[1])

#define dxdy (sizeOutput.zw)
#define ddxddy (originalSize.zw)

// -- Window Size --
#define taps 4
#define even (taps - 2 * (taps / 2) == 0)
#define minX (1-ceil(taps/2.0))
#define maxX (floor(taps/2.0))

#define factor (ddxddy/dxdy)
#define pi acos(-1)
#define Kernel(x) (cos(pi*(x)/taps)) // Hann kernel

// -- Convenience --
#define sqr(x) dot(x,x)

// -- Colour space Processing --
#include "../Common/ColourProcessing.hlsl"

// -- Input processing --
//Current high res value
#define Get(x,y)        (tex2Dlod(s0, float4(tex + sqrt(ddxddy/dxdy)*dxdy*int2(x,y),0,0)))
// #define GetY(x,y)   (tex2Dlod(sDiff,float4(ddxddy*(pos+int2(x,y)+0.5),              0,0)).a)
#define GetOriginal(x,y)   (tex2Dlod(sOriginal, float4(ddxddy*(pos+int2(x,y)+0.5),0,0)))
//Downsampled result
#define Diff(x,y)       (tex2Dlod(sDiff,  float4(ddxddy*(pos+int2(x,y)+0.5),0,0)))

// -- Main Code --
float4 main(float2 tex : TEXCOORD0) : COLOR{
    float4 c0 = tex2D(s0, tex);
    float4 Lin = c0;

    // Calculate position
    float2 pos = tex * originalSize.xy - 0.5;
    float2 offset = pos - (even ? floor(pos) : round(pos));
    pos -= offset;

    // Calculate faithfulness force
    float weightSum = 0;
    float3 diff = 0;
    float3 soft = 0;
   
    [unroll] for (int X = minX; X <= maxX; X++)
    [unroll] for (int Y = minX; Y <= maxX; Y++)
    {
        float2 kernel = Kernel(float2(X,Y) - offset);
        float weight = kernel.x * kernel.y / (sqr(Luma(c0 - GetOriginal(X,Y))) + GetOriginal(X,Y).w + sqr(0.5/255.0));
        // float weight = kernel.x * kernel.y * (- log(sqr(Luma(c0 - GetOriginal(X,Y))) + GetOriginal(X,Y).w + sqr(0.5/255.0)));

        diff += weight * ((1-Diff(X,Y).w)*c0 - Diff(X,Y));
        weightSum += weight;
    }
    diff /= weightSum;

    c0.xyz -= strength * diff;

    #ifdef FinalPass
        c0.xyz = Gamma(2*(c0.xyz - 0.25));
    #endif

    return c0;
}