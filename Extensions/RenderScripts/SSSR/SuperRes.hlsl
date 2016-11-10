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

// -- Misc --
sampler s0          : register(s0);
sampler sDiff       : register(s1);
sampler sOriginal   : register(s2);

float4 size1      : register(c2); // Original size
float4 sizeOutput : register(c3);

// -- Size handling --
#define originalSize size1

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
#define noise (0.5/255.0)

// -- Colour space Processing --
#include "../Common/ColourProcessing.hlsl"

// -- Input processing --
//Current high res value
#define Get(x,y)        (tex2Dlod(s0,        float4(tex + sqrt(ddxddy/dxdy)*dxdy*int2(x,y),0,0)))
#define GetLoRes(x,y)   (tex2Dlod(sOriginal, float4(ddxddy*(pos+int2(x,y)+0.5),0,0)))
//Downsampled result
#define Diff(x,y)       (tex2Dlod(sDiff,     float4(ddxddy*(pos+int2(x,y)+0.5),0,0)))

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
   
    [unroll] for (int X = minX; X <= maxX; X++)
    [unroll] for (int Y = minX; Y <= maxX; Y++)
    {
        float R = Diff(X,Y).w;
        float Var = GetLoRes(X,Y).w;

        float2 kernel = Kernel(float2(X,Y) - offset);
        float weight = kernel.x * kernel.y / (sqr(Luma(c0 - GetLoRes(X,Y))) + Var + sqr(noise));

        diff += weight * (Diff(X,Y) - (1-R) * c0.xyz);
        weightSum += weight;
    }
    diff /= weightSum;
    
    c0.xyz += diff;

    #ifdef FinalPass
        #ifdef LinearLight
        c0.xyz = Gamma(c0.xyz);
        #endif
        // c0.xyz = Diff(0,0).w * 0.5; // Debugging
    #endif

    return c0;
}