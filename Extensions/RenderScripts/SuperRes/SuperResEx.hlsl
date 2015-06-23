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
#ifndef strength
    #define strength (args0[0])
#endif
#ifndef softness
    #define softness (args0[1])
#endif

// -- Misc --
sampler s0    : register(s0);
sampler sDiff : register(s1);
float4 p0      : register(c0);
float2 p1      : register(c1);
float4 size1  : register(c2); // Original size
float4 args0  : register(c3);

// -- Edge detection options -- 
#define acuity 6.0
#define radius 0.66
#define power 3.0

#define originalSize size1

#define width  (p0[0])
#define height (p0[1])

#define dxdy (p1.xy)
#define ddxddy (originalSize.zw)

#define sqr(x) dot(x,x)

// -- Colour space Processing --
#include "../Common/ColourProcessing.hlsl"

// -- Input processing --
//Current high res value
// #define Get(x,y)      (tex2D(s0,tex+dxdy*int2(x,y)).xyz)
#define Get(x,y)    (tex2D(s0,ddxddy*(pos+int2(x,y)+0.5)).xyz)
//Downsampled result
#define Diff(x,y)     (tex2D(sDiff,ddxddy*(pos+int2(x,y)+0.5)).xyz)

// -- Main Code --
float4 main(float2 tex : TEXCOORD0) : COLOR{
    float4 c0 = tex2D(s0, tex);

    // Calculate position
    float2 pos = tex * originalSize.xy - 0.5;
    float2 offset = pos - round(pos);
    pos -= offset;

    // Calculate faithfulness force
    float W = 0;
    float3 diff = 0;
    float3 stab = 0;
    float var = 0;

    [unroll] for (int X = -1; X <= 1; X++)
	[unroll] for (int Y = -1; Y <= 1; Y++)
    {
		float dI2 = sqr(acuity*(c0.rgb - Get(X,Y)));
        float dXY2 = sqr(float2(X,Y) - offset);
        float w = exp(-dXY2/(2*radius*radius))*pow(1 + dI2/power, - power);

        diff += w*Diff(X,Y);
        stab += w*(c0.rgb - Get(X,Y));
        var += w*dI2;
        W += w;
    }
    diff /= W;
    stab /= W;
    var = (var / W) - sqr(acuity*stab);

    // Calculate edge statistics
    float varD = softness * sqr(acuity*stab);
    float varS = (1 - softness) * var;

    // Apply force
    c0.xyz -= strength*lerp(diff, stab, softness);

    return c0;
}