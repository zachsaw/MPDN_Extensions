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

// -- Edge detection options -- 
#define acuity 6.0
#define radius 0.66
#define power 1.0

// -- Misc --
sampler s0 	  : register(s0);
sampler sDiff : register(s1);

float4 p0	  : register(c0);
float2 p1	  : register(c1);
float4 size1  : register(c2);
float4 args0  : register(c3);
float4 args1  : register(c4);

#define width  (p0[0])
#define height (p0[1])
#define chromaSize size1

#define dxdy (p1.xy)
#define ddxddy (chromaSize.zw)
#define chromaOffset (args1.xy)

#define sqr(x) dot(x,x)

// -- Colour space Processing --
#define Kb args0[2]
#define Kr args0[3]
#include "../../Common/ColourProcessing.hlsl"

// -- Input processing --
//Current high res value
#define Get(x,y)      (tex2D(s0,ddxddy*(pos+chromaOffset+int2(x,y)+0.5)).xyz)
#define GetY(x,y)      (tex2D(sDiff,ddxddy*(pos+int2(x,y)+0.5)).a)
//Downsampled result
#define Diff(x,y)     (tex2D(sDiff,ddxddy*(pos+int2(x,y)+0.5)))

// -- Main Code --
float4 main(float2 tex : TEXCOORD0) : COLOR{
    float4 c0 = tex2D(s0, tex);

    // Calculate position
    float2 pos = tex * chromaSize.xy - chromaOffset - 0.5;
    float2 offset = pos - round(pos);
    pos -= offset;

    // Calculate faithfulness force
    float weightSum = 0;
    float4 diff = 0;

    [unroll] for (int X = -1; X <= 1; X++)
    [unroll] for (int Y = -1; Y <= 1; Y++)
    {
        float dI2 = sqr(acuity*(Luma(c0.rgb) - GetY(X,Y)));
        float dXY2 = sqr(float2(X,Y) - offset);
        float weight = exp(-dXY2 / (2 * radius * radius)) * pow(1 + dI2 / power, -power);

        diff += weight*Diff(X,Y);
        weightSum += weight;
    }
    diff /= weightSum;

    // Apply force
    c0.yz -= strength*diff;
    
    return c0;
}