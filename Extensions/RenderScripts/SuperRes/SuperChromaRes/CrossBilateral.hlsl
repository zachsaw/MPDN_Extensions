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
// -- Edge detection options -- 
#define acuity 12.0
#define radius 0.66
#define power 3.0

// -- Misc --
sampler s0 	  : register(s0);
sampler sUV	  : register(s1);

float4 p0	  : register(c0);
float2 p1	  : register(c1);
float4 size1  : register(c2);
float4 args0  : register(c3);

#define width  (p0[0])
#define height (p0[1])
#define chromaSize size1

#define dxdy (p1.xy)
#define ddxddy (chromaSize.zw)
#define chromaOffset (args0.xy)

#define sqr(x) dot(x,x)

// -- Input processing --
//Current high res value
#define Get(x,y)      (tex2D(s0,ddxddy*(pos+chromaOffset+int2(x,y)+0.5)).xyz)
//Low res values
#define LoRes(x,y)    (tex2D(sUV,ddxddy*(pos+int2(x,y)+0.5)).xyz)

// -- Colour space Processing --
#include "../../Common/ColourProcessing.hlsl"
#define Kb args0[2] //redefinition
#define Kr args0[3] //redefinition

// -- Main Code --
float4 main(float2 tex : TEXCOORD0) : COLOR{
    float4 c0 = tex2D(s0, tex);

    // Calculate position
    float2 pos = tex * chromaSize.xy - chromaOffset - 0.5;
    float2 offset = pos - round(pos);
    pos -= offset;

    // Calculate mean
    float W = 0;
    float3 mean = 0;

    [unroll] for (int X = -1; X <= 1; X++)
    [unroll] for (int Y = -1; Y <= 1; Y++)
    {
        float dI2 = sqr(acuity*(c0.rgb - Get(X,Y))[0]);
        float dXY2 = sqr(float2(X,Y) - offset);
        float w = exp(-dXY2/(2*radius*radius))*pow(1 + dI2/power, - power);

        mean += w*LoRes(X,Y);
        W += w;
    }
    mean /= W;

    // Update c0
    c0.gb = mean.gb;
    
    return c0;
}