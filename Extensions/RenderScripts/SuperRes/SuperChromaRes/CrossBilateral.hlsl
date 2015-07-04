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
#define acuity 100.0
#define radius 0.66
#define power 0.5

// -- Misc --
sampler s0 : register(s0);
sampler sU : register(s1);
sampler sV : register(s2);

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
#define GetY(x,y)      (tex2D(s0,ddxddy*(pos+chromaOffset+int2(x,y)+0.5))[0])
//Low res values
#define GetUV(x,y)    (float2(tex2D(sU,ddxddy*(pos+int2(x,y)+0.5))[0], tex2D(sV,ddxddy*(pos+int2(x,y)+0.5))[0]))

// -- Colour space Processing --
#define Kb args0[2]
#define Kr args0[3]
#include "../../Common/ColourProcessing.hlsl"

// -- Main Code --
float4 main(float2 tex : TEXCOORD0) : COLOR{
    float4 c0 = tex2D(s0, tex);
    float y = c0.x;

    // Calculate position
    float2 pos = tex * chromaSize.xy - chromaOffset - 0.5;
    float2 offset = pos - round(pos);
    pos -= offset;

    // Calculate mean
    float weightSum = 0;
    float2 meanUV = 0;

    [unroll] for (int X = -1; X <= 1; X++)
    [unroll] for (int Y = -1; Y <= 1; Y++)
    {
        float dI2 = sqr(acuity*(y - GetY(X,Y)));
        float dXY2 = sqr(float2(X,Y) - offset);
        float weight = exp(-dXY2 / (2 * radius * radius)) * pow(1 + dI2 / power, -power);
        
        meanUV += weight*GetUV(X,Y);
        weightSum += weight;
    }
    meanUV /= weightSum;

    // Update c0
    c0.gb = meanUV;
    
    return c0;
}