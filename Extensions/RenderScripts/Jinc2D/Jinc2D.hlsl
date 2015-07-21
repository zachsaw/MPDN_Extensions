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
sampler s0    : register(s0);
sampler s1    : register(s1);
sampler s2    : register(s2);
sampler s3    : register(s3);
sampler s4    : register(s4);
float4  p0    : register(c0);
float2  p1    : register(c1);
float4  size0 : register(c2);

#define width  (p0[0])
#define height (p0[1])

#define inputTexelSize size0.zw

#define px (p1[0])
#define py (p1[1])

#define LOBES 2

#define Get(x,y)              (tex2D(s0, pos + inputTexelSize*int2(x,y)).rgb)
#define Weights1(offset)      (tex2D(s1, offset))
#define Weights2(offset)      (tex2D(s2, offset))
#define Weights3(offset)      (tex2D(s3, offset))
#define Weights4(offset)      (tex2D(s4, offset))

float3 ApplyAntiRinging(float2 pos, float3 color);

float4 main(float2 tex : TEXCOORD0) : COLOR
{
    // Calculate position
    float2 pos = (tex / inputTexelSize) - 0.5f;
    float2 offset = frac(pos);
    float2 texelTopLeft = pos - offset;
    pos = (texelTopLeft + 0.5f) * inputTexelSize;
    
    float3 avg = 0;
    float W = 0;
    
    float4x4 ws;
    ws[0] = Weights1(offset);
    ws[1] = Weights2(offset);
    ws[2] = Weights3(offset);
    ws[3] = Weights4(offset);

    {
        [unroll] for (int Y = -LOBES+1; Y<=LOBES; Y++)
        [unroll] for (int X = -LOBES+1; X<=LOBES; X++)
        {
            int2 XY = {X,Y};
            float w = ws[Y+LOBES-1][X+LOBES-1];
            avg += Get(X, Y)*w;
            W += w;
        }
    }
    
    return float4(ApplyAntiRinging(pos, avg/W), 1);
}

float3 ApplyAntiRinging(float2 pos, float3 color)
{
#if AR==1
    float3 sampleMin = 1e+8;
    float3 sampleMax = 1e-8;
    
    {
        [unroll] for (int Y = 0; Y<=1; Y++)
        [unroll] for (int X = 0; X<=1; X++)
        {
            float3 c = Get(X, Y);
            sampleMin = min(sampleMin, c);
            sampleMax = max(sampleMax, c);
        }
    }
    
    // Anti-ringing
    float3 original = color;
    color = clamp(color, sampleMin, sampleMax);
    color = lerp(original, color, AR_STRENGTH);
#endif
    return color;
}
