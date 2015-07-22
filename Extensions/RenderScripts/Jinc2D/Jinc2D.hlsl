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

// TODO: Scale LUMA only

sampler s0    : register(s0);
sampler s1    : register(s1);
sampler s2    : register(s2);
sampler s3    : register(s3);
sampler s4    : register(s4);
float4  p0    : register(c0);
float2  p1    : register(c1);
float4  size0 : register(c2);
float4  args0 : register(c3);

#define width  (p0[0])
#define height (p0[1])

#define inputTexelSize size0.zw
#define antiRingingStrength args0[0]

#define px (p1[0])
#define py (p1[1])

#define Get(x,y)              (tex2D(s0, pos + inputTexelSize*int2(x,y)).rgb)
#define Weights1(offset)      (tex2D(s1,  offset))
#define Weights2(offset)      (tex2D(s2,  offset))
#define Weights3(offset)      (tex2D(s3,  offset))
#define Weights4(offset)      (tex2D(s4,  offset))

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
    
    float4 ws1[4];
    float4 ws2[4];
    float4 ws3[4];
    float4 ws4[4];
    ws1[0] = Weights1(offset);
    ws1[1] = Weights2(offset);
    ws1[2] = Weights3(offset);
    ws1[3] = Weights4(offset);
    ws2[0] = Weights1(float2(1-offset.x, offset.y));
    ws2[1] = Weights2(float2(1-offset.x, offset.y));
    ws2[2] = Weights3(float2(1-offset.x, offset.y));
    ws2[3] = Weights4(float2(1-offset.x, offset.y));
    ws3[0] = Weights1(float2(offset.x, 1-offset.y));
    ws3[1] = Weights2(float2(offset.x, 1-offset.y));
    ws3[2] = Weights3(float2(offset.x, 1-offset.y));
    ws3[3] = Weights4(float2(offset.x, 1-offset.y));
    ws4[0] = Weights1(1-offset);
    ws4[1] = Weights2(1-offset);
    ws4[2] = Weights3(1-offset);
    ws4[3] = Weights4(1-offset);

    {
        {
#if LOOP==1
            [loop]
#else
            [unroll]
#endif
            for (int Y = -LOBES+1; Y<=0; Y++)
            [unroll] for (int X = -LOBES+1; X<=LOBES; X++)
            {
                int2 XY = {X,Y};
                float w;
                if (X <= 0)
                {
                    w = ws1[-Y][-X];
                }
                else
                {
                    w = ws2[-Y][X-1];
                }
                avg += Get(X, Y)*w;
                W += w;
            }
        }
        
        {
#if LOOP==1
            [loop]
#else
            [unroll]
#endif
            for (int Y = 1; Y<=LOBES; Y++)
            [unroll] for (int X = -LOBES+1; X<=LOBES; X++)
            {
                int2 XY = {X,Y};
                float w;
                if (X <= 0)
                {
                    w = ws3[Y-1][-X];
                }
                else
                {
                    w = ws4[Y-1][X-1];
                }
                avg += Get(X, Y)*w;
                W += w;
            }
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
    color = lerp(original, color, antiRingingStrength);
#endif
    return color;
}
