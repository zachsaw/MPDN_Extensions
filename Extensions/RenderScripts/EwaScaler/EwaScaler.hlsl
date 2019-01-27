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

#ifdef CHROMA
sampler s5    : register(s5);
#endif

float4  sizeOutput : register(c0);
float4  size0 : register(c1);
float4  size1 : register(c2);
float4  args0 : register(c3);

#if LOBES>2
    #define LOOP 1
#else
    #define LOOP 0
#endif

#define imageTexelSize          (size0.zw)
#define chromaTexelSize         (size1.zw)

#ifdef CHROMA
    #define inputTexelSize      chromaTexelSize
    #define texelOffset         (args0.yz)
#else
    #define inputTexelSize      imageTexelSize
    #define texelOffset         (0.5f)
#endif

#define antiRingingStrength     (args0[0])

#ifdef CHROMA
    #define color_t            float2
    #define Get(x,y)           ((tex2D(s1, pos + inputTexelSize*int2((x),(y)))).yz)
    #define GetResult(c)       (float4(tex2D(s0, tex)[0], c, 1))
#else
    #define color_t            float3
    #define Get(x,y)           ((tex2D(s0, pos + inputTexelSize*int2((x),(y)))).rgb)
    #define GetResult(c)       (float4(c, 1))
#endif

#ifdef CHROMA
    #define Weights1(x,y)      (tex2D(s2, float2(x,y)))
    #define Weights2(x,y)      (tex2D(s3, float2(x,y)))
    #define Weights3(x,y)      (tex2D(s4, float2(x,y)))
    #define Weights4(x,y)      (tex2D(s5, float2(x,y)))
#else
    #define Weights1(x,y)      (tex2D(s1, float2(x,y)))
    #define Weights2(x,y)      (tex2D(s2, float2(x,y)))
    #define Weights3(x,y)      (tex2D(s3, float2(x,y)))
    #define Weights4(x,y)      (tex2D(s4, float2(x,y)))
#endif

color_t ApplyAntiRinging(float2 pos, color_t color);

float4 main(float2 tex : TEXCOORD0) : COLOR
{
    // Calculate position
    float2 pos = (tex / inputTexelSize) - texelOffset;
    float2 offset = frac(pos);
    float2 texelTopLeft = pos - offset;
    pos = (texelTopLeft + 0.5f) * inputTexelSize;
    
    color_t avg = 0;
    float W = 0;
    
    float4 ws1[4];
    float4 ws2[4];
    float4 ws3[4];
    float4 ws4[4];
    float offsetmx = 1-offset.x;
    float offsetmy = 1-offset.y;
    ws1[0] = Weights1(offset.x, offset.y);
    ws1[1] = Weights2(offset.x, offset.y);
    ws1[2] = Weights3(offset.x, offset.y);
    ws1[3] = Weights4(offset.x, offset.y);
    ws2[0] = Weights1(offsetmx, offset.y);
    ws2[1] = Weights2(offsetmx, offset.y);
    ws2[2] = Weights3(offsetmx, offset.y);
    ws2[3] = Weights4(offsetmx, offset.y);
    ws3[0] = Weights1(offset.x, offsetmy);
    ws3[1] = Weights2(offset.x, offsetmy);
    ws3[2] = Weights3(offset.x, offsetmy);
    ws3[3] = Weights4(offset.x, offsetmy);
    ws4[0] = Weights1(offsetmx, offsetmy);
    ws4[1] = Weights2(offsetmx, offsetmy);
    ws4[2] = Weights3(offsetmx, offsetmy);
    ws4[3] = Weights4(offsetmx, offsetmy);

    {
        {
#if LOOP==1
            [loop]
#else
            [unroll]
#endif
            for (int Y = -LOBES+1; Y<=0; Y++)
            for (int X = -LOBES+1; X<=LOBES; X++)
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
            for (int X = -LOBES+1; X<=LOBES; X++)
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

    return GetResult(ApplyAntiRinging(pos, avg/W));
}

color_t ApplyAntiRinging(float2 pos, color_t color)
{
#if AR==1
    color_t sampleMin = 1e+8;
    color_t sampleMax = 1e-8;
    
    {
        [unroll] for (int Y = 0; Y<=1; Y++)
        [unroll] for (int X = 0; X<=1; X++)
        {
            color_t c = Get(X, Y);
            sampleMin = min(sampleMin, c);
            sampleMax = max(sampleMax, c);
        }
    }
    
    // Anti-ringing
    color_t original = color;
    color = clamp(color, sampleMin, sampleMax);
    color = lerp(original, color, antiRingingStrength);
#endif
    return color;
}
