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
float4 p0      : register(c0);
float2 p1      : register(c1);
float4 size1  : register(c2); // Original size
float4 args0  : register(c3);

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

#define width  (p0[0])
#define height (p0[1])

#define dxdy (p1.xy)
#define ddxddy (originalSize.zw)

// -- Window Size --
#define taps 4
#define even (taps - 2 * (taps / 2) == 0)
#define minX (1-ceil(taps/2.0))
#define maxX (floor(taps/2.0))

#define factor (ddxddy/dxdy)
#define Kernel(x) saturate((taps*0.5 - abs(x)) * factor)

// -- Convenience --
#define sqr(x) dot(x,x)

// -- Colour space Processing --
#include "../Common/ColourProcessing.hlsl"

// -- Input processing --
//Current high res value
#define Get(x,y)    (tex2Dlod(s0,   float4(tex + sqrt(ddxddy/dxdy)*dxdy*int2(x,y),  0,0)).xyz)
#define GetY(x,y)   (tex2Dlod(sDiff,float4(ddxddy*(pos+int2(x,y)+0.5),              0,0)).a)
//Downsampled result
#define Diff(x,y)   (tex2Dlod(sDiff,float4(ddxddy*(pos+int2(x,y)+0.5),              0,0)).xyz)

// -- Main Code --
float4 main(float2 tex : TEXCOORD0) : COLOR{
    float4 c0 = tex2D(s0, tex);
    float4 Lin = c0;
    c0.xyz = Gamma(c0.xyz);

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
        float dI2 = sqr(acuity*(Luma(c0) - GetY(X,Y)));
        //float dXY2 = sqr((float2(X,Y) - offset)/radius);
        //float weight = exp(-0.5*dXY2) * pow(1 + dI2/power, - power);
        float2 kernel = Kernel(float2(X,Y) - offset);
        float weight = kernel.x * kernel.y * pow(1 + dI2/power, - power);

        diff += weight*Diff(X,Y);
        weightSum += weight;
    }
    diff /= weightSum;

    [branch] if (!skip)
    {
        c0.xyz -= strength * diff;
        // c0.a = length(diff);
    }

#ifndef FinalPass
    // Convert back to linear light;
    c0.xyz = GammaInv(c0.xyz);

    #ifndef SkipSoftening
        weightSum=0;
        #define softAcuity 6.0

        [unroll] for (int X = -1; X <= 1; X++)
        [unroll] for (int Y = -1; Y <= 1; Y++)
        if (X != 0 || Y != 0) {
            float3 dI = Get(X,Y) - Lin;
            float dI2 = sqr(softAcuity*dI);
            float dXY2 = sqr(float2(X,Y)/radius);
            float weight = pow(rsqrt(dXY2 + dI2),3); // Fundamental solution to the 5d Laplace equation
            // float weight = exp(-0.5*dXY2) * pow(1 + dI2/power, - power);

            soft += weight * dI;
            weightSum += weight;
        }
        soft /= weightSum;

        [branch] if (!skip)
            c0.xyz += softness * soft;
    #endif
#else
    c0.a = 1;
#endif

    return c0;
}