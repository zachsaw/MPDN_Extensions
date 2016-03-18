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
sampler s0 : register(s0);
sampler sUV : register(s1);

float4 p0     : register(c0);
float2 p1     : register(c1);
float4 size1  : register(c2);
float4 chromaParams : register(c3);
float  power  : register(c4);

// -- Convenience --
#define sqr(x) dot(x,x)
#define noise 1.0/(2.0*255.0)

#define width  (p0[0])
#define height (p0[1])
#define chromaSize size1

#define dxdy (p1.xy)
#define ddxddy (chromaSize.zw)
#define chromaOffset (chromaParams.xy)
#define radius 0.66

// -- Window Size --
#define taps 4
#define even (taps - 2 * (taps / 2) == 0)
#define minX (1-ceil(taps/2.0))
#define maxX (floor(taps/2.0))

#define factor (dxdy/ddxddy)
#define Kernel(x) saturate(0.5 + (0.5 - abs(x)) * 2)

// -- Input processing --
// Luma value
#define GetLuma(x,y)   tex2D(sUV,tex + dxdy*int2(x,y))[0]
// Chroma value
#define GetChroma(x,y) tex2D(sUV,ddxddy*(pos+int2(x,y)+0.5))

// -- Main Code --
float4 main(float2 tex : TEXCOORD0) : COLOR{
    float4 c0 = tex2D(s0, tex);

    // Calculate position
    float2 pos = tex * chromaSize.xy - chromaOffset - 0.5;
    float2 offset = pos - floor(pos);
    pos -= offset;

    // Linear interpolation of chroma
    float4 interp = 
#if even
    lerp(
        lerp(GetChroma(0,0), GetChroma(1,0), offset.x),
        lerp(GetChroma(0,1), GetChroma(1,1), offset.x), offset.y);
#else
    // TODO: unroll all cases to avoid extra texture lookups (not needed with even #taps)
    lerp(
        lerp(GetChroma(0,0), GetChroma(sign(offset.x),0), abs(offset.x)),
        lerp(GetChroma(0,sign(offset.y)), GetChroma(sign(offset).x,sign(offset).y), offset.x), abs(offset.y));
#endif

    // Estimate local variance
    float localVar = interp.w + sqr(interp.x - c0.x);

    // Bilateral weighted interpolation
    float4 mean = 0;
    float3 mean2 = 0;
    float2 meanYUV = 0;
    [unroll] for (int X = minX; X <= maxX; X++)
    [unroll] for (int Y = minX; Y <= maxX; Y++)
    {
        float dI2 = sqr(GetChroma(X,Y).x - c0.x);
        float var = GetChroma(X,Y).w + sqr(noise);
        float dXY2 = sqr((float2(X,Y) - offset)/radius);

        float weight = exp(-0.5*dXY2) / (dI2 + var + localVar);
        
        mean += weight*float4(GetChroma(X,Y).xyz,1);
        mean2 += weight*(var + GetChroma(X,Y)*GetChroma(X,Y));
        meanYUV += weight*GetChroma(X,Y).x*GetChroma(X,Y).yz;
    }
    float weightSum = mean.w;

    // Linear fit
    mean /= weightSum;
    float3 Var = (mean2 / weightSum) - mean*mean;
    float2 Cov = (meanYUV / weightSum) - mean.x*mean.yz;

    // Dampen bit level noise.
    Var.x += sqr(noise);

    // Estimate error
    float2 n = weightSum * (sqr(c0.x - mean.x) + Var.x + localVar);
    float2 R2 = saturate((Cov * Cov) / (Var.x * Var.yz));
    float2 err = (1-R2) * (sqr((c0 - mean).x) / Var.x);

    // Balance error of interpolation with error of fit.
    float2 strength = power / lerp(err, 1, power);
    
    // Show strength (debugging)
    // return float4(dot(strength, 1/2.0), 0.5, 0.5, 1);

    // Update c0
    c0.yz = mean.yz + ((c0 - mean).x * Cov / Var.x);

    // Fall back to linear interpolation if necessary
    c0.yz = lerp(interp.yz, c0.yz, strength);

    return c0;
}