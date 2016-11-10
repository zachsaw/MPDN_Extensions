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

float4 size1  : register(c2);
float4 sizeOutput : register(c3);
float4 chromaParams : register(c4);
float  power  : register(c5);

// -- Convenience --
#define sqr(x) dot(x,x)
#define noise 0.05
#define bitnoise 1.0/(2.0*255.0)

#define chromaSize size1

#define dxdy (sizeOutput.zw)
#define ddxddy (chromaSize.zw)
#define chromaOffset (chromaParams.xy)
#define radius 0.5

// -- Window Size --
#define taps 4
#define even (taps - 2 * (taps / 2) == 0)
#define minX (1-ceil(taps/2.0))
#define maxX (floor(taps/2.0))

#define factor (dxdy/ddxddy)
// #define Kernel(x) saturate(0.5 + (0.5 - abs(x)) * 2)
#define pi acos(-1)
#define Kernel(x) (cos(pi*(x)/taps)) // Hann kernel

#define sinc(x) sin(pi*(x))/(x)
#define BCWeights(B,C,x) (x > 2.0 ? 0 : x <= 1.0 ? ((2-1.5*B-C)*x + (-3+2*B+C))*x*x + (1-B/3.) : (((-B/6.-C)*x + (B+5*C))*x + (-2*B-8*C))*x+((4./3.)*B+4*C))
#define IntKernel(x) (BCWeights(1.0/3.0, 1.0/3.0, abs(x)))
// #define IntKernel(x) (cos(0.5*pi*saturate(abs(x))))

#include "../Common/ColourProcessing.hlsl"

// -- Input processing --
// Luma value
#define GetLuma(x,y)   tex2D(sUV, tex + dxdy*float2(x,y))[0]
// Chroma value
#define GetChroma(x,y) tex2D(sUV, ddxddy*(pos+float2(x,y)+0.5))
#define GetChromaXY(xy) GetChroma(xy[0], xy[1])

float4 main(float2 tex : TEXCOORD0) : COLOR{
    float4 c0 = tex2D(s0, tex);

    // Calculate position
    float2 pos = tex * chromaSize.xy - chromaOffset - 0.5;
    float2 offset = pos - floor(pos);
    pos -= offset;

    #define localVar sqr(noise)

    // Bilateral weighted interpolation
    float4 fitAvg = 0;
    float4 fitVar = 0;
    float4 fitCov = 0;
    float4 intAvg = 0;
    float4 intVar = 0;
    for (int X = minX; X <= maxX; X++)
    for (int Y = minX; Y <= maxX; Y++)
    {
        float dI2 = sqr(GetChroma(X,Y).x - c0.x);
        float var = GetChroma(X,Y).w + sqr(bitnoise);
        float dXY2 = sqr((float2(X,Y) - offset)/radius);

        float2 kernel = Kernel(float2(X,Y) - offset);
        float weight = kernel.x * kernel.y / (dI2 + var + localVar);
        // float weight = - kernel.x * kernel.y * log(dI2 + var + localVar);
        // float weight = kernel.x * kernel.y * exp(-0.5*(dI2)/(var + localVar))/sqrt(var + localVar);
        
        fitAvg += weight*float4(GetChroma(X,Y).xyz, 1);
        fitVar += weight*float4((float4(var, sqr(bitnoise), sqr(bitnoise), 0) + GetChroma(X,Y)*GetChroma(X,Y)).xyz, weight);
        fitCov += weight*float4(GetChroma(X,Y).x*GetChroma(X,Y).yz, var, 0);

        kernel = IntKernel(float2(X,Y) - offset);
        weight = kernel.x * kernel.y;
        intAvg += weight*float4(GetChroma(X,Y).xyz, 1);
        intVar += weight*float4((float4(var, sqr(bitnoise), sqr(bitnoise), 0) + GetChroma(X,Y)*GetChroma(X,Y)).xyz, weight);
    }
    float weightSum = fitAvg.w;
    float weightSqrSum = fitVar.w;

    // Linear fit
    fitAvg /= weightSum;
    float3 Var = (fitVar / weightSum) - fitAvg*fitAvg;
    float2 Cov = (fitCov / weightSum) - fitAvg.x*fitAvg.yz;

    // Interpolation
    float intWeightSum = intAvg.w;
    float intWeightSqrSum = intVar.w;
    intAvg /= intWeightSum;
    intVar = (intVar / intWeightSum) - intAvg*intAvg;

    // Estimate error
    
    // Coefficient of determination
    float2 R2 = saturate((Cov * Cov) / (Var.x * Var.yz));
    // Error of fit
    float2 errFit = (1-R2) * (weightSqrSum / sqr(weightSum) + sqr((c0 - fitAvg).x) / Var.x) / (1 - weightSqrSum / sqr(weightSum));
    // Error of interpolation
    float2 errInt = lerp((intVar.yz / Var.yz) * intWeightSqrSum / sqr(intWeightSum), (sqr((c0 - intAvg).x) + intVar.x) / Var.x, R2);

    // Balance error of interpolation with error of fit.
    float2 strength = saturate(power * errInt / lerp(errFit, errInt, power));

    // Debugging
    // return float4(dot(strength,0.5), 0.5, 0.5, 1);
    // return float4(sqrt(Var.x)*10, 0.5, 0.5, 1);

    // Update c0
    // c0.yz = lerp(intAvg.yz, fitAvg.yz + ((c0 - fitAvg).x * Cov / Var.x), strength);
    c0.yz = fitAvg.yz + ((c0 - fitAvg).x * Cov / Var.x);

    return c0;
}