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
#define KernelRadius 3

#if Mode == 0 
    #define fixLuma 0
    #define old main
#elif Mode == 1
    #define radius 1
    #define noise 0.1
    #define localVar 0//sqr(bitnoise)
    #define fixLuma 0
    #define krig main
#elif Mode == 2
    #define radius (2.0/3.0)
    #define noise 0.01
    #define localVar 0//sqr(bitnoise)
    #define fixLuma (1/sqr(bitnoise))
    #define test7 main
#endif

#define C(i,j) (rsqrt(sqr(noise) + X[i].w + X[j].w) * exp(-0.5*(sqr(X[i].x - X[j].x)/(sqr(noise) + X[i].w + X[j].w) + sqr((coords[i] - coords[j])/radius))) + fixLuma * (X[i].x - c0.x) * (X[j].x - c0.x))
#define c(i) (rsqrt(sqr(noise) + X[i].w + localVar) * exp(-0.5*(sqr(X[i].x - c0.x)/(sqr(noise) + X[i].w + localVar) + sqr((coords[i] - offset)/radius))))

// #define cubic(xy) (pow(1 + (xy).x + (xy).y, 3))
// #define C(i,j) (rsqrt(sqr(noise) + X[i].w + X[j].w) * exp(-0.5*(sqr(X[i].x - X[j].x)/(sqr(noise) + X[i].w + X[j].w))) + 1e5*cubic((coords[i] - offset) * (coords[j] - offset)))
// #define c(i) (rsqrt(sqr(noise) + X[i].w + localVar) * exp(-0.5*(sqr(X[i].x - c0.x)/(sqr(noise) + X[i].w + localVar))))

// #define Manhatten(xy) (abs((xy)[0]) + abs((xy)[1]))
// #define C(i,j) (rsqrt(sqr(noise) + X[i].w + X[j].w) * exp(-0.5*(sqr(X[i].x - X[j].x)/(sqr(noise) + X[i].w + X[j].w)) - Manhatten((coords[i] - coords[j])/radius)) + fixLuma * (X[i].x - c0.x) * (X[j].x - c0.x))
// #define c(i) (rsqrt(sqr(noise) + X[i].w + localVar) * exp(-0.5*(sqr(X[i].x - c0.x)/(sqr(noise) + X[i].w + localVar)) - Manhatten((coords[i] - offset)/radius)))

// -- Main Code --
float4 old(float2 tex : TEXCOORD0) : COLOR{
    float4 c0 = tex2D(s0, tex);

    // Calculate position
    float2 pos = tex * chromaSize.xy - chromaOffset - 0.5;
    float2 offset = pos - floor(pos);
    pos -= offset;

#ifdef old
    #define localVar sqr(noise)
#endif

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

#define taps KernelRadius
#define N (KernelRadius*KernelRadius - 1)

// -- Main Code --
float4 krig(float2 tex : TEXCOORD0) : COLOR{
    float4 c0 = tex2D(s0, tex);

    // Calculate position
    float2 pos = tex * chromaSize.xy - chromaOffset - 0.5;
    float2 offset = pos - round(pos);
    pos -= offset;

    float2 coords[N+1];
    int i=0;
    [unroll] for (int xx = minX; xx <= maxX; xx++)
    [unroll] for (int yy = minX; yy <= maxX; yy++) 
        if (!(xx == 0 && yy == 0))
        coords[i++] = float2(xx,yy);
    coords[N] = float2(0,0);

    float4 X[N+1];
    for (int i=0; i<N+1; i++)
        X[i] = GetChromaXY(coords[i]);

    // float4 mean = 0;
    // float4 mean2 = 0;
    // float2 cov = 0;
    // float totalC = 0;
    // #define weight(i) 1
    // for (int i=0; i<N+1; i++) {
    //     mean += weight(i)*X[i];
    //     mean2 += weight(i)*X[i]*X[i];
    //     cov += weight(i)*X[i].x*X[i].yz;
    //     totalC += weight(i);
    // }
    // mean /= totalC;
    // float4 var = (mean2/totalC) + sqr(bitnoise) - mean*mean;
    // cov = (cov/totalC) - mean.x*mean.yz;
    // // var.x += mean.w;
    // float R2 = dot(0.5, (cov*cov)/(var.x*var.yz));
    // return float4(R2, 0.5, 0.5, 1);

    // return float4(fixLumaBool, 0.5, 0.5, 1);

    // float4 mean = 0;
    // float3x3 Q = 0;
    // float totalC = 0;
    // for (int i=0; i<N+1; i++) {
    //     mean += c(i)*X[i];
    //     totalC += c(i);
    // }
    // mean /= totalC;
    // for (int i=0; i<N+1; i++) {
    //     Q += mul(float3x1((X[i] - mean).xyz), float1x3((X[i] - mean).xyz));
    //     Q[0][0] += X[i].w;
    // }
    // #define trace(M) (M[0][0] + M[1][1] + M[2][2])

    // float q = trace(Q)/3;
    // Q = Q - q * float3x3(1,0,0,0,1,0,0,0,1);
    // float p = sqrt(trace(mul(Q,Q))/6);
    // float r = determinant(Q) / (2.0 * pow(p,3));
    // float phi = acos(clamp(r, -1, 1)) / 3.0;

    // float sigma2 = q + 2 * p * cos(phi);
    // float R2 = sigma2 / (3*q);

    // return float4(0.1*R2/(1-R2), 0.5, 0.5, 1);
    // return float4(fixLuma * sqr(0.01), 0.5, 0.5, 1);
    
    float M[N][N];
    float b[N];

    [unroll] for (int i=0; i<N; i++) {
        b[i] = c(i) - c(N) - C(i,N) + C(N,N);
        [unroll] for (int j=i; j<N; j++) {
            M[i][j] = C(i,j) - C(i,N) - C(j,N) + C(N,N);
        }
    }

    [unroll] for (int i=0; i<N; i++) {
        [unroll] for (int j=i+1; j<N; j++) {
            b[j] -= b[i] * M[i][j] / M[i][i];
            [unroll] for (int k=j; k<N; k++) {
                M[j][k] -= M[i][k] * M[i][j] / M[i][i];
            }
        }
    }

    float w[N];
    float det = 1;
    float Tr = 0;
    [unroll] for (int i=N-1; i>=0; i--) {
        w[i] = b[i];
        [unroll] for (int j=i+1; j<N; j++) {
            w[i] -= M[i][j] * w[j];
        }
        w[i] /= M[i][i];
        det *= M[i][i];
        Tr += M[i][i];
    }

    float4 interp = X[N];
    for (int i=0; i<N; i++)
        interp += w[i] * (X[i] - X[N]);

    // return float4(abs(c0.x - interp.x)*100, 0.5, 0.5, 1);
    // return float4(det*0.000001, 0.5, 0.5, 1);

    // Update c0
    c0.yz = interp.yz;

    // c0.yz -= 0.5;
    // c0.xyz = normalize(c0.xyz)/2;
    // c0.yz += 0.5;

    return c0;
}

#define fixLuma 0
#define taps KernelRadius
#define N (KernelRadius*KernelRadius)

// -- Main Code --
float4 test7(float2 tex : TEXCOORD0) : COLOR{
    float4 c0 = tex2D(s0, tex);

    // Calculate position
    float2 pos = tex * chromaSize.xy - chromaOffset - 0.5;
    float2 offset = pos - round(pos);
    pos -= offset;

    float2 coords[N];
    int i=0;
    [unroll] for (int xx = minX; xx <= maxX; xx++)
    [unroll] for (int yy = minX; yy <= maxX; yy++) 
        coords[i++] = float2(xx,yy);

    float4 X[N];
    [unroll] for (int i=0; i<N; i++)
        X[i] = GetChromaXY(coords[i]);

    float M[N+2][N+2];
    float b[N+2];

    [unroll] for (int i=0; i<N; i++) {
        b[i] = c(i);
        [unroll] for (int j=i; j<N; j++)
            M[i][j] = C(i,j);
        M[i][N] = 1;
        M[i][N+1] = X[i].x;
    }
    M[N][N] = 0;
    M[N][N+1] = 0;
    M[N+1][N+1] = 0;
    b[N] = 1;
    b[N+1] = c0.x;

    float bx[N+2];
    [unroll] for (int i=0; i<N+2; i++)
        bx[i] = b[i];

    [unroll] for (int i=0; i<N+2; i++) {
        [unroll] for (int j=i+1; j<N+2; j++) {
            b[j] -= b[i] * M[i][j] / M[i][i];
            [unroll] for (int k=j; k<N+2; k++)
                M[j][k] -= M[i][k] * M[i][j] / M[i][i];
        }
    }

    float w[N+2];
    float det = 1;
    [unroll] for (int i=N+1; i>=0; i--) {
        w[i] = b[i];
        [unroll] for (int j=i+1; j<N+2; j++) {
            w[i] -= M[i][j] * w[j];
        }
        w[i] /= M[i][i];
        det *= M[i][i];
    }

    float4 interp = 0;
    [unroll] for (int i=0; i<N; i++)
        interp += w[i] * X[i];

    float sigma = 0;
    [unroll] for (int i=0; i<N; i++)
        sigma += w[i] * b[i];
    // return float4(1-sigma/9, 0.5, 0.5, 1);

    // Update c0
    c0.yz = interp.yz;

    return c0;
}

#define KernelRadius 3
#define taps KernelRadius
#define N (KernelRadius*KernelRadius - 1)

// -- Main Code --
float4 test8(float2 tex : TEXCOORD0) : COLOR{
    float4 c0 = tex2D(s0, tex);

    // Calculate position
    float2 pos = tex * chromaSize.xy - chromaOffset - 0.5;
    float2 offset = pos - round(pos);
    pos -= offset;

    float2 coords[N+1];
    int i=0;
    [unroll] for (int xx = minX; xx <= maxX; xx++)
    [unroll] for (int yy = minX; yy <= maxX; yy++) 
        if (!(xx == 0 && yy == 0))
        coords[i++] = float2(xx,yy);
    coords[N] = float2(0,0);

    float4 X[N+1];
    for (int i=0; i<N+1; i++)
        X[i] = GetChromaXY(coords[i]);

    float M[N][N];
    float b[N];

    [unroll] for (int i=0; i<N; i++) {
        b[i] = c(i) - c(N) - C(i,N) + C(N,N);
        [unroll] for (int j=i; j<N; j++) {
            M[i][j] = C(i,j) - C(i,N) - C(j,N) + C(N,N);
        }
    }

    [unroll] for (int i=0; i<N; i++) {
        [unroll] for (int j=i+1; j<N; j++) {
            b[j] -= b[i] * M[i][j] / M[i][i];
            [unroll] for (int k=j; k<N; k++) {
                M[j][k] -= M[i][k] * M[i][j] / M[i][i];
            }
        }
    }

    float w[N];
    float det = 1;
    float Tr = 0;
    [unroll] for (int i=N-1; i>=0; i--) {
        w[i] = b[i];
        [unroll] for (int j=i+1; j<N; j++) {
            w[i] -= M[i][j] * w[j];
        }
        w[i] /= M[i][i];
        det *= M[i][i];
        Tr += M[i][i];
    }

    float4 interp = X[N];
    for (int i=0; i<N; i++)
        interp += w[i] * (X[i] - X[N]);

    // return float4(abs(c0.x - interp.x)*100, 0.5, 0.5, 1);
    // return float4(det*0.000001, 0.5, 0.5, 1);

    // Update c0
    c0.yz = interp.yz;

    // c0.yz -= 0.5;
    // c0.xyz = normalize(c0.xyz)/2;
    // c0.yz += 0.5;

    return c0;
}