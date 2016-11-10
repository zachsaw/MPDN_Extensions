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
#define bitnoise (1.0/(2.0*255.0))
#define noise (0.05)//(5*bitnoise)

#define chromaSize size1

#define dxdy (sizeOutput.zw)
#define ddxddy (chromaSize.zw)
#define chromaOffset (chromaParams.xy)

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

// #define radius 1
// #define localVar sqr(0.15)

// #define C(i,j) (rsqrt(localVar + X[i].w + X[j].w) * exp(-0.5*(sqr(X[i].x - X[j].x)/(localVar + X[i].w + X[j].w) + sqr((coords[i] - coords[j])/(localRadius + radius)))))
// #define c(i) (rsqrt(localVar + X[i].w) * exp(-0.5*(sqr(X[i].x - c0.x)/(localVar + X[i].w) + sqr((coords[i] - offset)/localRadius))))

// #define C(i,j) (exp(-0.5*(sqr(X[i].x - X[j].x)/localVar + sqr((coords[i] - coords[j])/radius))))
// #define c(i) (exp(-0.5*(sqr(X[i].x - c0.x)/localVar + sqr((coords[i] - offset)/radius))))

#define C(i,j) (1 / (1 + 0.5*(sqr(X[i].x - X[j].x)/localVar + sqr((coords[i] - coords[j])/radius))))
#define c(i) (1 / (1 + 0.5*(sqr(X[i].x - c0.x)/localVar + sqr((coords[i] - offset)/radius))))

#define KernelRadius 3
#define taps KernelRadius
#define N (KernelRadius*KernelRadius - 1)

struct ChromaValues {
    float2 coordinates[N+1];
    float4 values[N+1];
};

ChromaValues getValues(float2 pos) {
    ChromaValues result;

    int i=0;
    [unroll] for (int xx = -1; xx <= 1; xx++)
    [unroll] for (int yy = -1; yy <= 1; yy++) 
        if (!(xx == 0 && yy == 0))
        result.coordinates[i++] = float2(xx,yy);
    result.coordinates[N] = float2(0,0);

    for (int i=0; i<N+1; i++)
        result.values[i] = GetChromaXY(result.coordinates[i]);
    
    return result;
}

#define coords chroma.coordinates
#define X      chroma.values

// -- Main Code --
float4 main(float2 tex : TEXCOORD0) : COLOR{
    float4 c0 = tex2D(s0, tex);

    // Calculate position
    float2 coord = tex * chromaSize.xy - chromaOffset - 0.5;
    const float2 offset = coord - round(coord);
    const float2 pos = coord - offset;
    const ChromaValues chroma = getValues(pos);
    
    float4 total = 0;
    [unroll] for (int i=0; i<N+1; i++) {
        float2 w = saturate(2 - abs(coords[i] - offset));//saturate(1.5 - abs(coords[i] - offset));
        total += w.x*w.y*float4(X[i].x, X[i].x*X[i].x, X[i].w, 1);
    }
    total /= total.w;
    float localVar = sqr(noise) + saturate(total.y - total.x*total.x) + sqr(c0.x - total.x) + total.z;
    // float radius = lerp(1, 2, sqr(noise) / localVar);
    float radius = lerp(1, 2, (localVar - total.z) / localVar);

    // return float4(radius / 2, 0.5, 0.5, 1);
    
    #define M(i,j) Mx[min(i,j)*8 + max(i,j) - (min(i,j)*(min(i,j)+1))/2]
    #define b(i)   bx[i]
    float Mx[N*(N+1)/2];
    float bx[N];

    [unroll] for (int i=0; i<N; i++) {
        b(i) = c(i) - c(N) - (( C(i,N) - C(N,N) ));
        [unroll] for (int j=i; j<N; j++) {
            M(i,j) = C(i,j) - C(j,N) - (( C(i,N) - C(N,N) ));
        }
    }

    [unroll] for (int i=0; i<N; i++) {
        [unroll] for (int j=i+1; j<N; j++) {
            b(j) -= (( M(j,i) / M(i,i) )) * b(i);
            [unroll] for (int k=j; k<N; k++) {
                M(j,k) -= (( M(j,i) / M(i,i) )) * M(i,k);
            }
        }
    }

    float w[N];
    [unroll] for (int i=N-1; i>=0; i--) {
        w[i] = b(i);
        [unroll] for (int j=i+1; j<N; j++) {
            w[i] -= M(i,j) * w[j];
        }
        w[i] /= M(i,i);
    }

    float4 interp = X[N];
    for (int i=0; i<N; i++)
        interp += w[i] * (X[i] - X[N]);

    // Update c0
    c0.yz = interp.yz;

    return c0;
}