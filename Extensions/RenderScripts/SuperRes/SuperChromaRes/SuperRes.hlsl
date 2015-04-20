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
#define sharpness (args0[1])
#define anti_aliasing (args0[2])
#define anti_ringing (args0[3])
#define softness (args1[0])

// -- Edge detection options -- 
#define edge_adaptiveness 1.0
#define baseline 0.0
#define acuity 6.0
#define radius 1.5

// -- Misc --
sampler s0 	  : register(s0);
sampler sDiff : register(s1);
sampler sU	  : register(s2);
sampler sV    : register(s3);

float4 p0	  : register(c0);
float2 p1	  : register(c1);
float4 size2  : register(c2);
float4 args0  : register(c3);
float4 args1  : register(c4);

#define width  (p0[0])
#define height (p0[1])

#define px (p1[0])
#define py (p1[1])

#define ppx (size2[2])
#define ppy (size2[3])

#define sqr(x) dot(x,x)
#define spread (exp(-1/(2.0*radius*radius)))
#define h 1.5

// -- Colour space Processing --
#include "../../Common/ColourProcessing.hlsl"
#define Kb args1[1] //redefinition
#define Kr args1[2] //redefinition

// -- Input processing --
//Current high res value
#define Get(x,y)  	(tex2D(s0,tex+float2(px,py)*int2(x,y)).xyz)
//Difference between downsampled result and original
#define Diff(x,y)	(tex2D(sDiff,tex+float2(px,py)*int2(x,y)).xyz)
//Original YUV
#define Original(x,y)	float2(tex2D(sU,tex+float2(ppx,ppy)*int2(x,y))[0], tex2D(sV,tex+float2(ppx,ppy)*int2(x,y))[0])

// -- Main Code --
float4 main(float2 tex : TEXCOORD0) : COLOR{
	float4 c0 = tex2D(s0, tex);
	
	float3 stab = 0;
	float W = 0;
	for (int i = -1; i <= 1; i++)
	for (int j = -1; j <= 1; j++) {
		float3 d = Get(0, 0) - Get(i, j);
		float x2 = sqr(acuity*d);
		float w = pow(spread, i*i + j*j)*lerp(1 / sqr(1 + x2), rsqrt(1 + x2), baseline);
		stab += d*w;
		W += w;
	}
	[branch] if (softness != 0)
	stab = softness*(stab / W)*pow(W / (1 + 4 * spread + 4 * spread*spread), edge_adaptiveness - 1.0);
	
	float3 Ix = (Get(1, 0) - Get(-1, 0)) / (2.0*h);
	float3 Iy = (Get(0, 1) - Get(0, -1)) / (2.0*h);
	float3 Ixx = (Get(1, 0) - 2 * Get(0, 0) + Get(-1, 0)) / (h*h);
	float3 Iyy = (Get(0, 1) - 2 * Get(0, 0) + Get(0, -1)) / (h*h);
	float3 Ixy = (Get(1, 1) - Get(1, -1) - Get(-1, 1) + Get(-1, -1)) / (4.0*h*h);
	//	Ixy = (Get(1,1) - Get(1,0) - Get(0,1) + 2*Get(0,0) - Get(-1,0) - Get(0,-1) + Get(-1,-1))/(2.0*h*h);
	float2x3 I = transpose(float3x2(
		normalize(float2(Ix[0], Iy[0])),
		normalize(float2(Ix[1], Iy[1])),
		normalize(float2(Ix[2], Iy[2]))
	));
	[branch] if (anti_aliasing != 0)
	stab -= anti_aliasing*(I[0] * I[0] * Iyy - 2 * I[0] * I[1] * Ixy + I[1] * I[1] * Ixx);
	stab += sharpness*0.5*(Ixx + Iyy);

	//Calculate faithfulness force
	float3 diff = Diff(0, 0);

	//Apply forces
	c0.yz -= strength*(diff + stab).yz;

	//Find extrema
	float2 Min = min(min(Original(0, 0), Original(1, 0)),
					 min(Original(0, 1), Original(1, 1)));
	float2 Max = max(max(Original(0, 0), Original(1, 0)),
					 max(Original(0, 1), Original(1, 1)));

	//Apply anti-ringing
	float2 AR = c0.yz - clamp(c0.yz, Min, Max);
	c0.yz -= AR*smoothstep(0, (Max - Min) / anti_ringing - (Max - Min) + pow(2, -16), abs(AR));

	return c0;
}
