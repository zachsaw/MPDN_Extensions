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
sampler s0 : register(s0);
sampler s1 : register(s1);
float4 args0 : register(c2);
float4 args1 : register(c3);

#define acuity args0[0]
#define margin args0[1]

#define norm(x) (rsqrt(rsqrt(dot(x*x,x*x))))

#include "../Common/Colourprocessing.hlsl"
#define Kb args0[2] //redefinition
#define Kr args0[3] //redefinition

float4 main(float2 tex : TEXCOORD0) : COLOR {
	float4 c0 = tex2D(s0, tex);
	float4 avg = tex2D(s1, tex);
	
	float3 diff = avg - c0;
	diff -= clamp(diff, -0.5/acuity, 0.5/acuity);
	diff = mul(YUVtoRGB, diff);
	float thr = smoothstep(0, 0.5, margin - length(diff*acuity));
	c0.xyz -= clamp(c0 - avg, -thr/acuity, thr/acuity);
	
	return c0;
}
