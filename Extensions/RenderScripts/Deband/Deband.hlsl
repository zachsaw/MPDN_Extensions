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
float4 p0 : register(c0);
float4 p1 : register(c1);
float4 args0 : register(c2);
float4 size0 : register(c3);
float4 size1 : register(c4);

#define ppx (size1[2])
#define ppy (size1[3])

#define acuity args0[0]
#define power  args0[1]
#define	margin args0[2]

#define sqr(x) dot(x,x)
#define norm(x) (rsqrt(rsqrt(sqr(sqr(x)))))

// Input Processing
#define Get(x,y)  	(tex2D(s1,float2(ppx,ppy)*(pos + 0.5 + float2(x,y))))

float4 main(float2 tex : TEXCOORD0) : COLOR {
	float4 c0 = tex2D(s0, tex);

	float2 pos = tex * size1.xy - 0.5;
	float2 offset = frac(pos);
	pos -= offset;

	// Load input
	float4x4 X = {Get(0,0) - c0, Get(1,0) - c0, Get(0,1) - c0, Get(1,1) - c0};
	
	// Use linear regression to interpolate
	float3x4 LinFit = {{-2, 2, -2, 2}, {-2, -2, 2, 2}, {1, 1, 1, 1}};
	float4 w = 0.25*mul(float1x3(offset-0.5,1), LinFit);
	float4 avg = c0 + mul(w,X);

	// Coefficient of determination
	float SSres = sqr(mul(float4(1,-1,-1,1),X).xyz);
	float SStot = (sqr(X[0].xyz) + sqr(X[1].xyz) + sqr(X[2].xyz) + sqr(X[3].xyz)) - sqr(mul(float4(0.25,0.25,0.25,0.25),X).xyz);
	float R = 1 - (SSres/SStot);

	// Merge with high res values
	float str = (1 + margin)*smoothstep(1 - power, 1, R);
	c0.rgb = clamp(avg, c0-0.5*str/acuity, c0+0.5*str/acuity);

	// Debugging
	//if (all(p0.xy == size0.xy)) return float4(R,0.5,0.5,1);

	return c0;
}