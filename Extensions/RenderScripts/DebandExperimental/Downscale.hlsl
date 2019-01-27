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

sampler s0   : register(s0);
float4 size0 : register(c0);
float4 args0 : register(c1);
float4 sizeOutput : register(c2);
float  iteration  : register(c3);

#define range args0[0]
#define power args0[1]

#define dxdy size0.zw

#define pi acos(-1)
#define phi ((1+sqrt(5))/2)
#define sqr(x) ((x)*(x))

#define factor (sizeOutput.xy / size0.xy)

#define Kernel(x) saturate(0.5 + (0.5 - abs(x)) / factor)

// Input Processing
#define GetXY(xy) 	(tex2D(s0,dxdy*(pos + 0.5 + (xy))))
#define Get(x,y)  	(GetXY(float2(x,y)))

#define FromMask(m) (dot(m, float4(1,2,4,0))/7.0)
#define ToMask(x) (frac((x) * (7.0/8.0) * float4(4,2,1,0)) > 7.0/16.0 ? 1 : 0)

#ifdef FirstPass
#define ToMask(x) float4(1,1,1,1)
#endif

float4 median(float2 tex) {
	float2 pos = tex * size0.xy - 0.5;
	float2 offset = pos - floor(pos);
	pos -= offset;

	float4x4 X = float4x4(Get(0,0), Get(0,1), Get(1,0), Get(1,1));
	float4x4 M = float4x4(ToMask(X[0].a), ToMask(X[1].a), ToMask(X[2].a), ToMask(X[3].a));
	float4 m = M[0] + M[1] + M[2] + M[3];

	float4 lo = min(min(X[0]*M[0], X[1]*M[1]), min(X[2]*M[2], X[3]*M[3]));
	float4 hi = max(max(X[0]*M[0], X[1]*M[1]), max(X[2]*M[2], X[3]*M[3]));

	return m > 3
		? (X[0]*M[0] + X[1]*M[1] + X[2]*M[2] + X[3]*M[3] - lo - hi) / (m - 2.0)
		: (X[0]*M[0] + X[1]*M[1] + X[2]*M[2] + X[3]*M[3]) / m;
}

float4 main(float2 tex : TEXCOORD0) : COLOR {
	float2 pos = tex * size0.xy - 0.5;
	float2 offset = pos - round(pos);
	pos -= offset;

	float4 mid = median(tex);

	float4 total = 0;
	float4 totalWeight = 0;

	for (int X=-1; X<=1; X++)
	for (int Y=-1; Y<=1; Y++)
	{
		float2 kernel = Kernel(float2(X,Y) - offset);
		float4 sample = Get(X,Y);
		float4 weight = kernel.x * kernel.y * saturate(1 - 10 * (range * abs(sample - mid) - 1));// * ToMask(sample.a);
		// float  weight = kernel.x * kernel.y * saturate(1 - dot(10, max(0, range * abs(sample - mid) - 1))) * sample.a;

		total += weight * sample;
		totalWeight += weight;
	}

	// float4 result = (totalWeight == 0 ? saturate(1000 * (1 - mid)) : total / totalWeight);
	float4 result = (totalWeight == 0 ? mid : total / totalWeight);
	result.a = FromMask(totalWeight == 0 ? 0 : 1);

	return result;
}