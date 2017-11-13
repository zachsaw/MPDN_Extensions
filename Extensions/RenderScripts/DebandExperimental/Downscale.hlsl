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

#define factor (size0.xy / sizeOutput.xy)

#define Kernel(x) saturate(0.5 + (0.5 - abs(x)) / factor)

// Input Processing
#define GetXY(xy) 	(tex2D(s0,dxdy*(pos + 0.5 + (xy))))
#define Get(x,y)  	(GetXY(float2(x,y)))

float4 main(float2 tex : TEXCOORD0) : COLOR{
	float2 pos = tex * size0.xy - 0.5;
	float2 offset = pos - round(pos);
	pos -= offset;

	float totalWeight = 0;
	float4 total = 0;

	for (int X=-1; X<=1; X++)
	for (int Y=-1; Y<=1; Y++) {
		float2 kernel = Kernel(float2(X,Y) - offset);
		float weight = kernel.x * kernel.y;
		float4 sample = Get(X,Y);

		total += weight * sample;
		totalWeight += weight;
	}
	total /= totalWeight;
	// total.w = totalVar;

	return total;
}