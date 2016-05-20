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
// -- Misc --
sampler s0 : register(s0);
sampler s1 : register(s1);
float4 p0 :  register(c0);
float2 p1 :  register(c1);

#define width  (p0[0])
#define height (p0[1])

#define dxdy (p1.xy)

#include "../Common/ColourProcessing.hlsl"
#define sqr(x)	pow(x,2)

#define GetH(x,y) tex2D(s0, tex + dxdy*float2(x,y))
#define GetL(x,y) tex2D(s1, tex + dxdy*float2(x,y))

#define spread 0.5

// -- Main code --
float4 main(float2 tex : TEXCOORD0) : COLOR {
    float4 c0 = tex2D(s0, tex);
    float4 c1 = tex2D(s1, tex);

	float varH = (sqr(Luma(GetH(-1, 0) - c0)) + sqr(Luma(GetH(0, 1) - c0)) + sqr(Luma(GetH(1, 0) - c0)) + sqr(Luma(GetH(0, -1) - c0))) * sqr(spread) / (1.0 + 4.0 * spread);
	float varL = (sqr(Luma(GetL(-1, 0) - c1)) + sqr(Luma(GetL(0, 1) - c1)) + sqr(Luma(GetL(1, 0) - c1)) + sqr(Luma(GetL(0, -1) - c1))) * sqr(spread) / (1.0 + 4.0 * spread);

	float4 meanH = (GetH(0,0) + spread * (GetH(-1, 0) + GetH(0, 1) + GetH(1, 0) + GetH(0, -1)))/(1.0 + 4.0 * spread);
	float4 meanL = (GetL(0,0) + spread * (GetL(-1, 0) + GetL(0, 1) + GetL(1, 0) + GetL(0, -1)))/(1.0 + 4.0 * spread);

	varH = saturate(varH - sqr(meanH - c0)) + meanH.w + sqr(0.5/255.0);
	varL = saturate(varL - sqr(meanL - c1))           + sqr(0.5/255.0);

	float R = sqrt(varL/varH);

	// Variance matching:
	// x -> mu + R (x - E[x]) = x + (mu - (1-R) * x - R * E[x])
	return float4((c1 - R*meanH).rgb, R);
}