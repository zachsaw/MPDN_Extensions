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
float spread : register(c2);
float oversharp : register(c3);
float4 sizeOutput : register(c4);

#define dxdy (sizeOutput.zw)

#include "../Common/ColourProcessing.hlsl"
#define sqr(x)	pow(x,2)

#define GetH(x,y) tex2D(s0, tex + dxdy*float2(x,y))
#define GetL(x,y) tex2D(s1, tex + dxdy*float2(x,y))

// -- Main code --
float4 main(float2 tex : TEXCOORD0) : COLOR {
    float4 c0 = tex2D(s0, tex);
    float4 c1 = tex2D(s1, tex);

	float4 meanH = (GetH(0,0) + spread * (GetH(-1, 0) + GetH(0, 1) + GetH(1, 0) + GetH(0, -1)))/(1.0 + 4.0 * spread);
	float4 meanL = (GetL(0,0) + spread * (GetL(-1, 0) + GetL(0, 1) + GetL(1, 0) + GetL(0, -1)))/(1.0 + 4.0 * spread);

	float varH = (sqr(Luma(GetH(0, 0) - meanH)) + spread * (sqr(Luma(GetH(-1, 0) - meanH)) + sqr(Luma(GetH(0, 1) - meanH)) + sqr(Luma(GetH(1, 0) - meanH)) + sqr(Luma(GetH(0, -1) - meanH)))) / (1 + 4 * spread);
	float varL = (sqr(Luma(GetL(0, 0) - meanL)) + spread * (sqr(Luma(GetL(-1, 0) - meanL)) + sqr(Luma(GetL(0, 1) - meanL)) + sqr(Luma(GetL(1, 0) - meanL)) + sqr(Luma(GetL(0, -1) - meanL)))) / (1 + 4 * spread);

	varH = varH + meanH.w + sqr(0.5/255.0);
	varL = varL + sqr(0.5/255.0);

	float R = (1 + oversharp) * sqrt(varL/varH);

	// Variance matching:
	// x -> mu + R (x - E[x]) = x + (mu - (1-R) * x - R * E[x])
#if MODE == 0
	return float4(c1.xyz - R * meanH, R);
#elif MODE == 1
	return float4(meanL.xyz - R * meanH, R);
#else
	return float4(c1.xyz - R * c0, R);
#endif
}