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

#define noise (0.01)

#define spreadH (spread)

#if MODE == 0
	#define spreadL (1e-3)
#else
	#define spreadL (spread)
#endif

#define dxdy (sizeOutput.zw)

#include "../Common/ColourProcessing.hlsl"
#define sqr(x)	pow(x,2)

#define GetH(x,y) tex2D(s0, tex + dxdy*float2(x,y))
#define GetL(x,y) tex2D(s1, tex + dxdy*float2(x,y))

// -- Main code --
float4 main(float2 tex : TEXCOORD0) : COLOR {
	float4 meanH = 0;
	float4 meanL = 0;
	for (int X=-1; X<=1; X++)
	for (int Y=-1; Y<=1; Y++) {
		meanH += GetH(X,Y) * pow(spreadH, sqr(X) + sqr(Y));
		meanL += GetL(X,Y) * pow(spreadL, sqr(X) + sqr(Y));
	}
	meanH /= (1 + 4*spreadH + 4*spreadH*spreadH);
	meanL /= (1 + 4*spreadL + 4*spreadL*spreadL);

	float varH = 0;
	float varL = 0;
	for (int X=-1; X<=1; X++)
	for (int Y=-1; Y<=1; Y++) {
		varH += sqr(Luma(GetH(X,Y) - meanH)) * pow(spreadH, sqr(X) + sqr(Y));
		varL += sqr(Luma(GetL(X,Y) - meanL)) * pow(spreadL, sqr(X) + sqr(Y));
	}
	varH /= (1 + 4*spreadH + 4*spreadH*spreadH) - (1 + 4*spreadH*spreadH + 4*spreadH*spreadH*spreadH*spreadH)/(1 + 4*spreadH + 4*spreadH*spreadH);
	varL /= (1 + 4*spreadL + 4*spreadL*spreadL) - (1 + 4*spreadL*spreadL + 4*spreadL*spreadL*spreadL*spreadL)/(1 + 4*spreadL + 4*spreadL*spreadL);

	varH = varH + meanH.w + sqr(noise);
	varL = varL + sqr(noise);

	float R = (1 + oversharp) * sqrt(varL/varH);

	// Variance matching:
	// x -> mu + R (x - E[x]) = x + (mu - (1-R) * x - R * E[x])
	return float4(meanL.xyz - R * meanH, R);
}