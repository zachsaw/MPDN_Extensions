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
sampler sUV : register(s1);
float4 args0 : register(c3);

// -- Downscaling -- 
#define offset args0.xy
#define EntryPoint Downscale
#include "../../SSimDownscaler/Scalers/Downscaler.hlsl"

// -- Main code --
float4 main(float2 tex : TEXCOORD0) : COLOR{
	float4 c0 = Downscale(tex);
	float2 uv = tex2D(sUV, tex).yz;

	float2 diff = c0.yz - uv;

	return float4(0, diff, c0.x);
}
