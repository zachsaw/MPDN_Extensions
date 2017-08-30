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

#define main Downscale
#include "Downscale.hlsl"
#undef main

sampler s1   : register(s1);

float4 main(float2 tex : TEXCOORD0) : COLOR{
	float4 result = Downscale(tex);

	// Copy Chroma
	result.yz = tex2D(s1, tex).yz;

	return result;
}