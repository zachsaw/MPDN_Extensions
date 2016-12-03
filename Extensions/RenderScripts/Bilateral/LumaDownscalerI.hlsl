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
float4 args0 : register(c3);

#define sqr(x) pow(x, 2)
#define offset args0.xy

// -- Downscaling --
#define AverageFormat	float2
#define Get(pos)		float2(GetFrom(s0, pos)[0], sqr(GetFrom(s0, pos)[0]))
#define PostProcessing(y)	float4(y[0], y[1] - sqr(y[0]), 0, 0)

#define pi acos(-1)
#define Kernel(x) cos(0.5*pi*x)
#define taps 2
#include "../SSimDownscaler/Scalers/Downscaler.hlsl"