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
#include "../Common/ColourProcessing.hlsl"

#define sqr(x)	pow((x),2)
#define Get(pos)	float4(GetFrom(s0, pos).xyz, GetFrom(s0, pos).w + sqr(Luma(GetFrom(s0, pos))))
#define PostProcessing(x) float4(x.xyz, (x[3] - sqr(Luma(x))))
#include "../SSimDownscaler/Scalers/Downscaler.hlsl"