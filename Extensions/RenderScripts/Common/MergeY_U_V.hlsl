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
sampler s2 : register(s2);

float4 main(float2 tex : TEXCOORD0) : COLOR
{
	float y = tex2D(s0, tex).x;
    float u = tex2D(s1, tex).y;
    float v = tex2D(s2, tex).z;
    
    return float4(y, u, v, 1);
}
