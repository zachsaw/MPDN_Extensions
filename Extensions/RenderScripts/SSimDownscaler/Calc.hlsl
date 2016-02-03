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

sampler sR:	register(s0);
sampler sM:	register(s1);
sampler sL:	register(s2);
float4 p0 : register(c0);
float2 p1 : register(c1);
float4 size0 : register(c2);
#define Initialized 1

#define GetR(pos) tex2D(sR, pos)
#define GetM(pos) tex2D(sM, pos)
#define ExtraArguments float4 L

// -- Define horizontal convolver --
#define EntryPoint ScaleH
#define Get(pos) lerp(GetM(pos), L, GetR(pos))
#define axis 0
#include "./Scalers/Convolver.hlsl"

// -- Define vertical convolver -- 
#define EntryPoint Calc
#define Get(pos) ScaleH(pos, L)
#define axis 1
#include "./Scalers/Convolver.hlsl"

// -- Main code --
float4 main(float2 tex : TEXCOORD0) : COLOR {
    float4 L = tex2D(sL, tex);

	return Calc(tex, L);
}