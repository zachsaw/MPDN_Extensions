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

sampler sL:	register(s0);
sampler sM:	register(s1);
sampler sR:	register(s2);
float4 sizeOutput : register(c0);
float4 size0 : register(c1);
float strength : register(c2);

#define Initialized 1

#define GetR(pos) tex2D(sR, pos)
#define GetM(pos) tex2D(sM, pos)
#define ExtraArguments float4 L

// -- Define horizontal convolver --
#define EntryPoint ScaleH
#define Get(pos) float4(lerp(GetM(pos), L, GetR(pos)).xyz, dot(GetR(pos).xyz, GetR(pos).xyz))
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

    float4 result = Calc(tex, L);

	return lerp(L, result, strength/(strength + (1 - strength) * result.a/200));
}