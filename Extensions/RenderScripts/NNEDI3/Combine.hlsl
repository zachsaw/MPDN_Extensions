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
float4  p0 : register(c0);
float2  p1 : register(c1);
float4 size0 : register(c2);
float4 size1 : register(c3);
float4 size2 : register(c4);

#define width  (p0[0])
#define height (p0[1])

#define px (p1[0])
#define py (p1[1])

#define dxdy0 (size0.zw)
#define dxdy1 (size1.zw)

float4 main(float2 tex : TEXCOORD0) : COLOR{
    float par = frac(tex.x*p0.x / 2.0).x;

    //Fix size mismatch.
    float2 pos = floor(tex*p0.xy / float2(2.0, 1.0));
    float2 p = (pos.yx + 0.5)*dxdy0; // Assumes size is the same for s0 and s1

    float r0 = tex2D(s0, p).x;
    float r1 = tex2D(s1, p).x;
    float y = (par < 0.5) ? r0 : r1;
              
    float2 uv = tex2D(s2, tex).yz;
    
    return float4(y, uv, 1);
}
