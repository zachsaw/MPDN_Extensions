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
float2 sizeOutput : register(c0);
float4 size0 : register(c1);
float4 size1 : register(c2);

#define dxdy0 (size0.zw)
#define dxdy1 (size1.zw)

float4 main(float2 tex : TEXCOORD0) : COLOR {
    int2 par = round(frac(tex*sizeOutput.xy / 2.0));

    //Fix size mismatch.
    int2 pos = floor(tex*sizeOutput.xy / float2(1.0, 2.0));

    if (par.x == par.y) {
        return tex2D(s0, (pos + 0.5)*dxdy0);
    } else {
        return tex2D(s1, (pos + 0.5)*dxdy1);
    }
}
