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

sampler s0 : register(s0);
sampler s1 : register(s1);
sampler s2 : register(s2);
float4  p0 : register(c0);
float4 args0 : register(c2);
float4 size0 : register(c3);
float4 size1 : register(c4);
float4 sizeOutput : register(c5);

#define acuity args0[0]
#define power  args0[1]

#define threshold (power)
#define margin 	  sqrt(2*threshold)

#ifdef PRESERVE_DETAIL
#define enhance	(0.5*(threshold + 0.5*margin))
#else
#define enhance	0
#endif

#define pi acos(-1)
#define sqr(x) ((x)*(x))

float4 main(float2 tex : TEXCOORD0) : COLOR {
	// Calculate laplacian pyramid component
	float4 diff = (tex2D(s1, tex) - tex2D(s0, tex)) * acuity;

	diff = (diff + enhance/diff) * smoothstep(threshold, threshold + margin, abs(diff));

	// Reconstruct Gaussian pyramid
	return tex2D(s2, tex) + (diff/acuity);
}
