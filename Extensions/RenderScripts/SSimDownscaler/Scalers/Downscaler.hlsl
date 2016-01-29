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

// -- Misc --
sampler s0:	register(s0);
float4 p0 : register(c0);
float2 p1 : register(c1);
float4 size0 : register(c2);

#define width  (p0[0])
#define height (p0[1])

#define dxdy (p1.xy)
#define ddxddy (size0.zw)

// -- Input processing --
#define GetFrom(s, pos) tex2Dlod(s, float4(pos, 0,0))

// -- Definitions --
#define factor ((ddxddy/dxdy)[axis])

// -- Handles --
#ifndef Get
	#define Get(pos)    (GetFrom(s0, pos))
#endif
#ifndef axis
	#define axis 0
#endif
#ifndef Initialization
	#define Initialization
#endif
#ifndef Postprocessing
	#define Postprocessing(x) x
#endif
#ifndef Kernel
	#define Kernel(x) saturate(0.5 + (0.5 - abs(x)) / factor)
	#define taps (1 + factor)
#endif

// -- Main code --
float4 main(float2 tex : TEXCOORD0) : COLOR{
    // Calculate bounds
	int low  = ceil ((tex - 0.5*taps*dxdy) * size0.xy - 0.5)[axis];
	int high = floor((tex + 0.5*taps*dxdy) * size0.xy - 0.5)[axis];

	float W = 0;
	float4 avg = 0;
	float2 pos = tex;
    Initialization;

	[loop] for (int k = low; k <= high; k++) {
		pos[axis] = ddxddy[axis] * (k + 0.5);
		float offset = (tex[axis] - pos[axis])*p0[axis];
		float w = Kernel(offset);
		
		avg += w*Get(pos);
		W += w;
	}
	avg /= W;
	
	return Postprocessing(avg);
}