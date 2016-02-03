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
#ifndef Initialized
	sampler s0:	register(s0);
	float4 p0 : register(c0);
	float2 p1 : register(c1);
	float4 size0 : register(c2);

	#define Initialized 1
#endif

#define dxdy (p1.xy)
#define ddxddy (size0.zw)

// -- Definitions --
#define factor ((ddxddy*p0.xy)[axis])
#define GetFrom(s, pos) (tex2D(s, pos, 0, 0))

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
	#define Kernel(x) saturate(0.5 + (0.5 - abs(x)) * factor)
	#define taps (1 + 1/factor)
	#define maxtaps 2
#endif

#ifndef EntryPoint
	#define EntryPoint main
#endif

#ifndef AverageFormat
	#define AverageFormat float4
#endif

#ifndef OutputFormat
	#define OutputFormat AverageFormat
#endif

// -- Main code --
OutputFormat (float2 tex : TEXCOORD0
#ifdef ExtraArguments
, ExtraArguments
#endif
) : COLOR{
    // Calculate bounds
	int low  = floor(tex * size0.xy - 0.5*taps + 0.5)[axis];
	int high = floor(tex * size0.xy + 0.5*taps + 0.5)[axis];

	float W = 0;
	AverageFormat avg = 0;
	float2 pos = tex;
    Initialization;

	#ifndef maxtaps
    	int maxtaps = high - low;
    	[loop]
    #else
    	[unroll]
    #endif
    for (int k = 0; k < maxtaps; k++) {
		pos[axis] = ddxddy[axis] * (k + low + 0.5);
		float offset = (pos[axis] - tex[axis])*size0[axis];
		float w = Kernel(offset);
		
		avg += w*Get(pos);
		W += w;
	}
	avg /= W;
	
	return Postprocessing(avg);
}