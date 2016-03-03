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

// -- Definitions --
#define factor (1)
#define GetFrom(s, pos) (tex2D(s, pos, 0, 0))

// -- Handles --
#ifndef Get
	#define Get(pos)    (GetFrom(s0, pos))
#endif

#ifndef axis
	#define axis 0
#endif

#ifndef offset
	#define offset float2(0,0)
#endif

#ifndef Initialization
	#define Initialization
#endif

#ifndef PostProcessing
	#define PostProcessing(x) x
#endif

#ifndef Kernel
	#define kernelWidth 2
	#define Kernel(x) saturate(0.5 + (kernelWidth*0.5 - abs(x)))
	#define taps (kernelWidth + 1)
	#define maxtaps taps
#endif

#ifndef EntryPoint
	#define EntryPoint main
#endif

#ifndef AverageFormat
	#define AverageFormat float4
#endif

#ifndef OutputFormat
	#define OutputFormat float4
#endif

// -- Main code --
OutputFormat EntryPoint(float2 tex : TEXCOORD0
#ifdef ExtraArguments
, ExtraArguments
#endif
) : COLOR{
    // Calculate bounds
	int low  = floor(-0.5*taps - offset[axis]); // + tex * size0 + 0.5
	int high = floor(+0.5*taps - offset[axis]); // + tex * size0 + 0.5

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
		pos[axis] = tex[axis] + dxdy[axis] * (k + low + 1);
		float rel = (k + low + 1) + offset[axis];
		float w = Kernel(rel);

		avg += w*(Get(pos));
		W += w;
	}
	avg /= W;
	
	return PostProcessing(avg);
}