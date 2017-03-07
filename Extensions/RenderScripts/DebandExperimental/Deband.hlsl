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

// #define threshold (power)
// #define margin 	  sqrt(2*threshold)

#ifdef PRESERVE_DETAIL
#define enhance	(0.5*(threshold + 0.5*margin))
#else
#define enhance	0
#endif

#ifdef PRESERVE_DETAIL
#define detail 20
#else
#define detail 0
#endif

#define pi acos(-1)
#define sqr(x) ((x)*(x))

#define Get(x,y)    	  (tex2Dlod(s0, float4(size0.zw*(pos + 0.5 + int2(x,y)), 0,0)))
#define GetResult(x,y)    (tex2Dlod(s2, float4(size0.zw*(pos + 0.5 + int2(x,y)), 0,0)))

float4 main(float2 tex : TEXCOORD0) : COLOR {
    // Calculate position
	float2 pos = tex * size0.xy - 0.5;
	float2 offset = pos - floor(pos);
	pos -= offset;

	float4 value = tex2D(s1, tex);

	float4 weights;
	for (int x=0; x<2; x++)
	for (int y=0; y<2; y++) {
		float3 d = (value - Get(x,y))*acuity;
		weights[y + 2*x] = 1 / (1 + 0.5*dot(d,d));
	}
	weights /= dot(weights, 1);

	// float4 mean = lerp(
	// 	lerp(Get(0,0), Get(1,0), offset.x),
	// 	lerp(Get(0,1), Get(1,1), offset.x), offset.y);
	// float4 result = lerp(
	// 	lerp(GetResult(0,0), GetResult(1,0), offset.x),
	// 	lerp(GetResult(0,1), GetResult(1,1), offset.x), offset.y);

	float4 mean = 0; float4 result = 0;
	for (int x=0; x<2; x++)
	for (int y=0; y<2; y++) {
		float w = weights[y + 2*x];
		mean += w*Get(x,y);
		result += w*GetResult(x,y);
	}

	float4 lo = min(min(Get(0,0), Get(1,0)), min(Get(0,1), Get(1,1)));
	float4 hi = max(max(Get(0,0), Get(1,0)), max(Get(0,1), Get(1,1)));

	float4 diff = (value - mean);

#define threshold(x) ((x)*(1 - power) < power)
#define soft_threshold(x) (1 - saturate((x)*(1-power)/power - 1))
	// diff = threshold((hi - lo)*acuity) && threshold(abs(diff)*acuity) && threshold((lo - value)*acuity*detail) && threshold((value - hi)*acuity*detail) ? 0 : diff;
	float4 error = 
		soft_threshold((hi - lo)*acuity)
		* soft_threshold(abs(diff)*acuity)
		* soft_threshold(detail*(lo - value)*acuity)
		* soft_threshold(detail*(value - hi)*acuity);
	diff = lerp(diff, 0, error);
	// diff = (diff + enhance/((diff*acuity)*acuity)) * smoothstep(threshold, threshold + margin, abs(diff)*acuity);

	// Reconstruct Gaussian pyramid
	return result + diff;
}