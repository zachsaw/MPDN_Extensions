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
// sampler sVar : register(s3);
float4 args0 : register(c0);
float4 size0 : register(c1);
float4 size1 : register(c2);
float4 sizeOutput : register(c3);

#define range args0[0]
#define power args0[1]

// #define PRESERVE_COLOUR

#ifdef PRESERVE_DETAIL
#define detail (1)
#else
#define detail (0)
#endif

#ifdef PRESERVE_COLOUR
	#define norm(x) (dot2((x).xyz)/3.0)
#else
	#define norm(x) sqr(x)
#endif

#define sqr(x)	pow(x, 2)
#define dot2(x) dot(x,x)

#define Get(x,y)    	  (tex2Dlod(s0, float4(size0.zw*(pos + 0.5 + int2(x,y)), 0,0)))
#define GetResult(x,y)    (tex2Dlod(s2, float4(size0.zw*(pos + 0.5 + int2(x,y)), 0,0)))

#define FromMask(m) (dot(m, float4(1,2,4,0))/7.0)
#define ToMask(x) (frac((x) * (7.0/8.0) * float4(4,2,1,0)) > 7.0/16.0 ? 1 : 0)
#define GetMask(x,y)  (ToMask(Get(x,y).a))

float4 main(float2 tex : TEXCOORD0) : COLOR {
	float4 value = tex2D(s1, tex);

	// Calculate position
	float2 pos = tex * size0.xy - 0.5;
	float2 offset = pos - floor(pos);
	pos = floor(pos);

	float4 lo = min(min(Get(0,0), Get(1,0)), min(Get(0,1), Get(1,1)));
	float4 hi = max(max(Get(0,0), Get(1,0)), max(Get(0,1), Get(1,1)));

	float4 mean = lerp(
		lerp(Get(0,0), Get(1,0), offset.x), 
		lerp(Get(0,1), Get(1,1), offset.x), offset.y);
	float4 result = lerp(
		lerp(GetResult(0,0), GetResult(1,0), offset.x), 
		lerp(GetResult(0,1), GetResult(1,1), offset.x), offset.y);
	float4 mask = lerp(
		lerp(GetMask(0,0), GetMask(1,0), offset.x), 
		lerp(GetMask(0,1), GetMask(1,1), offset.x), offset.y);

	float4 diff = (value - mean);

#define detail_check(x) saturate(1 - (x) * (2 * range * detail))
#define banding_check(x) (1 - saturate((x) * range - 1))

	float4 LBanding = 1
		* banding_check(hi - value)
		* banding_check(value - lo)
		// * mask
		* mask.x * mask.y * mask.z
		* detail_check(lo - value)
		* detail_check(value - hi);
	float BandingPrior = 1 - 0.95;

	diff = lerp(diff, 0, (LBanding * BandingPrior) / (LBanding * BandingPrior + (1 - LBanding) * (1 - BandingPrior)));

	// float4 LRinging = 1-LBanding;
	// float RingingPrior = 0.01;

	// return lerp(result + diff, value, (LRinging * RingingPrior) / (LRinging * RingingPrior + (1 - LRinging) * (1 - RingingPrior)));

	// Reconstruct Gaussian pyramid
	return result + diff;
}