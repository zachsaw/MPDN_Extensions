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

#define acuity ((1-power)/power)

// #define PRESERVE_COLOUR

#ifdef PRESERVE_DETAIL
#define detail (1)
#else
#define detail (0)
#endif

#ifdef PRESERVE_COLOUR
	#define norm(x) (dot((x).xyz,(x).xyz)/3.0)
#else
	#define norm(x) sqr(x)
#endif

#define sqr(x)	pow(x, 2)

#define Get(x,y)    	  (tex2Dlod(s0, float4(size0.zw*(pos + 0.5 + int2(x,y)), 0,0)))
#define GetResult(x,y)    (tex2Dlod(s2, float4(size0.zw*(pos + 0.5 + int2(x,y)), 0,0)))

float4 main(float2 tex : TEXCOORD0) : COLOR {
	float4 value = tex2D(s1, tex);
	// float4 var = tex2D(sVar, tex);

	// Calculate position
	float2 pos = tex * size0.xy - 0.5;
	float2 offset = pos - clamp(floor(pos), 0, size0.xy - 2);
	pos -= offset;

	// // Calculate position
	// float2 pos = tex * size0.xy - 0.5;
	// float2 offset = pos - floor(pos);
	// pos = floor(pos);

	float4x2 grad = 0;
	float4 mean = 0;
	float4 mean2 = 0;
	float4 result = 0;
	float4 totalWeight = 0;
	float4 totalWeight2 = 0;
	for (int x=0; x<2; x++)
	for (int y=0; y<2; y++) {
		float4 d = max(0, abs(value - Get(x,y)) * acuity * range - 1);
		float2 k = saturate(1 - abs(offset - float2(x,y)));
		float4 w = k.x * k.y / (1 + norm(d));

		grad += mul(float4x1(Get(x,y) / (1 + norm(d))), float1x2( (abs(offset - float2(x,y)) < 1 ? sign(offset - float2(x,y)) : 0) * float2(k.y, k.x) ));
		mean2 += w*sqr(Get(x,y) - value);
		mean += w*Get(x,y);
		result += w*GetResult(x,y);
		totalWeight += w;
	}
	grad /= transpose(float2x4(totalWeight, totalWeight));
	mean /= totalWeight;
	mean2 /= totalWeight;
	result /= totalWeight;

	// float4 var = (mean2 - sqr(mean - value)) / (1 - totalWeight2 / sqr(totalWeight));

	float4 lo = min(min(Get(0,0), Get(1,0)), min(Get(0,1), Get(1,1)));
	float4 hi = max(max(Get(0,0), Get(1,0)), max(Get(0,1), Get(1,1)));

	// float4 mean = lerp(
	// 	lerp(Get(0,0), Get(1,0), offset.x), 
	// 	lerp(Get(0,1), Get(1,1), offset.x), offset.y);
	// float4 result = lerp(
	// 	lerp(GetResult(0,0), GetResult(1,0), offset.x), 
	// 	lerp(GetResult(0,1), GetResult(1,1), offset.x), offset.y);

	// // Load input
	// float4x4 X = {Get(0,0), Get(1,0), Get(0,1), Get(1,1)};
	// float4x4 Y = {GetResult(0,0), GetResult(1,0), GetResult(0,1), GetResult(1,1)};
	
	// // Use linear regression to interpolate
	// float3x4 LinFit = {{-0.5, 0.5, -0.5, 0.5}, {-0.5, -0.5, 0.5, 0.5}, {0.25, 0.25, 0.25, 0.25}};
	// float4 w = mul(float1x3(offset-0.5,1), LinFit);
	
	// float4 mean = mul(w, X);
	// float4 result = mul(w, Y);

	// Calculate gradient
	// float4x2 grad = transpose(mul(float2x4(LinFit[0], LinFit[1]), X));
	// float4 grad2 = float4(dot(grad[0], grad[0]), dot(grad[1], grad[1]), dot(grad[2], grad[2]), dot(grad[3], grad[3]));

// #ifdef PRESERVE_COLOUR
// 	grad2 = dot(grad2.xyz, 1.0/3.0);
// #endif

	float4 diff = (value - mean);

	// diff += sign(diff) * (1 / max(diff*acuity, 1) - 1) / acuity;

#define detail_check(x) saturate(1 - (x) * (2 * range * detail))
#define sanity_check(x) saturate(1 - (x))
// #define banding_check(x) saturate(1 / ((x) * sqr(range * acuity)))
#define banding_check(x) (1 - saturate((x) * range * acuity - 1))
	float4 LBanding = 1
		// * banding_check(var / 0.25)
		// * banding_check(grad2)
		* banding_check(hi - lo)
		// * banding_check(hi - value)
		// * banding_check(value - lo)
		// * sanity_check(mean.w a sqr(range * acuity))
		// * sanity_check(10 * mean.w / sqr(hi - lo))
		// * sanity_check(10 * mean.w / (sqr(diff) + 1/sqr(range * acuity)))
		// * sanity_check(norm(diff) / norm(hi - lo))
		// * sanity_check(0.5 + (lo - hi) * range)
		* sanity_check(norm(diff * range) - sqr(0.5))
		// * sanity_check(abs(diff * range) - 0.5)
		// * detail_check(resLo - value)
		// * detail_check(value - resHi);
		* detail_check(lo - value)
		* detail_check(value - hi);
	float prior = 1 - 0.95;
	diff = lerp(diff, 0, (LBanding * prior) / (LBanding * prior + (1 - LBanding) * (1 - prior)));
	// diff = lerp(diff, 0, smoothstep(0.90, 0.95, LBanding));

	// Reconstruct Gaussian pyramid
	return result + diff;
}