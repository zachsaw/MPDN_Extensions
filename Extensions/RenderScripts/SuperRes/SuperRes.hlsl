// -- Main parameters --
#define strength (args0[0])
#define sharpness (args0[1])
#define anti_aliasing (args0[2])
#define anti_ringing (args0[3])

// -- Edge detection options -- 
#define acuity 4.0
#define edge_adaptiveness 2.0
#define baseline 0.2
#define radius 1.5

// -- Misc --
sampler s0 	  : register(s0);
sampler sDiff : register(s1);
sampler s2	  : register(s2); // Original
float4 p0	  : register(c0);
float2 p1	  : register(c1);
float4 size2  : register(c2); // Original size
float4 args0  : register(c3);

#define originalSize size2

#define width  (p0[0])
#define height (p0[1])

#define px (p1[0])
#define py (p1[1])

#define ppx (originalSize[2])
#define ppy (originalSize[3])

#define sqr(x) dot(x,x)
#define spread (exp(-1/(2.0*radius*radius)))
#define h 1.5

// -- Colour space Processing --
#include "../Common/ColourProcessing.hlsl"

// -- Input processing --
//Current high res value
#define Get(x,y)  	(tex2D(s0,tex+float2(px,py)*int2(x,y)).rgb)
//Difference between downsampled result and original
#define Diff(x,y)	(tex2D(sDiff,tex+float2(px,py)*int2(x,y)).rgb)
//Original values
#define Original(x,y)	(tex2D(s2,float2(ppx,ppy)*(pos2+int2(x,y)+0.5)).rgb)

// -- Main Code --
float4 main(float2 tex : TEXCOORD0) : COLOR{
	float4 c0 = tex2D(s0, tex);

	float3 Ix = (Get(1, 0) - Get(-1, 0)) / (2.0*h);
	float3 Iy = (Get(0, 1) - Get(0, -1)) / (2.0*h);
	float3 Ixx = (Get(1, 0) - 2 * Get(0, 0) + Get(-1, 0)) / (h*h);
	float3 Iyy = (Get(0, 1) - 2 * Get(0, 0) + Get(0, -1)) / (h*h);
	float3 Ixy = (Get(1, 1) - Get(1, -1) - Get(-1, 1) + Get(-1, -1)) / (4.0*h*h);
	//	Ixy = (Get(1,1) - Get(1,0) - Get(0,1) + 2*Get(0,0) - Get(-1,0) - Get(0,-1) + Get(-1,-1))/(2.0*h*h);
	float2x3 I = transpose(float3x2(
		normalize(float2(Ix[0], Iy[0])),
		normalize(float2(Ix[1], Iy[1])),
		normalize(float2(Ix[2], Iy[2]))
	));
	float3 stab = -anti_aliasing*(I[0] * I[0] * Iyy - 2 * I[0] * I[1] * Ixy + I[1] * I[1] * Ixx);
	stab += sharpness*0.5*(Ixx + Iyy);

	//Calculate faithfulness force
	float3 diff = Diff(0, 0);
	//diff = mul(DinvLabtoRGB(c0.xyz), diff);

	//Apply forces
	c0.xyz -= strength*(diff + stab);

	//Calculate position
	int2 pos = floor(tex*p0.xy);
	int2 pos2 = floor((pos + 0.5) * originalSize / p0.xy - 0.5);

	//Find extrema
	float3 Min = min(min(Original(0, 0), Original(1, 0)),
					 min(Original(0, 1), Original(1, 1)));
	float3 Max = max(max(Original(0, 0), Original(1, 0)),
					 max(Original(0, 1), Original(1, 1)));

	//Apply anti-ringing
	float3 AR = c0.xyz - clamp(c0.xyz, Min, Max);
	c0.xyz -= AR*smoothstep(0, (Max - Min) / anti_ringing - (Max - Min) + pow(2,-16), abs(AR));

	return c0;
}