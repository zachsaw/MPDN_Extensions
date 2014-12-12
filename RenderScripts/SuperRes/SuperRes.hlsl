// -- Main parameters --
#define strength (args0[0])
#define softness (args0[1])
#define anti_aliasing (args0[2])
#define anti_ringing (args0[3])

// -- Edge detection options -- 
#define acuity 4.0
#define edge_adaptiveness 2.0
#define baseline 0.2
#define radius 1.5

// -- Color space options --
#define GammaCurve sRGB
#define gamma 2.2

// -- Misc --
sampler s0 	  : register(s0);
sampler sDiff : register(s1);
sampler s2	  : register(s2); // Original
float4 p0	  : register(c0);
float2 p1	  : register(c1);
float4 size2  : register(c2); // Original size
float4 args0  : register(c3);

#define width  (p0[0])
#define height (p0[1])

#define px (p1[0])
#define py (p1[1])

#define ppx (size2[2])
#define ppy (size2[3])

#define sqr(x) dot(x,x)
#define spread (exp(-1/(2.0*radius*radius)))
#define h 1.5

// -- Option values --
#define None  1
#define sRGB  2
#define Power 3
#define Fast  4
#define true  5
#define false 6

// -- Gamma processing --
#define RGBtoXYZ float3x3(float3(0.4124,0.3576,0.1805),float3(0.2126,0.7152,0.0722),float3(0.0193,0.1192,0.9502))
#define XYZtoRGB (625.0*float3x3(float3(67097680, -31827592, -10327488), float3(-20061906, 38837883, 859902), float3(1153856, -4225640, 21892272))/12940760409.0)
#define A (0.272433)

#if GammaCurve == sRGB
float3 Gamma(float3 x)   { return lerp(x * 12.9232102, 1.055*pow(x, 1 / 2.4) - 0.055, saturate(0.5 + 65536 * (x - (0.0392857 / 12.9232102)))); }
float3 GammaInv(float3 x){ return lerp(x / 12.9232102, pow((x + 0.055) / 1.055, 2.4), saturate(0.5 + 65536 * (x - 0.0392857))); }
#elif GammaCurve == Power
float3 Gamma(float3 x)   { return pow(saturate(x), 1 / gamma); }
float3 GammaInv(float3 x){ return pow(saturate(x), gamma); }
#elif GammaCurve == Fast
float3 Gamma(float3 x)   { return saturate(x)*rsqrt(saturate(x)); }
float3 GammaInv(float3 x){ return x*x; }
#elif GammaCurve == None
float3 Gamma(float3 x)   { return x; }
float3 GammaInv(float3 x){ return x; }
#endif

// -- Color space Processing --
float3 Labf(float3 x)   { return lerp(x * (29 * 29) / (3 * 6 * 6) + ( 4 / 29), pow(x, 1 / 3.0), saturate(0.5 + 65536 * (x - (6 * 6 * 6 / (29 * 29 * 29))))); }
float3 Labfinv(float3 x){ return lerp(( x - 4 / 29) * (3 * 6 * 6) / (29 * 29), x*x*x		  , saturate(0.5 + 65536 * (x - (6 / 29)))); }

float3 RGBToLab(float3 rgb) {
	float3 xyz = mul(RGBtoXYZ, rgb);
	xyz = Labf(xyz);
	return float3(1.16*xyz.y - 0.16, 5.0*(xyz.x - xyz.y), 2.0*(xyz.y - xyz.z));
}

float3 LabToRGB(float3 res) {
	float3 xyz = (res.x + 0.16) / 1.16 + float3(res.y / 5.0, 0, -res.z / 2.0);
	return saturate(mul(XYZtoRGB, Labfinv(xyz)));
}

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
	stab -= softness*0.5*(Ixx + Iyy);

	//Calculate faithfulness force
	float3 diff = Diff(0, 0);

	//Apply forces
	c0.xyz -= strength*(diff + stab);

	//Calculate position
	int2 pos = floor(tex*p0.xy);
	int2 pos2 = floor((pos + 0.5) * size2 / p0.xy - 0.5);

	//Find extrema
	float3 Min = min(min(Original(0, 0), Original(1, 0)),
					 min(Original(0, 1), Original(1, 1)));
	float3 Max = max(max(Original(0, 0), Original(1, 0)),
					 max(Original(0, 1), Original(1, 1)));

	//Apply anti-ringing
	c0.xyz -= anti_ringing*(c0.xyz - min(Max, max(Min, c0.xyz)));

	//Convert to linear light
	c0.rgb = LabToRGB(c0.xyz);

	return c0;
}