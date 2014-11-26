// -- Main parameters --
#define strength 0.75
#define softness 0.25

// -- Edge detection options -- 
#define edge_adaptiveness 1.0
#define baseline 0.25
#define acuity 7.5
#define radius 0.65

// -- Color space options --
#define GammaCurve sRGB
#define gamma 2.2

// -- Misc --
sampler s0 	  : register(s0);
sampler sDiff : register(s1);
float4 p0	  : register(c0);
float2 p1	  : register(c1);

#define width  (p0[0])
#define height (p0[1])

#define px (p1[0])
#define py (p1[1])

#define sqr(x) dot(x,x)
#define spread (exp(-(radius*radius)/2.0))

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

// -- Main Code --
float4 main(float2 tex : TEXCOORD0) : COLOR{
	float4 c0 = tex2D(s0, tex);

	float3 stab = 0;
	float W = 0;
	for (int i = -1; i <= 1; i++)
	for (int j = -1; j <= 1; j++) {
		float3 d = Get(0, 0) - Get(i, j);
		float x2 = sqr(acuity*d);
		float w = pow(spread, i*i + j*j)*lerp(1 / sqr(1 + x2), rsqrt(1 + x2), baseline);
		stab += d*w;
		W += w;
	}
	stab = (stab / W)*pow(W / (1 + 4 * spread + 4 * spread*spread), edge_adaptiveness - 1.0);

	//Calculate faithfulness force
	float3 diff = Diff(0, 0);

	//Apply forces
	c0.xyz -= strength*(diff + stab*softness);

	return c0;
}