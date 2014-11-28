// -- Color space options --
#define GammaCurve sRGB
#define gamma 2.2

// -- Misc --
sampler s0 : register(s0);
sampler sY : register(s1);
sampler sU : register(s2);
sampler sV : register(s3);
float4 p0 :  register(c0);
float2 p1 :  register(c1);
float4 size2 : register(c2);

#define sizeUV size2

#define width  (p0[0])
#define height (p0[1])

#define px (sizeUV[2])
#define py (sizeUV[3])

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
float3 Labf(float3 x)   { return lerp(x * (29 * 29) / (3 * 6 * 6) + (4 / 29), pow(x, 1 / 3.0), saturate(0.5 + 65536 * (x - (6 * 6 * 6 / (29 * 29 * 29))))); }
float3 Labfinv(float3 x){ return lerp((x - 4 / 29) * (3 * 6 * 6) / (29 * 29), x*x*x, saturate(0.5 + 65536 * (x - (6 / 29)))); }

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
#define Position(x,y) (float2(px,py)*(floor(pos2)+float2(x,y)+0.5))
#define Get(x,y)	  (float2(tex2D(sU, Position(x,y))[0], tex2D(sV, Position(x,y))[0]))

// -- Main code --
float4 main(float2 tex : TEXCOORD0) : COLOR{
	float4 c0 = tex2D(s0, tex);

	//Calculate position
	int2 pos = floor(tex*p0.xy);
	float2 pos2 = (pos + 0.5) * size2.xy / p0.xy - 0.5;

	//Find extrema
	float2 Min = min(min(Get(0, 0), Get(1, 0)),
					 min(Get(0, 1), Get(1, 1)));
	float2 Max = max(max(Get(0, 0), Get(1, 0)),
					 max(Get(0, 1), Get(1, 1)));

	//Apply anti-ringing
	c0.yz = min(Max, max(Min, c0.yz));
	c0.x = tex2D(sY, tex)[0];

	return c0;
}