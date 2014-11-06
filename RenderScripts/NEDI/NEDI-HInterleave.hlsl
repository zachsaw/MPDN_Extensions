// $MinimumShaderProfile: ps_3_0
sampler s0 : register(s0);
sampler s1 : register(s1);
float4  p0 : register(c0);
float2  p1 : register(c1);

#define width  (p0[0])
#define height (p0[1])

#define px (p1[0])
#define py (p1[1])

float4 main(float2 tex : TEXCOORD0) : COLOR {
	int2 par = round(frac(tex*p0.xy/2.0));

	if (par.x == 0) { 
		return tex2D(s0,tex);
	} else { 
		return tex2D(s1,tex);
	}
}
