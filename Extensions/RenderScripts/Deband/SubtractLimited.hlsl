sampler s0 : register(s0);
sampler s1 : register(s1);
float4 args0 : register(c2);

#define acuity args0[0]
#define threshold args0[1]

#define norm(x) (rsqrt(rsqrt(dot(x*x,x*x))))

float4 main(float2 tex : TEXCOORD0) : COLOR {
	float4 c0 = tex2D(s0, tex);
	float4 c1 = tex2D(s1, tex);
	
	c0.rgb -= clamp(c1.xyz, -0.5/acuity, 0.5/acuity );

	return c0;
}