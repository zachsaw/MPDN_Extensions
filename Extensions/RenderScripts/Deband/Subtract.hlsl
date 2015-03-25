sampler s0 : register(s0);
sampler s1 : register(s1);

float4 main(float2 tex : TEXCOORD0) : COLOR {
	float4 c0 = tex2D(s0, tex);
	float4 c1 = tex2D(s1, tex);

	c0.xyz -= c1.xyz;

	return c0;
}