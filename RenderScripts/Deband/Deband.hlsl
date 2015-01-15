sampler s0 : register(s0);
sampler s1 : register(s1);
float4 args0 : register(c3);

#define acuity args0[0]
#define threshold args0[1]
#define margin args0[2]

float4 main(float2 tex : TEXCOORD0) : COLOR {
	float4 c0 = tex2D(s0, tex);
	float4 avg = tex2D(s1, tex);

	// maximum at 0.5 + margin * (sqrt(57) - 5) / 16 ~= 0.5 + margin*0.159365
	c0.rgb = lerp(c0, avg, smoothstep(0, margin, threshold + margin - abs((avg - c0).rgb*acuity)));

	return c0;
}