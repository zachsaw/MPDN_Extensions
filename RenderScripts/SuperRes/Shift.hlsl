// -- Misc --
sampler s0 : register(s0);
sampler s1 : register(s1);
float4 p0 :  register(c0);
float2 p1 :  register(c1);

#define width  (p0[0])
#define height (p0[1])

#define px (p1[0])
#define py (p1[1])

// -- Main code --
float4 main(float2 tex : TEXCOORD0) : COLOR {
	float4 c0 = tex2D(s0, tex - 0.5*float2(px,py));

	return c0;
}
