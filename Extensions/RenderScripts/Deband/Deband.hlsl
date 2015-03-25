sampler s0 : register(s0);
sampler s1 : register(s1);
float4 args0 : register(c2);
float4 args1 : register(c3);

#define acuity args0[0]
#define margin args0[1]

#define norm(x) (rsqrt(rsqrt(dot(x*x,x*x))))

#include "../Common/Colourprocessing.hlsl"
#define Kb args0[2] //redefinition
#define Kr args0[3] //redefinition

float4 main(float2 tex : TEXCOORD0) : COLOR {
	float4 c0 = tex2D(s0, tex);
	float4 avg = tex2D(s1, tex);
	
	float3 diff = avg - c0;
	diff -= clamp(diff, -0.5/acuity, 0.5/acuity);
	diff = mul(YUVtoRGB, diff);
	float thr = smoothstep(0, 0.5, margin - length(diff*acuity));
	c0.xyz -= clamp(c0 - avg, -thr/acuity, thr/acuity);
	
	return c0;
}