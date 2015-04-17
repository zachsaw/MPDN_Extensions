sampler s0 : register(s0);
sampler s1 : register(s1);
float4 p0 : register(c0);
float4 p1 : register(c1);
float4 args0 : register(c2);
float4 size1 : register(c3);

#define ppx (size1[2])
#define ppy (size1[3])

#define acuity args0[0]
#define threshold args0[1]

#include "../Common/Colourprocessing.hlsl"
#define Kb args0[2] //redefinition
#define Kr args0[3] //redefinition

#define sqr(x) dot(x,x)
#define norm(x) (rsqrt(rsqrt(sqr(sqr(x)))))

// Input Processing
#define Get(x,y)  	(tex2D(s1,float2(ppx,ppy)*(pos + 0.5 + float2(x,y))).xyz)

float4 main(float2 tex : TEXCOORD0) : COLOR {
	float4 c0 = tex2D(s0, tex);

	float2 pos = tex * size1.xy - 0.5;
	float2 offset = frac(pos);
	pos -= offset;

	float4x3 X = {Get(0,0), Get(1,0), Get(0,1), Get(1,1)};
	
	float4 w = {(1-offset.x)*(1-offset.y), offset.x*(1-offset.y), (1-offset.x)*offset.y, offset.x*offset.y };
	for (int i = 0; i < 4; i++) {
		float3 d = X[i] - c0;
		d -= clamp(d, -2.0/acuity, 2.0/acuity);
		//d = mul(YUVtoRGB, d);
		w[i] *= smoothstep(0.1, 1.0, 4*threshold - length(d*acuity));
	}
	
	float3 avg = mul(float1x4(w),X)/dot(w,1);

	float3 diff = avg - c0;
	diff -= clamp(diff, -0.5/acuity, 0.5/acuity);
	//diff = mul(YUVtoRGB, diff);
	float str = smoothstep(0, 0.5, threshold - length(diff*acuity));
	c0.xyz = lerp(c0, avg, str);

	return c0;
}