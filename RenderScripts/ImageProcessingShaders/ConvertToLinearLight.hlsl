sampler s0 : register(s0);
float4  p0 : register(c0);
float2  p1 : register(c1);

#define width  (p0[0])
#define height (p0[1])

#define px (p1[0])
#define py (p1[1])

float4 main(float2 tex : TEXCOORD0) : COLOR 
{
    return float4(pow(tex2D(s0, tex).rgb, 1/2.2), 1);
}