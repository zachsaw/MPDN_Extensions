sampler s0 : register(s0);
sampler s1 : register(s1);
sampler s2 : register(s2);
float4  p0 : register(c0);
float2  p1 : register(c1);
float4 size0 : register(c2);
float4 size1 : register(c3);
float4 size2 : register(c4);

#define width  (p0[0])
#define height (p0[1])

#define px (p1[0])
#define py (p1[1])

float4 main(float2 tex : TEXCOORD0) : COLOR
{
    return saturate(tex2D(s0, tex) + tex2D(s1, tex) - tex2D(s2, tex));
}