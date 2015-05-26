sampler s0 : register(s0);
sampler s1 : register(s1);
float4  p0 : register(c0);
float2  p1 : register(c1);
float4 size0 : register(c2);
float4 size1 : register(c3);

#define width  (p0[0])
#define height (p0[1])

#define px (p1[0])
#define py (p1[1])

#define dxdy0 (size0.zw)
#define dxdy1 (size1.zw)

float4 main(float2 tex : TEXCOORD0) : COLOR{
    float y = tex2D(s0, tex).x;
    float2 uv = tex2D(s1, tex).yz;
    
    return float4(y, uv, 1);
}
