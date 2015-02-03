// Declarations

Texture2D inputTexture : register(t0);

cbuffer args : register(b0)
{
    float outWidth;
    float outHeight;
    float counter;
    float timeStamp;
};

cbuffer size0 : register(b1)
{
    float width;
    float height;
    float ppx;
    float ppy;
};

cbuffer args0 : register(b2)
{
    float r;
	float g;
	float b;
    float a;
};

SamplerState ss;

struct PS_IN
{
	float4 Position   : SV_POSITION;
	float2 Texture    : TEXCOORD0;
};

// Implementation

float4 main( PS_IN In ) : SV_TARGET
{
    float2 pos = In.Texture;
    return float4(saturate(inputTexture.Sample(ss, pos).rgb + float3(r, g, b)), 1);
}
