// Declarations

RWTexture2D<float4> Output;

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

// Implementation

[numthreads(32, 32, 1)]
void main(uint3 threadID : SV_DispatchThreadID)
{
    float2 pos = threadID.xy/float2(outWidth, outHeight);
    Output[threadID.xy] = float4(saturate(inputTexture.SampleLevel(ss, pos, 0).rgb + float3(r, g, b)), 1);
}
