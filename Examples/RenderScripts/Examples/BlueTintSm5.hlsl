// This file is a part of MPDN Extensions.
// https://github.com/zachsaw/MPDN_Extensions
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
// 
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public
// License along with this library.
// 
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
