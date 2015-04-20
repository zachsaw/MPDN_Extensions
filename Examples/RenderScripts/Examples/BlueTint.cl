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
// Use arguments for color
const sampler_t smp = CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void BlueTint(__write_only image2d_t dst, __read_only image2d_t src, float r, float g, float b)
{
    const int2 pos = { get_global_id(0), get_global_id(1) };
    float3 color = { r, g, b };
    float3 val = clamp(read_imagef(src, smp, pos).xyz + color, 0.0f, 1.0f);
    float4 result = { val, 1.0f };
    write_imagef(dst, pos, result);
}

/*
// Use a buffer for color
const sampler_t smp = CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void BlueTint(__write_only image2d_t dst, __read_only image2d_t src, __global const float* restrict c)
{
    const int2 pos = { get_global_id(0), get_global_id(1) };
    float3 color = { c[0], c[1], c[2] };
    float3 val = clamp(read_imagef(src, smp, pos).xyz + color, 0.0f, 1.0f);
    float4 result = { val, 1.0f };
    write_imagef(dst, pos, result);
}
*/

/*
// Use a float4 for color
const sampler_t smp = CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void BlueTint(__write_only image2d_t dst, __read_only image2d_t src, float4 c)
{
    const int2 pos = { get_global_id(0), get_global_id(1) };
    float3 color = { c.s0, c.s1, c.s2 };
    float3 val = clamp(read_imagef(src, smp, pos).xyz + color, 0.0f, 1.0f);
    float4 result = { val, 1.0f };
    write_imagef(dst, pos, result);
}
*/
