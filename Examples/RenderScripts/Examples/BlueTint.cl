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