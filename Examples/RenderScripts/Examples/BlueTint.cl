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
// Use a struct for color (structs don't work on Intel!)
const sampler_t smp = CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

typedef struct tag_color
{
    float r;
    float g;
    float b;
} color_t;

__kernel void BlueTint(__write_only image2d_t dst, __read_only image2d_t src, __global color_t* restrict c)
{
    const int2 pos = { get_global_id(0), get_global_id(1) };
    float3 color = { c->r, c->g, c->b };
    float3 val = clamp(read_imagef(src, smp, pos).xyz + color, 0.0f, 1.0f);
    float4 result = { val, 1.0f };
    write_imagef(dst, pos, result);
}
*/
