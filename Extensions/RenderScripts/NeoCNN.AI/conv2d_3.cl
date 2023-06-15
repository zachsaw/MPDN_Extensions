#define pos2_x(x) (clamp((x), 0, width-1))
#define pos2_y(y) ((clamp((y), 0, height-1)) * width)
#define pos_x(x) (x)
#define pos_y(y) ((y) * width)
#define pos_z(z) ((z) * width * height)

#define HALF_FILTER_SIZE (FILTER_SIZE/2)
#define BLOCK_SIZE_X (FILTER_SIZE + WORKGROUP_SIZE_X)
#define BLOCK_SIZE_Y (FILTER_SIZE + WORKGROUP_SIZE_Y)

void calc_pixels(__global const float* src, float pixels[BLOCK_SIZE_Y][BLOCK_SIZE_X],
    __constant float* weights, float bias, int x2, int y2, int width, int height, int k)
{
    {
        int pk = k * PREV_KERNEL_DEPTH;
        float4 w = {weights[pk+0], weights[pk+1], weights[pk+2], weights[pk+3]};

        int pz0 = pos_z(0);
        int pz1 = pos_z(1);
        int pz2 = pos_z(2);
        int pz3 = pos_z(3);

        #pragma unroll
        for (int iyp=0; iyp<BLOCK_SIZE_Y; iyp++)
        {
            int py = pos2_y(y2+iyp);
            #pragma unroll
            for (int ixp=0; ixp<BLOCK_SIZE_X; ixp++)
            {
                int px = pos2_x(x2+ixp);
                int pxy = py + px;
                float4 pixel = {src[pz0 + pxy], src[pz1 + pxy], src[pz2 + pxy], src[pz3 + pxy]};
                pixels[iyp][ixp] = dot(w, pixel);
            }
        }
    }

    #pragma unroll
    for (int pk=4; pk<PREV_KERNEL_DEPTH; pk++)
    {
        int pz = pos_z(pk);
        float w = weights[k * PREV_KERNEL_DEPTH + pk];
        #pragma unroll
        for (int iyp=0; iyp<BLOCK_SIZE_Y; iyp++)
        {
            int py = pos2_y(y2+iyp);
            int pzy = pz + py;
            #pragma unroll
            for (int ixp=0; ixp<BLOCK_SIZE_X; ixp++)
            {
                float pixel = src[pzy + pos2_x(x2+ixp)];
                pixels[iyp][ixp] = mad(w, pixel, pixels[iyp][ixp]);
            }
        }
    }

    #pragma unroll
    for (int iyp=0; iyp<BLOCK_SIZE_Y; iyp++)
    #pragma unroll
    for (int ixp=0; ixp<BLOCK_SIZE_X; ixp++)
        pixels[iyp][ixp] = max(pixels[iyp][ixp] + bias, 0.f);
}

__kernel __attribute__((vec_type_hint(float4)))
void run(__global const float* src,
    __constant float* prev_weights, __constant float* prev_bias,
    int width, int height, __constant float* weights, __constant float* bias,
    __global float* dst)
{
    const int x = WORKGROUP_SIZE_X * get_global_id(1);
    const int y = WORKGROUP_SIZE_Y * get_global_id(0);

    if (x >= width || y >= height)
        return;

    const int x2 = x-FILTER_SIZE/2;
    const int y2 = y-FILTER_SIZE/2;

    float sums[OUTPUT_DEPTH][WORKGROUP_SIZE_Y][WORKGROUP_SIZE_X];
    #pragma unroll
    for (int z=0; z<OUTPUT_DEPTH; z++)
    #pragma unroll
    for (int iy=0; iy<WORKGROUP_SIZE_Y; iy++)
    #pragma unroll
    for (int ix=0; ix<WORKGROUP_SIZE_X; ix++)
        sums[z][iy][ix] = 0.f;

    float pixels[BLOCK_SIZE_Y][BLOCK_SIZE_X];
    for (int k=0; k<KERNEL_DEPTH; k++)
    {
        // calculate and cache source pixels
        calc_pixels(src, pixels, prev_weights, prev_bias[k], x2, y2, width, height, k);

        #pragma unroll
        for (int iy=0; iy<WORKGROUP_SIZE_Y; iy++)
        #pragma unroll
        for (int ix=0; ix<WORKGROUP_SIZE_X; ix++)
        #pragma unroll
        for (int z=0; z<OUTPUT_DEPTH; z++)
        {
            int w_idx = (k + z * KERNEL_DEPTH) * FILTER_SIZE * FILTER_SIZE;
            #pragma unroll
            for (int j=0; j<FILTER_SIZE; j++)
            #pragma unroll
            for (int i=0; i<FILTER_SIZE; i++)
            {
                float pixel = pixels[iy+j][ix+i];
                float w = weights[w_idx++];
                float* sum_ptr = &sums[z][iy][ix];
                *sum_ptr = mad(w, pixel, *sum_ptr);
            }
        }
    }

    #pragma unroll
    for (int z=0; z<OUTPUT_DEPTH; z++)
    {
        int pz = pos_z(z);
        #pragma unroll
        for (int iy=0; iy<WORKGROUP_SIZE_Y; iy++)
        {
            int py = pos_y(iy+y);
            int pzy = pz + py;
            #pragma unroll
            for (int ix=0; ix<WORKGROUP_SIZE_X; ix++)
                dst[pzy + pos_x(ix+x)] = max(sums[z][iy][ix] + bias[z], 0.f);
        }
    }
}
