#define pos2_x(x) (clamp((x), 0, width-1))
#define pos2_y(y) ((clamp((y), 0, height-1)) * width)
#define pos_x(x) (x)
#define pos_y(y) ((y) * width)
#define pos_z(z) ((z) * width * height)

#define BLOCK_SIZE_X (FILTER_SIZE + WORKGROUP_SIZE_X)
#define BLOCK_SIZE_Y (FILTER_SIZE + WORKGROUP_SIZE_Y)

__kernel __attribute__((vec_type_hint(float4)))
void run(__global const float* src, int width, int height,
    __constant float* weights, __constant float* bias, __global float* dst)
{
    const int x = WORKGROUP_SIZE_X * get_global_id(1);
    const int y = WORKGROUP_SIZE_Y * get_global_id(0);

    if (x >= width || y >= height)
        return;

    const int x2 = x-FILTER_SIZE/2;
    const int y2 = y-FILTER_SIZE/2;

    // cache source pixels
    float pixels[KERNEL_DEPTH][BLOCK_SIZE_Y][BLOCK_SIZE_X];
    {
        #pragma unroll
        for (int z=0; z<KERNEL_DEPTH; z++)
        {
            int pz = pos_z(z);
            #pragma unroll
            for (int j=0; j<BLOCK_SIZE_Y; j++)
            {
                int py = pz + pos2_y(y2+j);
                #pragma unroll
                for (int i=0; i<BLOCK_SIZE_X; i++)
                {
                    pixels[z][j][i] = src[py+pos2_x(x2+i)];
                }
            }
        }
    }

    for (int z=0; z<OUTPUT_DEPTH; z++)
    {
        int k_idx_z = z * KERNEL_DEPTH * FILTER_SIZE * FILTER_SIZE;
        int pz = pos_z(z);

        #pragma unroll
        for (int iy=0; iy<WORKGROUP_SIZE_Y; iy++)
        {
            int py = pos_y(iy+y);
            #pragma unroll
            for (int ix=0; ix<WORKGROUP_SIZE_X; ix++)
            {
                float sum = 0.f;
                int k_idx = k_idx_z;
                #pragma unroll
                for (int k=0; k<KERNEL_DEPTH; k++)
                #pragma unroll
                for (int j=0; j<FILTER_SIZE; j++)
                #pragma unroll
                for (int i=0; i<FILTER_SIZE; i++)
                {
                    float pixel = pixels[k][j+iy][i+ix];
                    sum = mad(weights[k_idx++], pixel, sum);
                }

                int dst_pos_xy = py + pos_x(ix+x);
                dst[pz + dst_pos_xy] = max(sum + bias[z], 0.f);
            }
        }
    }
}