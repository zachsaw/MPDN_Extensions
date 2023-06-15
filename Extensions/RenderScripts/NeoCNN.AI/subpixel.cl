#define pos(x,y,z) ((z) * height * width + (y) * width + (x))
#define pos_sp(x, y) ((y) * width * SCALE + (x))

__kernel __attribute__((vec_type_hint(float4)))
void run(__global const float* src, int width, int height, __global float* dst)
{
    int x = SCALE * WORKGROUP_SIZE_X * get_global_id(1);
    int y = SCALE * WORKGROUP_SIZE_Y * get_global_id(0);

    #pragma unroll
    for (int j=0; j<SCALE * WORKGROUP_SIZE_Y; j++)
    #pragma unroll
    for (int i=0; i<SCALE * WORKGROUP_SIZE_X; i++)
    {
        int ix = i + x;
        int iy = j + y;

        int a = ix / SCALE;
        int b = iy / SCALE;
        const int MAX_D = SCALE * SCALE - 1;
        int d = MAX_D - (SCALE * (ix % SCALE) + (iy % SCALE));

        dst[pos_sp(ix,iy)] = src[pos(a,b,d)];
    }
}