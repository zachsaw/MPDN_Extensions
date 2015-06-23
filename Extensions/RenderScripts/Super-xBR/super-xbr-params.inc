float4 args0 : register(c2);
float4 size0 : register(c3);

#define XBR_EDGE_STR args0[0]
#define XBR_WEIGHT args0[1]

#define input_size (size0.xy)
#define pixel_size (size0.zw)