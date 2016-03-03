#pragma parameter XBR_EDGE_STR "Xbr - Edge Strength p0" 1.0 0.0 5.0 0.2
#pragma parameter XBR_WEIGHT "Xbr - Filter Weight" 1.0 0.00 1.50 0.01

#include "super-xbr-params.inc"

/* COMPATIBILITY
	 - HLSL compilers
	 - Cg   compilers
*/

/*
	 
	*******  Super XBR Shader  *******
	 
	Copyright (c) 2015 Hyllian - sergiogdb@gmail.com

	Permission is hereby granted, free of charge, to any person obtaining a copy
	of this software and associated documentation files (the "Software"), to deal
	in the Software without restriction, including without limitation the rights
	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
	copies of the Software, and to permit persons to whom the Software is
	furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in
	all copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
	THE SOFTWARE.

*/

#if Pass == 0
	#define wp1   1.0
	#define wp2   0.0
	#define wp3   0.0
	#define wp4   2.0
	#define wp5  -1.0
	#define wp6   0.0
#elif Pass == 1
	#define wp1   1.0
	#define wp2   0.0
	#define wp3   0.0
	#define wp4   4.0
	#define wp5   0.0
	#define wp6   0.0
#else
	#define wp1   1.0
	#define wp2   0.0
	#define wp3   0.0
	#define wp4   0.0
	#define wp5  -1.0
	#define wp6   0.0
#endif

#if Pass == 1
	#define weight1 (XBR_WEIGHT*1.75068/10.0)
	#define weight2 (XBR_WEIGHT*1.29633/10.0/2.0)
#else
	#define weight1 (XBR_WEIGHT*1.29633/10.0)
	#define weight2 (XBR_WEIGHT*1.75068/10.0/2.0)
#endif

#if Pass == 0
	#define Get(x,y) (tex2D(s0, VAR.texCoord + pixel_size*float2(x,y)).xyz)
#elif Pass == 1
	#define Get(x,y) (tex2D(s0, VAR.texCoord + pixel_size*float2((x)+(y) - 1,(y) - (x))).xyz)
#else
	#define Get(x,y) (tex2D(s0, VAR.texCoord - pixel_size*float2(x,y)).xyz)
#endif

const static float3 Y = float3(.2126, .7152, .0722);

float RGBtoYUV(float3 color)
{
	return dot(color, Y);
}

float df(float A, float B)
{
	return abs(A-B);
}

float d_wd(float b0, float b1, float c0, float c1, float c2, float d0, float d1, float d2, float d3, float e1, float e2, float e3, float f2, float f3)
{
	return (wp1*(df(c1,c2) + df(c1,c0) + df(e2,e1) + df(e2,e3)) + wp2*(df(d2,d3) + df(d0,d1)) + wp3*(df(d1,d3) + df(d0,d2)) + wp4*df(d1,d2) + wp5*(df(c0,c2) + df(e1,e3)) + wp6*(df(b0,b1) + df(f2,f3)));
}

float hv_wd(float i1, float i2, float i3, float i4, float e1, float e2, float e3, float e4)
{
	return ( wp4*(df(i1,i2)+df(i3,i4)) + wp1*(df(i1,e1)+df(i2,e2)+df(i3,e3)+df(i4,e4)) + wp3*(df(i1,e2)+df(i3,e4)+df(e1,i2)+df(e3,i4)));
}

float3 min4(float3 a, float3 b, float3 c, float3 d)
{
		return min(a, min(b, min(c, d)));
}
float3 max4(float3 a, float3 b, float3 c, float3 d)
{
		return max(a, max(b, max(c, d)));
}

struct input
{
		float2 video_size;
		float2 texture_size;
		float2 output_size;
		float  frame_count;
		float  frame_direction;
		float frame_rotation;
};
 
 
struct out_vertex {
		float4 position : POSITION;
		float4 color    : COLOR;
		float2 texCoord : TEXCOORD0;
};
 
/*    VERTEX_SHADER    */
out_vertex main_vertex
(
		float4 position   : POSITION,
		float4 color      : COLOR,
		float2 texCoord1  : TEXCOORD0,
 
		uniform float4x4 modelViewProj,
		uniform input IN
)

{
	float2 tex = texCoord1;

	out_vertex OUT = {
			mul(modelViewProj, position),
			color,
			tex
	};
		 
	return OUT;
}
 
float4 main_fragment(in out_vertex VAR, uniform sampler2D s0 : TEXUNIT0, uniform input IN) : COLOR
{
	//Skip pixels on wrong grid
#if Pass==0
	if (any(frac(VAR.texCoord*input_size)<(0.5))) return tex2D(s0, VAR.texCoord);
#elif Pass==1
	float2 dir = frac(VAR.texCoord*input_size/2.0) - (0.5);
	if ((dir.x*dir.y)>0.0) return tex2D(s0, VAR.texCoord);
#endif

	float3 P0 = Get(-1,-1);
	float3 P1 = Get( 2,-1);
	float3 P2 = Get(-1, 2);
	float3 P3 = Get( 2, 2);

	float3  B = Get( 0,-1);
	float3  C = Get( 1,-1);
	float3  D = Get(-1, 0);
	float3  E = Get( 0, 0);
	float3  F = Get( 1, 0);
	float3  G = Get(-1, 1);
	float3  H = Get( 0, 1);
	float3  I = Get( 1, 1);

	float3 F4 = Get(2, 0);
	float3 I4 = Get(2, 1);
	float3 H5 = Get(0, 2);
	float3 I5 = Get(1, 2);

	float b = RGBtoYUV( B );
	float c = RGBtoYUV( C );
	float d = RGBtoYUV( D );
	float e = RGBtoYUV( E );
	float f = RGBtoYUV( F );
	float g = RGBtoYUV( G );
	float h = RGBtoYUV( H );
	float i = RGBtoYUV( I );

	float i4 = RGBtoYUV( I4 ); float p0 = RGBtoYUV( P0 );
	float i5 = RGBtoYUV( I5 ); float p1 = RGBtoYUV( P1 );
	float h5 = RGBtoYUV( H5 ); float p2 = RGBtoYUV( P2 );
	float f4 = RGBtoYUV( F4 ); float p3 = RGBtoYUV( P3 );

/*
								  P1
		 |P0|B |C |P1|         C     F4          |a0|b1|c2|d3|
		 |D |E |F |F4|      B     F     I4       |b0|c1|d2|e3|   |e1|i1|i2|e2|
		 |G |H |I |I4|   P0    E  A  I     P3    |c0|d1|e2|f3|   |e3|i3|i4|e4|
		 |P2|H5|I5|P3|      D     H     I5       |d0|e1|f2|g3|
							   G     H5
								  P2
*/

	/* Calc edgeness in diagonal directions. */
	float d_edge  = (d_wd( d, b, g, e, c, p2, h, f, p1, h5, i, f4, i5, i4 ) - d_wd( c, f4, b, f, i4, p0, e, i, p3, d, h, i5, g, h5 ));

	/* Calc edgeness in horizontal/vertical directions. */
	float hv_edge = (hv_wd(f, i, e, h, c, i5, b, h5) - hv_wd(e, f, h, i, d, f4, g, i4));

	/* Filter weights. Two taps only. */
	float4 w1 = float4(-weight1, weight1+0.5, weight1+0.5, -weight1);
	float4 w2 = float4(-weight2, weight2+0.25, weight2+0.25, -weight2);

	/* Filtering and normalization in four direction generating four colors. */
	float3 c1 = mul(w1, float4x3(P2, H, F, P1));
	float3 c2 = mul(w1, float4x3(P0, E, I, P3));
	float3 c3 = (mul(w2, float4x3( D, E, F, F4)) + mul(w2, float4x3( G, H, I, I4)));
	float3 c4 = (mul(w2, float4x3( C, F, I, I5)) + mul(w2, float4x3( B, E, H, H5)));

#ifndef FAST_METHOD
	float limits = XBR_EDGE_STR + 0.000001;
	float edge_strength = smoothstep(0.0, limits, abs(d_edge));

	/* Smoothly blends the two strongest directions (one in diagonal and the other in vert/horiz direction). */
	float3 color =  lerp(lerp(c1, c2, step(0.0, d_edge)), lerp(c3, c4, step(0.0, hv_edge)), 1 - edge_strength); 
#else
	float limits = XBR_EDGE_STR + 0.000001;
	float edge_strength = smoothstep(-limits, limits, d_edge);

	/* Smoothly blends the two directions according to edge strength. */
	float3 color =  lerp(c1, c2, edge_strength);
#endif

	/* Anti-ringing code. */
	float3 min_sample = min4( E, F, H, I ) + lerp((P2-H)*(F-P1), (P0-E)*(I-P3), step(0.0, d_edge));
	float3 max_sample = max4( E, F, H, I ) - lerp((P2-H)*(F-P1), (P0-E)*(I-P3), step(0.0, d_edge));
	color = clamp(color, min_sample, max_sample);

	return float4(color, 1.0);
}