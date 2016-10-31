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
// -- Color space options --
#ifndef GammaCurve
    #define GammaCurve Rec709
#endif
#ifndef RGBSpace
    #define RGBSpace sRGB
#endif
#ifndef GammaCoeff
    #define GammaCoeff 2.2
#endif

#ifndef Kb
    #define Kb 0.0722
#endif
#ifndef Kr
    #define Kr 0.2126
#endif
#ifndef LimitedRange
    #define LimitedRange 1
#endif
#ifndef Range
    #define Range 255.0
#endif

#define MERGE(a, b) a##b
#define Gamma MERGE(GammaCurve, _Gamma)
#define GammaInv MERGE(GammaCurve, _GammaInv)

// -- Gamma processing --
float3 Rec709_Gamma(float3 x)   { return x < 0.018                      ? x * 4.506198600878514 : 1.099 * pow(x, 0.45) - 0.099; }
float3 Rec709_GammaInv(float3 x){ return x < 0.018 * 4.506198600878514  ? x / 4.506198600878514 : pow((x + 0.099) / 1.099, 1 / 0.45); }

float3 sRGB_Gamma(float3 x)   { return x < 0.00303993442528169  ? x * 12.9232102 : 1.055*pow(x, 1 / 2.4) - 0.055; }
float3 sRGB_GammaInv(float3 x){ return x < 0.039285711572131475 ? x / 12.9232102 : pow((x + 0.055) / 1.055, 2.4); }

float3 Power_Gamma(float3 x)   { return pow(saturate(x), 1 / GammaCoeff); }
float3 Power_GammaInv(float3 x){ return pow(saturate(x), GammaCoeff); }

float3 Fast_Gamma(float3 x)   { return saturate(x)*rsqrt(saturate(x)); }
float3 Fast_GammaInv(float3 x){ return x*x; }

float3 None_Gamma(float3 x)   { return x; }
float3 None_GammaInv(float3 x){ return x; }

// -- Colour space Processing --

// Primaries

float3 Cramer3x3(float3x3 A,float3 b) {
    float3 x;
    A = transpose(A);
    x[0] = determinant(float3x3(b,A[1],A[2]));
    x[1] = determinant(float3x3(A[0],b,A[2]));
    x[2] = determinant(float3x3(A[0],A[1],b));
    return x/determinant(A);
}

float3x3 MatrixInverse3x3(float3x3 A) {
    A = transpose(A);
    float3x3 B = {  Cramer3x3(A,float3(1,0,0)),
                    Cramer3x3(A,float3(0,1,0)),
                    Cramer3x3(A,float3(0,0,1))};
    return B;
}

float3x3 Calc_RGBtoXYZmatrix(float4 x, float4 y) {
    float4 z = 1-x-y;
    float3x3 C = {x.rgb/y.rgb, 1,1,1 ,z.rgb/y.rgb};
    float3 S = Cramer3x3(C, float3(x.w/y.w,1,z.w/y.w));
    float3x3 T = float3x3(S*C[0],S*C[1],S*C[2]);
    return T;
}

float3x3 Calc_XYZtoRGBmatrix(float4 x, float4 y) { return MatrixInverse3x3(Calc_RGBtoXYZmatrix(x,y)); }

#define sRGBx float4(0.64, 0.3, 0.15, 0.312713)
#define sRGBy float4(0.33, 0.6, 0.06, 0.329016)

#define SMPETCx float4(0.630, 0.310, 0.155, 0.312713)
#define SMPETCy float4(0.340, 0.595, 0.070, 0.329016)

#define REC2020x float4(0.708, 0.170, 0.131, 0.312713)
#define REC2020y float4(0.292, 0.797, 0.046, 0.329016)

#define RGBtoXYZmatrix(space) Calc_RGBtoXYZmatrix(space##x,space##y)
#define XYZtoRGBmatrix(space) Calc_XYZtoRGBmatrix(space##x,space##y)

#define RGBtoXYZ RGBtoXYZmatrix(RGBSpace)
#define XYZtoRGB XYZtoRGBmatrix(RGBSpace)

// YUV

#define RGBtoYUV float3x3(float3(Kr, 1 - Kr - Kb, Kb), float3(-Kr, Kr + Kb - 1, 1 - Kb) / (2*(1 - Kb)), float3(1 - Kr, Kr + Kb - 1, -Kb) / (2*(1 - Kr)))
#define YUVtoRGB float3x3(float3(1, 0, 2*(1 - Kr)), float3(Kb + Kr - 1, 2*(1 - Kb)*Kb, 2*Kr*(1 - Kr)) / (Kb + Kr - 1), float3(1, 2*(1 - Kb),0))

#define YUVtoXYZ mul(RGBtoXYZ, YUVtoRGB)
#define XYZtoYUV mul(RGBtoYUV, XYZtoRGB)

float3 Labf(float3 x)   { return x < (6.0*6.0*6.0) / (29.0*29.0*29.0) ? (x * (29.0 * 29.0) / (3.0 * 6.0 * 6.0)) + (4.0 / 29.0) : pow(x, 1.0 / 3.0); }
float3 Labfinv(float3 x){ return x < (6.0 / 29.0)                      ? (x - (4.0 / 29.0)) * (3.0 * 6.0 * 6.0) / (29.0 * 29.0) : x*x*x; }

float3 DLabf(float3 x)   { return min((29.0 * 29.0) / (3.0 * 6.0 * 6.0), (1.0/3.0) / pow(x, (2.0 / 3.0))); }
float3 DLabfinv(float3 x){ return max((3.0 * 6.0 * 6.0) / (29.0 * 29.0), 3.0*x*x); }

float3 RGBtoLab(float3 rgb) {    
    float3 xyz = mul(RGBtoXYZ, rgb);
    xyz = Labf(xyz);
    return float3(1.16*xyz.y - 0.16, 5.0*(xyz.x - xyz.y), 2.0*(xyz.y - xyz.z));
}

// Lab

float3 LabtoRGB(float3 lab) {
    float3 xyz = (lab.x + 0.16) / 1.16 + float3(lab.y / 5.0, 0, -lab.z / 2.0);
    return mul(XYZtoRGB, Labfinv(xyz));
}

float3x3 DRGBtoLab(float3 rgb) {
    float3 xyz = mul(RGBtoXYZ, rgb);
    xyz = DLabf(xyz);
    float3x3 D = { { xyz.x, 0, 0 }, { 0, xyz.y, 0 }, { 0, 0, xyz.z } };
    return mul(D, RGBtoXYZ);
}

float3x3 DLabtoRGB(float3 lab) {
    float3 xyz = (lab.x + 0.16) / 1.16 + float3(lab.y / 5.0, 0, -lab.z / 2.0);
    xyz = DLabfinv(xyz);
    float3x3 D = { { xyz.x, 0, 0 }, { 0, xyz.y, 0 }, { 0, 0, xyz.z } };
    return mul(XYZtoRGB, D);
}

float3x3 DinvRGBtoLab(float3 lab) {
    float3 xyz = (lab.x + 0.16) / 1.16 + float3(lab.y / 5.0, 0, -lab.z / 2.0);
    xyz = 1 / DLabfinv(xyz);
    float3x3 D = { { xyz.x, 0, 0 }, { 0, xyz.y, 0 }, { 0, 0, xyz.z } };
    return mul(XYZtoRGB, D);
}

float3x3 DinvLabtoRGB(float3 rgb) {
    float3 xyz = mul(RGBtoXYZ, rgb);
    xyz = 1 / DLabf(xyz);
    float3x3 D = { { xyz.x, 0, 0 }, { 0, xyz.y, 0 }, { 0, 0, xyz.z } };
    return mul(D, RGBtoXYZ);
}

// Utility / Reference functions

float3 LimitChroma(float3 rgb) {
	float3 Y = RGBtoYUV[0];
	float3 S = saturate(rgb);
	float3 X = dot(Y,rgb - S)*(rgb - S) > 0 ? 0 : S;
	return S + X*dot(Y,rgb - S)/max(1e-6, dot(Y,X));
}

// WIP
// float3 LimitChroma(float3 rgb) {
//     float3 Y = RGBtoYUV[0];
//     float3 C = saturate(dot(rgb, Y));

//     /* unroll first iteration */{
//         float3 D = rgb - C;
//         float3 y = Y;

//         D -= y * dot(D, Y) / dot(y, Y);

//         float3 L = abs(D) > 1e-6 ? (D > 0 ? (1 - C) : -C) / D : 1;
//         C += D * saturate(min(L[0], min(L[1], L[2])));
//     }

//     /* second iteration */  {
//         float3 D = rgb - C;
//         float3 y = Y;
//         bool3 Active = abs(C - round(C)) < 1e-6;

//         D = (Active ? 0 : D);
//         y = (Active ? 0 : y);

//         if (dot(y,Y) != 0)
//             D -= y * dot(D, Y) / dot(y, Y);

//         float3 L = abs(D) > 1e-6 ? saturate((D > 0 ? (1 - C) : -C) / D) : 1;
//         C += D * min(L[0], min(L[1], L[2]));
//     }

//     return C;
// }

float Luma(float3 rgb) {
	return dot(RGBtoYUV[0], rgb);
}

float3 ConvertToYUV(float3 rgb) {
    float midpoint = 0.5 + 0.5/Range;
    float3 yuv = mul(RGBtoYUV, rgb);
	if (LimitedRange == 0)
		return yuv + float3(0,midpoint,midpoint);
	else
		return yuv*float3(219.0, 224.0, 224.0)/255.0 + float3(16.0/255.0,midpoint,midpoint);
}

float3 ConvertToRGB(float3 yuv) {
    float midpoint = 0.5 + 0.5/Range;
	if (LimitedRange == 0)
		yuv = yuv - float3(0,midpoint,midpoint);
	else
		yuv = (yuv - float3(16.0/255.0,midpoint,midpoint))*255.0/float3(219.0, 224.0, 224.0);
    return mul(YUVtoRGB, yuv);
}