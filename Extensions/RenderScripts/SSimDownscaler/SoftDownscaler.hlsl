#define pi acos(-1)

#define MN(B,C,x) (x <= 1.0 ? ((2-1.5*B-C)*x + (-3+2*B+C))*x*x + (1-B/3.) : (((-B/6.-C)*x + (B+5*C))*x + (-2*B-8*C))*x+((4./3.)*B+4*C))

// #define Kernel(x) 1 - abs(x)
// #define Kernel(x) MN(0.2620, 0.3690, abs(x))
// #define Kernel(x) MN(-1,0, abs(x))	
#define Kernel(x) exp(-2*x*x) // Gaussian
// #define Kernel(x) cos(0.5*pi*x) // Hann
// #define Kernel(x) exp(-abs(x)*2) // Poisson
// #define Kernel(x) (sin(pi*x)/(pi*x)) // Lanczos
// #define Kernel(x) dot(float3(0.42659, -0.49656, 0.076849), cos(float3(0,1,2)*pi*(x + 0.5*taps)*(2.0/taps))) // Blackmann
// #define Kernel(x) dot(float4(0.3635819, -0.4891775, 0.1365995, -0.0106411), cos(float4(0,1,2,3)*pi*(x+1))) // Nutall
// #define Kernel(x) dot(float4(0.355768, -0.487396, 0.144232, -0.012604), cos(float4(0,1,2,3)*pi*(x+1))) // Blackmann-Nutall
// #define Kernel(x) (1 + dot(float4(-1.93, 1.29, -0.388, 0.028), cos(float4(1,2,3,4)*pi*(x+1)))) // Flat-top

#define taps (2)

#include "./Scalers/Downscaler.hlsl"