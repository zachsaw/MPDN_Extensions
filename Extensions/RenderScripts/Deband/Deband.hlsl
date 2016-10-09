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
sampler s0 : register(s0);
sampler s1 : register(s1);
float4 args0 : register(c2);
float4 size0 : register(c3);
float4 size1 : register(c4);
float4 sizeOutput : register(c5);

#define dxdy size0.zw
#define ddxddy size1.zw

#define acuity args0[0]
#define power  args0[1]

#define pi acos(-1)
#define sqr(x) ((x)*(x))

/* Noise between -0.5 and 0.5 */
float noise(float2 c0){
	for (int i = 0; i<2; i++) {
		sincos(mul(c0.xy, float2(12.9898,78.233)),c0.x,c0.y);
	    c0 = frac(c0 * 43758.5453);
	}
    return c0 - 0.5;
}

/* Noise between -0.5 and 0.5 */
float tempNoise(float2 tex, float t) 
{
	const float PI = acos(-1);
	float s = frac(t);
	s = s*s*(3-s*2);
	t = cosh(3)+floor(t);

	float4 rand1 = noise(float2(tex.x+1,(tex.y+2)+t/100));
	float4 rand2 = noise(float2(tex.x+1,(tex.y+2)+(t+1)/100));
	return atan(lerp(tan(rand1*PI), tan(rand2*PI), s))/PI;
}

// Input Processing
#define GetXY(xy) 	(tex2D(s1,ddxddy*(pos + 0.5 + (xy))))
#define Get(x,y)  	(GetXY(float2(x,y)))

#define GetHR(x,y)  	(tex2D(s0, tex + dxdy*float2(x, y)))

float4 main(float2 tex : TEXCOORD0) : COLOR {
	float4 c0 = tex2D(s0, tex);

	float2 pos = tex * size1.xy - 0.5;
	float2 offset = pos - clamp(floor(pos), 0, size1-2);
	pos -= offset;

	// Load input
	float4x4 X = {Get(0,0), Get(1,0), Get(0,1), Get(1,1)};
	
	// Use linear regression to interpolate
	float3x4 LinFit = {{-0.5, 0.5, -0.5, 0.5}, {-0.5, -0.5, 0.5, 0.5}, {0.25, 0.25, 0.25, 0.25}};
	float4 w = mul(float1x3(offset-0.5,1), LinFit);
	float4 avg = mul(w,X);

	// Statistical analysis
	float4x4 Y = (X - float4x4(c0, c0, c0, c0))*acuity;
	float3 SSres = sqr(mul(float4(0.5,-0.5,-0.5,0.5),Y).xyz); // Residual sum of squares
	float3 SStot = (sqr(Y[0].xyz) + sqr(Y[1].xyz) + sqr(Y[2].xyz) + sqr(Y[3].xyz)) - sqr(mul(float4(0.5,0.5,0.5,0.5),Y).xyz); // Total sum of squares
	float3 R = 1 - (SSres/SStot); // Coefficient of determination

	// Calculate variance of debanded value
	float3 varY = SSres * (1.0/4.0 + dot(offset-0.5, offset-0.5));

	float4x2 grad = transpose(mul(float2x4(LinFit[0], LinFit[1]), Y));
	float4 grad2 = float4(dot(grad[0], grad[0]), dot(grad[1], grad[1]), dot(grad[2], grad[2]), dot(grad[3], grad[3]));

	float3 m = (avg-c0)*acuity;
	float3 diff = 0.5 * sign(m) * max(0, abs(m)+1 - sqrt(sqr(abs(m)-1) + 4*varY)); // Maximum likelihood estimate

	// varY += 1e-10; // Add noise to avoid numerical issues
	// float3 p = 2 * saturate(1 - abs(diff)) * rsqrt(2*pi*varY) * exp(-0.5*sqr(diff - m)/varY); // Posterior Probability

#define PRESERVE_DETAIL
#ifdef PRESERVE_DETAIL
	diff -= abs(diff) * ((GetHR(1,0) + GetHR(-1,0) + GetHR(0,1) + GetHR(0,-1))/4.0 - c0) * acuity;
#endif

	c0.xyz = c0 + (diff / acuity) * (sqr(m)*grad2*(1 - power) <= power);

	// Dithering
#ifndef SkipDithering
	[branch] if (all(sizeOutput.xy == size0.xy)) {
		float noise = tempNoise(tex, p0[2]/4);
		noise = noise * sqrt(12) / acuity;
		c0.rgb += noise * sqrt(varX*str);
	}
#endif

	// Debugging
	// if (all(p0.xy == size0.xy)) {
	// 	c0 = round(c0*acuity)/acuity;
	// }
	
	return c0;
}