#define Kernel(x) pow(0.25, abs(x))
#define taps 3	
#define maxtaps taps

// -- Define horizontal convolver --
#define EntryPoint ScaleH
#define axis 0
#include "./Scalers/Convolver.hlsl"

// -- Define vertical convolver -- 
#define EntryPoint main
#define Get(pos) ScaleH(pos)
#define axis 1
#include "./Scalers/Convolver.hlsl"