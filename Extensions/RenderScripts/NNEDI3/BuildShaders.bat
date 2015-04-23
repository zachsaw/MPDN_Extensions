@ ECHO OFF
FOR %%I IN (16,32,64,128,256) DO (
echo.
echo Compiling NNEDI3 %%I Neurons...
echo.
fxc.exe /nologo /D nns=%%I /T ps_5_0 /O3 /Fo nnedi3_%%I_A.cso nnedi3.hlsl
fxc.exe /nologo /D nns=%%I /D VECTOR_DOT=1 /T ps_5_0 /O3 /Fo nnedi3_%%I_B.cso nnedi3.hlsl
fxc.exe /nologo /D nns=%%I /D VECTOR_DOT=1 /D UNROLLED=1 /T ps_5_0 /O3 /Fo nnedi3_%%I_C.cso nnedi3.hlsl
)
echo.
echo.
