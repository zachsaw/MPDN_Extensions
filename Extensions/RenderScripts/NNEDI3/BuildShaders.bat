@ ECHO OFF
FOR %%I IN (16,32,64,128,256) DO (
echo.
echo Compiling NNEDI3 %%I Neurons...
echo.
fxc.exe /nologo /D nns=%%I /T ps_5_0 /O3 /Fo nnedi3_%%I.cso nnedi3.hlsl
)
echo.
echo.
