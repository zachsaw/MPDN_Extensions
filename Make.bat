@echo off
setlocal
cd "%~dp0"

rmdir /q /s bin 1>nul 2>nul
rmdir /q /s obj 1>nul 2>nul
rmdir /q /s Release 1>nul 2>nul

mkdir Release 1>nul 2>nul

set buildPlatform=Release

for /D %%D in (%SYSTEMROOT%\Microsoft.NET\Framework\v4*) do set msbuild.exe=%%D\MSBuild.exe
if not defined msbuild.exe echo error: can't find MSBuild.exe & goto Quit
if not exist "%msbuild.exe%" echo error: %msbuild.exe%: not found & goto Quit

Echo Making MPDN Extensions...
Echo.
%msbuild.exe% Mpdn.Extensions.sln /m /p:Configuration=%buildPlatform% /v:q /t:rebuild
Echo.

if not "%ERRORLEVEL%"=="0" echo error: build failed & goto Quit

xcopy /y bin\Release\Mpdn.Extensions.dll Release\  1>nul 2>nul
xcopy /y bin\Release\ClearScript.dll Release\  1>nul 2>nul
xcopy /y bin\Release\ICSharpCode.AvalonEdit.dll Release\  1>nul 2>nul

Echo Completed successfully.
start Release\

:Quit
Echo.
Pause