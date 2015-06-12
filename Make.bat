@echo off
if "%1"=="" (
  echo Usage: Make.bat [version x.y.z.rev] (e.g. Make.bat 1.2.3.456^)
  echo        where 'rev' is the commit count on GitHub as of this revision
  goto Quit
)

setlocal

cd "%~dp0"
set releaseVersion=%1

rmdir /q /s bin 1>nul 2>nul
rmdir /q /s obj 1>nul 2>nul
rmdir /q /s Release 1>nul 2>nul

mkdir Release 1>nul 2>nul

set buildPlatform=Release

set zipper="%ProgramFiles%\7-zip\7z.exe"
if not exist %zipper% (
  echo Error: 7-zip (native version^) is not installed
  goto Quit
)

for /D %%D in (%SYSTEMROOT%\Microsoft.NET\Framework\v4*) do set msbuildexe=%%D\MSBuild.exe
if not defined msbuildexe echo error: can't find MSBuild.exe & goto Quit
if not exist "%msbuildexe%" echo error: %msbuildexe%: not found & goto Quit

Echo Making MPDN Extensions...
Echo.
call GenerateAssemblyInfo.bat %releaseVersion% > Properties\AssemblyInfo.cs
%msbuildexe% Mpdn.Extensions.sln /m /p:Configuration=%buildPlatform% /v:q /t:rebuild
del Properties\AssemblyInfo.cs
Echo.

if not "%ERRORLEVEL%"=="0" echo error: build failed & goto Quit

xcopy /y bin\Release\Mpdn.Extensions.dll Release\Extensions\  1>nul 2>nul

echo .cs\ > excludedfiles.txt
xcopy /y /e /exclude:excludedfiles.txt "Extensions\RenderScripts\*.*" "Release\Extensions\RenderScripts\" 1>nul 2>nul
del excludedfiles.txt

xcopy /y /e "Extensions\Libs\*.*" "Release\Extensions\Libs\" 1>nul 2>nul

echo Zipping up...
Echo.
cd "Release"
%zipper% a -r -tzip -mm=lzma -mx9 "%~dp0Release\Mpdn.Extensions.zip" * > NUL
rmdir /q /s Extensions 1>nul 2>nul

Echo Completed successfully.
start .

:Quit
Echo.
Pause