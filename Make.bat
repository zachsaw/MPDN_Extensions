@echo off
REM if "%1"=="" (
REM   echo Usage: Make.bat [version x.y.z.rev] (e.g. Make.bat 1.2.3.456^)
REM   echo        where 'rev' is the commit count on GitHub as of this revision
  REM goto Quit
REM )

setlocal

cd "%~dp0"
git describe --abbrev=0 --tags > latestTag.txt
for /f "delims=" %%i in ('git rev-list HEAD --count') do set commitCount=%%i
set /p latestTag=<latestTag.txt
delete latestTag.txt
set releaseVersion=%latestTag%.%commitCount%

if not exist "MPDN\Mpdn.Core.dll" (
    echo Error: Make sure you've copied MPDN into the MPDN folder first!
    goto Quit
)

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
del Properties\AssemblyInfo.cs 1>nul 2>nul
del Extensions\PlayerExtensions\*.resx 1>nul 2>nul
del Extensions\RenderScripts\*.resx 1>nul 2>nul
call GenerateAssemblyInfo.bat %releaseVersion% > Properties\AssemblyInfo.cs
%msbuildexe% Mpdn.Extensions.sln /m /p:Configuration=%buildPlatform% /p:Platform="Any CPU" /v:q /t:rebuild
if not "%ERRORLEVEL%"=="0" (set builderror=1)
del Properties\AssemblyInfo.cs 1>nul 2>nul
call GenerateAssemblyInfo.bat 0.0.0.0 > Properties\AssemblyInfo.cs
Echo.

if "%builderror%"=="1" echo error: build failed & goto Quit

xcopy /y bin\Release\Mpdn.Extensions.dll Release\Extensions\  1>nul 2>nul

echo .cs\ > excludedfiles.txt
xcopy /y /e /exclude:excludedfiles.txt "Extensions\RenderScripts\*.*" "Release\Extensions\RenderScripts\" 1>nul 2>nul
del excludedfiles.txt

xcopy /y /e "Extensions\Libs\*.*" "Release\Extensions\Libs\" 1>nul 2>nul

if exist Sign.bat (
    echo Signing release...
    Echo.
    call Sign.bat Release\Extensions\Mpdn.Extensions.dll
)

echo Zipping release...
Echo.
cd "Release"
%zipper% a -r -tzip -mx9 "%~dp0Release\Mpdn.Extensions.zip" * > NUL
rmdir /q /s Extensions 1>nul 2>nul

cd "%~dp0"
call ./Installer/Make-Installer.bat
if not "%ERRORLEVEL%"=="0" echo error: make installer failed & goto Quit

echo.
Echo All operations completed successfully.
start Release

:Quit
Echo.
Pause