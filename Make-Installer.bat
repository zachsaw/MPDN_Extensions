@echo off
setlocal
set releaseVer=%1
if "%releaseVer%"=="" (
  echo Error: Please specify version: Make-Installer <version> ^^^(e.g. Make-Installer 1.6.2^^^)
  goto Quit
)
cd "%~dp0"
if not exist "Release\Mpdn.Extensions.zip" (
  echo Error: Mpdn.Extensions.zip is not found - run Make.bat first!
  goto Quit
)
set zipper="%ProgramFiles%\7-zip\7z.exe"
if not exist %zipper% (
  echo Error: 7-zip ^^^(native version^^^) is not installed
  goto Quit
)
IF "%PROCESSOR_ARCHITECTURE%"=="x86" (set progfiles86=C:\Program Files) else (set progfiles86=C:\Program Files ^(x86^))
set makensis="%progfiles86%\NSIS\Bin\makensis.exe"
if not exist %makensis% (
  echo Error: NSIS is not installed ^^^(NSIS v3.0b1^^^)
  goto Quit
)

echo Making installer...

rmdir /s /q Temp 1>nul 2>nul
%zipper% x "Release\Mpdn.Extensions.zip" -oTemp *.* -r 1>nul 2>nul
if not "%ERRORLEVEL%"=="0" echo error: extraction failed & goto Quit

%makensis% "/DPROJECT_NAME=MPDN-Extensions" "/DVERSION=%releaseVer%" /V1 Installer.nsi
if not "%ERRORLEVEL%"=="0" echo error: makensis failed & goto Quit

rmdir /s /q Temp 1>nul 2>nul

move MPDN-Extensions_v*_Installer.exe Release\ 1>nul 2>nul

echo Done.
echo.
pause

:Quit

