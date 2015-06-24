@echo off
setlocal
cd /d "%~dp0"
if not exist "..\Release\Mpdn.Extensions.zip" (
  echo Error: Mpdn.Extensions.zip is not found - run Make.bat first!
  goto Quit
)
set zipper="%ProgramFiles%\7-zip\7z.exe"
if not exist %zipper% (
  echo Error: 7-zip (native version^) is not installed
  goto Quit
)
IF "%PROCESSOR_ARCHITECTURE%"=="x86" (set progfiles86=C:\Program Files) else (set progfiles86=C:\Program Files ^(x86^))
set makensis="%progfiles86%\NSIS\Bin\makensis.exe"
if not exist %makensis% (
  echo Error: NSIS is not installed (NSIS v3.0b1^)
  goto Quit
)

echo Making installer...

git describe --abbrev=0 --tags > latestTag.txt
for /f "delims=" %%i in ('git rev-list HEAD --count') do set commitCount=%%i
set /p latestTag=<latestTag.txt
del latestTag.txt
set releaseVersion=%latestTag%.%commitCount%

rmdir /s /q Temp 1>nul 2>nul
%zipper% x "..\Release\Mpdn.Extensions.zip" -oTemp *.* -r 1>nul 2>nul
if not "%ERRORLEVEL%"=="0" echo error: extraction failed & goto Quit

REM Generator of Uninstaller
%makensis% /V1 unList.nsi

if exist UnInstallLog.log (
    del UnInstallLog.log
)
REM The Installer
unList.exe /DATE=1  /INSTDIR=TEMP\Extensions\  /LOG=UnInstallLog.log  /PREFIX="	"  /UNDIR_VAR="$mpdn_root\Extensions"  /MB=0
%makensis% "/DPROJECT_NAME=MPDN-Extensions" "/DMPDN_REGNAME=MediaPlayerDotNet" "/DVERSION=%releaseVersion%" /V1 Installer.nsi
if not "%ERRORLEVEL%"=="0" echo error: makensis failed & goto Quit

rmdir /s /q Temp 1>nul 2>nul

move MPDN-Extensions_v*_Installer.exe ..\Release\ 1>nul 2>nul

cd "%~dp0\.."

if exist Sign.bat (
    echo Signing installer...
    Echo.
    for %%D in (Release\MPDN-Extensions_v*_Installer.exe) do (
        call Sign.bat %%D
    )
)

goto Done

:Quit

exit /b 1

:Done