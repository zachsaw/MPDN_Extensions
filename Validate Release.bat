@echo off
echo This tool validates the extensions for release.
echo.
echo Preparing extensions for validation...
setlocal
cd "%~dp0"\MPDN
rmdir /s /q Extensions 1>NUL 2>NUL
xcopy /e /q /y ..\Extensions\*.* Extensions\ 1>NUL 2>NUL
echo Done.
echo.
echo Starting MPDN to validate extensions...
echo Please make sure MPDN loads the extensions successfully.
echo You can then close the MPDN window after you have finished testing your extensions.
MediaPlayerDotNet.exe