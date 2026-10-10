@echo off
rem Install FSR 4 v07 INT8/DOT4 shader assets for bbhost on Windows
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install-FSR4.ps1" %*
endlocal