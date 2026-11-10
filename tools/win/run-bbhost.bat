@echo off
rem Starts bbhost and writes its log to the logs folder (the newest ten are kept).
rem   run-bbhost.bat            the per-user config (%APPDATA%\bbhost\bbhost.toml)
rem   run-bbhost.bat --setup    the setup window first
cd /d "%~dp0"
if not exist logs mkdir logs
powershell -NoProfile -Command "Get-ChildItem logs\bbhost-*.log | Sort-Object LastWriteTime -Descending | Select-Object -Skip 9 | Remove-Item" 2>nul
set TS=
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set TS=%%i
if "%TS%"=="" set TS=latest
set LOG=logs\bbhost-%TS%.log
echo.
echo  bbhost is starting. Keep this window open while you play.
echo  Log file: %CD%\%LOG%
echo  (send that file when you report a problem)
echo.
bbhost.exe %* > "%LOG%" 2>&1
set RC=%ERRORLEVEL%
echo.
echo  bbhost ended (exit code %RC%). The last lines of the log:
echo.
powershell -NoProfile -Command "Get-Content -Encoding UTF8 -Tail 15 '%LOG%'"
echo.
pause
