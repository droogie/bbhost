@echo off
rem Starts bbhost with the playtest settings and writes its log to the logs folder.
cd /d "%~dp0"
if not exist logs mkdir logs
set TS=
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set TS=%%i
if "%TS%"=="" set TS=latest
set LOG=logs\bbhost-%TS%.log
echo.
echo  bbhost is starting. Keep this window open while you play.
echo  Log file: %CD%\%LOG%
echo  (send that file when you report a problem)
echo.
bbhost.exe --config bbhost-playtest.toml %* > "%LOG%" 2>&1
set RC=%ERRORLEVEL%
echo.
echo  bbhost ended (exit code %RC%). The last lines of the log:
echo.
powershell -NoProfile -Command "Get-Content -Encoding UTF8 -Tail 15 '%LOG%'"
echo.
pause
