@echo off
chcp 65001 >nul
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
echo  bbhost を起動しています。プレイ中はこのウィンドウを開いたままにしてください。
echo  ログ: %CD%\%LOG%
echo  （不具合報告の際はこのログを添えてください）
echo.
bbhost.exe %* > "%LOG%" 2>&1
set RC=%ERRORLEVEL%
echo.
echo  bbhost が終了しました（終了コード %RC%）。ログの末尾:
echo.
powershell -NoProfile -Command "Get-Content -Tail 15 '%LOG%'"
echo.
pause
