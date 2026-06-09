@echo off
REM Start the Foldback Service executable
REM Place your .env file in the same directory as the executable before running.

set SCRIPT_DIR=%~dp0
set EXE_DIR=%SCRIPT_DIR%dist\foldback-service

cd /d "%EXE_DIR%"
echo Starting Foldback Service...
echo.
foldback-service.exe
pause
