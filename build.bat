@echo off
REM Country Style Dough Inspector - Windows Build Script (Batch version)
REM Simple batch file wrapper for PowerShell script

powershell.exe -ExecutionPolicy Bypass -File "%~dp0build.ps1"
if %ERRORLEVEL% NEQ 0 (
    pause
    exit /b %ERRORLEVEL%
)

