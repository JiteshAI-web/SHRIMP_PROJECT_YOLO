@echo off
REM Automated Ollama Installation Checker

echo ============================================================
echo OLLAMA INSTALLATION CHECKER
echo ============================================================
echo.

REM Check if Ollama is installed
where ollama >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✓ Ollama is already installed!
    ollama --version
    echo.
    echo You can now run: ollama serve
) else (
    echo ✗ Ollama is NOT installed
    echo.
    echo ============================================================
    echo INSTALLATION INSTRUCTIONS
    echo ============================================================
    echo.
    echo OPTION 1: Download from website (Recommended)
    echo   1. Open: https://ollama.com/download
    echo   2. Click "Download for Windows"
    echo   3. Run the installer
    echo   4. Restart PowerShell after installation
    echo.
    echo OPTION 2: Install via winget
    echo   Run this command in PowerShell:
    echo   winget install Ollama.Ollama
    echo.
    echo OPTION 3: Install via chocolatey
    echo   Run this command in PowerShell (Admin):
    echo   choco install ollama
    echo.
    echo ============================================================
    echo AFTER INSTALLATION
    echo ============================================================
    echo.
    echo 1. Close and reopen PowerShell
    echo 2. Run: ollama --version
    echo 3. Run: ollama pull llava:13b
    echo 4. Run: ollama serve
    echo.
)

echo.
pause
