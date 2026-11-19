@echo off
REM Quick Start Script for LLM Interpretation System
REM This script helps you set up and run the LLM-powered shrimp interpretation system

echo ============================================================
echo SHRIMP DETECTION - LLM INTERPRETATION SYSTEM SETUP
echo ============================================================
echo.

REM Step 1: Check if Ollama is installed
echo [1/5] Checking Ollama installation...
where ollama >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Ollama is not installed!
    echo Please download and install Ollama from: https://ollama.com/download
    echo.
    pause
    exit /b 1
) else (
    echo ✓ Ollama is installed
    ollama --version
)
echo.

REM Step 2: Check if Ollama server is running
echo [2/5] Checking Ollama server status...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ⚠ Ollama server is not running
    echo Starting Ollama server in background...
    start "" "ollama" cmd /k "ollama serve"
    timeout /t 5 /nobreak >nul
    echo ✓ Ollama server started
) else (
    echo ✓ Ollama server is running
)
echo.

REM Step 3: Check if Llava model is installed
echo [3/5] Checking Llava model...
ollama list | findstr /i "llava" >nul
if %ERRORLEVEL% NEQ 0 (
    echo ⚠ Llava model not found
    echo Downloading Llava 13B model (this may take 5-10 minutes)...
    echo Model size: ~7.5 GB
    echo.
    ollama pull llava:13b
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo ERROR: Failed to download Llava model
        echo Please check your internet connection and try again
        pause
        exit /b 1
    )
    echo ✓ Llava model downloaded successfully
) else (
    echo ✓ Llava model is installed
    ollama list | findstr /i "llava"
)
echo.

REM Step 4: Install Python dependencies
echo [4/5] Installing Python dependencies...
cd /d "%~dp0"
pip install -r requirements.txt >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ⚠ Warning: Some dependencies may have failed to install
    echo Try running: pip install -r requirements.txt manually
) else (
    echo ✓ Dependencies installed
)
echo.

REM Step 5: Run setup test
echo [5/5] Running setup verification...
echo.
python test_llm_setup.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ⚠ Setup test encountered some issues
    echo Please review the output above
    echo.
    pause
)
echo.

echo ============================================================
echo SETUP COMPLETE!
echo ============================================================
echo.
echo Your LLM interpretation system is ready to use!
echo.
echo Next steps:
echo 1. Keep this window open (Ollama server running)
echo 2. Open a new PowerShell window
echo 3. Navigate to: cd %~dp0
echo 4. Start the web app: python app.py
echo 5. Open browser to: http://localhost:5000
echo 6. Click the "👨‍⚕️ Generate Interpretation" button
echo.
echo For detailed instructions, see:
echo - OLLAMA_SETUP.md (Setup guide)
echo - LLM_INTERPRETATION_README.md (Feature documentation)
echo.
pause
