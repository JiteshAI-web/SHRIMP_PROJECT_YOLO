@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo Shrimp Detection ^& Monitoring Dashboard (Windows Launcher)
echo ============================================================

REM Create virtual environment if missing
if not exist "venv" (
    echo [INFO] Creating Python virtual environment (venv)...
    python -m venv venv
    call venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

echo [INFO] Starting Flask web dashboard on http://127.0.0.1:5000 ...
cd shrimp_detection\webapp
python app.py
pause
