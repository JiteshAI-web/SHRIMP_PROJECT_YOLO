@echo off
echo ========================================
echo Shrimp Detection Web Application
echo ========================================
echo.

REM Check if virtual environment exists
if not exist ..\venv (
    echo Creating virtual environment...
    python -m venv ..\venv
)

REM Activate virtual environment
call ..\venv\Scripts\activate.bat

REM Install requirements
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Starting Flask Web Server...
echo.
echo URL: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo ========================================

python app.py

pause
