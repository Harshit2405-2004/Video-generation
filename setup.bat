@echo off
echo Setting up the AI Video Generator application...

REM Check if Python 3.10 is installed
py -3.10 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python 3.10 is not installed or not in the PATH. Please install Python 3.10 and try again.
    pause
    exit /b 1
)

REM Create a virtual environment using Python 3.10
echo Creating virtual environment with Python 3.10...
py -3.10 -m venv venv
if %errorlevel% neq 0 (
    echo Failed to create virtual environment.
    pause
    exit /b 1
)

REM Activate the virtual environment
echo Activating virtual environment...
call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install requirements.
    pause
    exit /b 1
)

echo Installation complete!
echo To run the application, use the run.bat script or:
echo 1. Activate the virtual environment: venv\Scripts\activate
echo 2. Run the application: python main.py

pause
