@echo off
echo ================================================
echo  AI BATTLE ARENA - Setup
echo ================================================
echo.
echo Installing dependencies...
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

pip install -r requirements.txt

echo.
echo ================================================
echo  Setup Complete!
echo ================================================
echo.
echo To train AIs:      python train_battle.py
echo To watch battle:   python watch_battle.py
echo.
pause
