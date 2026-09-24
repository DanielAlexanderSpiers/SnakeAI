@echo off
title SnakeAI
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo First run: setting up Python and installing numpy, pygame, torch, numba.
    echo This downloads a few hundred MB and only happens once.
    echo.
    py -3 -m venv .venv 2>nul || python -m venv .venv
    if not exist ".venv\Scripts\python.exe" (
        echo Could not create the Python environment. Is Python 3 installed?
        pause
        exit /b 1
    )
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Installing packages failed.
        rmdir /s /q .venv
        pause
        exit /b 1
    )
)

".venv\Scripts\python.exe" menu.py
if errorlevel 1 pause
