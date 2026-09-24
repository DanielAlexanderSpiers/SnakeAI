@echo off
title SnakeAI - delete models
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" menu.py --delete
echo.
pause
