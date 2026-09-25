@echo off
title SnakeAI - training the best model
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Run run.bat once first to set everything up.
    pause
    exit /b 1
)

set "SNAKEAI_CONFIG=GAMMA=0.99,REPLAY_CAPACITY=1000000,MIN_IMPROVEMENT=0.005"
set "NAME=best"
if not "%~1"=="" set "NAME=%~1"

if exist "models\%NAME%\latest.pt" (
    echo Carrying on training "%NAME%" from where it stopped.
    ".venv\Scripts\python.exe" train.py --name "%NAME%" --resume --rounds 500 --patience 30
) else (
    ".venv\Scripts\python.exe" train.py --name "%NAME%" --snakes 1000 --grid 20 --senses vision5 --hidden 256 --rounds 500 --patience 30
)
echo.
pause
