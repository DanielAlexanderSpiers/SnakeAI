@echo off
title SnakeAI - training the best model
cd /d "%~dp0"

rem Best-performance settings (training time does not matter):
rem   vision5 senses, look-ahead, route safety, 3 training stages  (all defaults)
rem   1000 snakes on 20x20, brain 256 wide, up to 500 rounds per stage
rem   gamma 0.99 (plans further ahead: values filling the board fast)
rem   1,000,000 remembered experiences (default 300,000)
rem   a stage only moves on after 30 rounds without at least 0.5% improvement

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
