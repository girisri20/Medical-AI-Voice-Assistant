@echo off
:: ============================================================================
:: Medical AI Voice Assistant — One-time setup
:: Creates/updates the project virtual environment (.venv311, Python 3.11)
:: and installs all dependencies. Run this ONCE before first use.
:: After that, use run.bat to start the app.
:: ============================================================================

setlocal enabledelayedexpansion

cd /d "%~dp0"
echo ============================================================
echo  Medical AI Voice Assistant — Installing dependencies
echo ============================================================

:: 1. Create virtual environment if missing
if not exist ".venv311\Scripts\python.exe" (
    echo [1/4] Creating virtual environment (.venv311) ...
    python -m venv .venv311
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [1/4] Virtual environment (.venv311) already exists.
)

:: 2. Upgrade pip inside the venv
echo [2/4] Upgrading pip ...
call .venv311\Scripts\python.exe -m pip install --upgrade pip >nul 2>&1

:: 3. Install all dependencies from requirements.txt
echo [3/4] Installing dependencies ...
call .venv311\Scripts\pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    echo.
    echo  NOTE: On Python 3.15, onnxruntime (a faster-whisper dep) has no wheel.
    echo  The .venv311 environment uses Python 3.11, where this works normally.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Setup complete!
echo ============================================================
echo.
echo  Next step: double-click run.bat (or run "run.bat") to start the app.
echo  The app opens at http://localhost:8501
echo.
pause
