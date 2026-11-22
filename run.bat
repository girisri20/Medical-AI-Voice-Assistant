@echo off
:: ============================================================================
:: Medical AI Voice Assistant — Run the app
:: Starts Streamlit using the project virtual environment (.venv311).
:: Run this every time you want to open the app (after install.bat).
:: ============================================================================

setlocal enabledelayedexpansion

cd /d "%~dp0"

:: Ensure virtual environment exists
if not exist ".venv311\Scripts\activate.bat" (
    echo Virtual environment not found. Running install.bat first...
    call install.bat
    if errorlevel 1 exit /b 1
)

echo ============================================================
echo  Starting Medical AI Voice Assistant ...
echo  App will open at: http://localhost:8501
echo ============================================================
echo.

call .venv311\Scripts\activate.bat
streamlit run app.py --server.port 8501

echo.
echo  Streamlit has stopped.
pause
