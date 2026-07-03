@echo off
rem RT Translator launcher. Creates venv and installs deps on first run.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

if exist ".venv\Scripts\python.exe" goto run

echo [setup] First run: creating venv and installing dependencies (takes a few minutes)...
py -3.12 -m venv .venv 2>nul || python -m venv .venv
if not exist ".venv\Scripts\python.exe" (
    echo [error] Python not found. Please install Python 3.10 - 3.12.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt

:run
".venv\Scripts\python.exe" -u -m rt_translator.main
if errorlevel 1 pause
