@echo off
REM Double-cliquez pour lancer une analyse rapide (TOP 10) dans la console.
cd /d "%~dp0"
set PYTHONUTF8=1
call ".venv\Scripts\activate.bat"
python "scripts\run_analysis_once.py"
pause
