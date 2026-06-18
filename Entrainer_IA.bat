@echo off
REM Double-cliquez pour (ré)entrainer l'IA locale sur l'historique de la watchlist.
cd /d "%~dp0"
set PYTHONUTF8=1
call ".venv\Scripts\activate.bat"
python "scripts\train_ai.py"
pause
