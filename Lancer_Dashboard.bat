@echo off
REM Double-cliquez sur ce fichier pour ouvrir le tableau de bord Market Sentinel AI.
cd /d "%~dp0"
call ".venv\Scripts\activate.bat"
echo Demarrage du tableau de bord... (laissez cette fenetre ouverte)
echo Ouvrez votre navigateur sur http://localhost:8501
streamlit run "dashboard\app.py"
pause
