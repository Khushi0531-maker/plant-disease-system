@echo off
title FloraScan AI - Plant Disease Detection System
echo ========================================================
echo   🌿 Starting FloraScan AI (Deep Learning Vision)
echo ========================================================
echo.

IF EXIST "B:\plant_ai_env\Scripts\activate.bat" (
    call B:\plant_ai_env\Scripts\activate.bat
    python -m streamlit run app.py --server.port 8503
) ELSE (
    python -m streamlit run app.py --server.port 8503
)

pause
