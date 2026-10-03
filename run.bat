@echo off
title EduPredict - Student Performance Prediction System
cd /d "%~dp0"
echo ===================================================================
echo     EduPredict - Student Performance Prediction System
echo ===================================================================
echo.
echo Starting Flask Server and opening your browser...
echo Local URL: http://127.0.0.1:5000
echo.
timeout /t 2 >nul
start http://127.0.0.1:5000
python app.py
pause
