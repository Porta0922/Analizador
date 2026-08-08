@echo off
title Verificador de Identidad
echo.
echo ============================================
echo   Verificador de Identidad - Backend
echo ============================================
echo.

REM Verificar si el venv existe
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Entorno virtual no encontrado.
    echo Ejecuta primero: install.ps1
    pause
    exit /b 1
)

REM Arrancar backend
echo Arrancando backend en http://localhost:8000 ...
echo Presiona CTRL+C para detener.
echo.
.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
pause
