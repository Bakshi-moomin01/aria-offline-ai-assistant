@echo off
title ARIA — Offline AI Assistant
color 0B
cls
echo ================================================================
echo   ARIA: Offline AI Assistant using Local LLMs (Ollama)
echo ================================================================
echo.

cd /d "%~dp0"

echo [1/3] Checking Ollama connection (port 11434)...
curl -s http://127.0.0.1:11434 >nul 2>&1
if %errorlevel% neq 0 (
    echo [NOTE] Ollama server is not detected on http://127.0.0.1:11434
    echo Please start Ollama or run: ollama run llama3.2
    echo.
)

echo [2/3] Verifying Python dependencies...
python -m pip install -r requirements.txt --quiet

echo [3/3] Starting ARIA Web Assistant...
echo Opening browser at: http://127.0.0.1:8080
echo.
python run.py

pause
