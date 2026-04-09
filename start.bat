@echo off
REM Smart-Factory AI — one-click launcher (Windows cmd)
REM Runs the unified FastAPI backend at http://127.0.0.1:7860/

setlocal
set VENV_PY=D:\Ai_Portfolio\.venv\Scripts\python.exe
set APP_DIR=%~dp0backend

if not exist "%VENV_PY%" (
  echo [ERROR] Python venv not found at %VENV_PY%
  echo         Make sure D:\Ai_Portfolio\.venv exists ^(see Ai_Portfolio\SETUP.md^).
  exit /b 1
)

echo ----------------------------------------------------------
echo  Smart-Factory AI — Unified Showcase App
echo  Serving on http://127.0.0.1:7860/
echo  OpenAPI  at http://127.0.0.1:7860/docs
echo  Press Ctrl+C to stop.
echo ----------------------------------------------------------

cd /d "%APP_DIR%"
"%VENV_PY%" -m uvicorn main:app --host 127.0.0.1 --port 7860
