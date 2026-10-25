@echo off
echo =====================================================
echo  Starting Hybrid-Guard Web Application...
echo  URL: http://127.0.0.1:8000
echo =====================================================

if exist .venv\Scripts\python.exe (
    set PYTHON_BIN=.venv\Scripts\python.exe
) else (
    set PYTHON_BIN=python
)

%PYTHON_BIN% -m uvicorn webapp.app.main:app --reload --host 127.0.0.1 --port 8000 %*
