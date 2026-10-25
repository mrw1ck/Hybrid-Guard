#!/usr/bin/env bash
set -e

if [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON_BIN=".venv/Scripts/python.exe"
elif [ -f ".venv/Scripts/python" ]; then
    PYTHON_BIN=".venv/Scripts/python"
elif [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
else
    PYTHON_BIN="python"
fi

echo "====================================================="
echo " Starting Hybrid-Guard Web Application..."
echo " URL: http://127.0.0.1:8000"
echo "====================================================="

"$PYTHON_BIN" -m uvicorn webapp.app.main:app --reload --host 127.0.0.1 --port 8000 "$@"
