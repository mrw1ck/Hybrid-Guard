#!/usr/bin/env bash
set -e

# Detect Python interpreter
if [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON_BIN=".venv/Scripts/python.exe"
elif [ -f ".venv/Scripts/python" ]; then
    PYTHON_BIN=".venv/Scripts/python"
elif [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
else
    PYTHON_BIN="python"
fi

"$PYTHON_BIN" train_all_models.py "$@"
