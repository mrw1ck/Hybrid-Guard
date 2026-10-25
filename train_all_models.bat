@echo off
if exist .venv\Scripts\python.exe (
    set PYTHON_BIN=.venv\Scripts\python.exe
) else (
    set PYTHON_BIN=python
)

%PYTHON_BIN% train_all_models.py %*
