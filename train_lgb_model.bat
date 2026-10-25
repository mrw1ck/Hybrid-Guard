@echo off
set DATA_PATH=%~1
if "%DATA_PATH%"=="" (
    if exist Dataset.csv.csv (
        set DATA_PATH=Dataset.csv.csv
    ) else (
        set DATA_PATH=Dataset.csv
    )
)

set OUTPUT_DIR=%~2
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=artifacts\lgb_final

echo =====================================================
echo  Training LightGBM (LGB) Model
echo  Dataset:    %DATA_PATH%
echo  Output dir: %OUTPUT_DIR%
echo =====================================================

if exist .venv\Scripts\python.exe (
    set PYTHON_BIN=.venv\Scripts\python.exe
) else (
    set PYTHON_BIN=python
)

%PYTHON_BIN% train_url_model.py --data "%DATA_PATH%" --model lgb --output-dir "%OUTPUT_DIR%" %*
