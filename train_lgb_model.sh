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

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    exec "$PYTHON_BIN" train_url_model.py --help
fi

DATA_PATH="Dataset.csv"
if [ ! -f "$DATA_PATH" ] && [ -f "Dataset.csv.csv" ]; then
    DATA_PATH="Dataset.csv.csv"
fi
OUTPUT_DIR="artifacts/lgb_final"

if [[ -n "$1" && "$1" != -* ]]; then
    DATA_PATH="$1"
    shift
fi

if [[ -n "$1" && "$1" != -* ]]; then
    OUTPUT_DIR="$1"
    shift
fi

echo "====================================================="
echo " Training LightGBM (LGB) Model"
echo " Dataset:    $DATA_PATH"
echo " Output dir: $OUTPUT_DIR"
echo "====================================================="

"$PYTHON_BIN" train_url_model.py \
    --data "$DATA_PATH" \
    --model lgb \
    --output-dir "$OUTPUT_DIR" \
    "$@"

echo "====================================================="
echo " LightGBM Training Completed!"
echo " Artifacts saved in: $OUTPUT_DIR"
echo "====================================================="
