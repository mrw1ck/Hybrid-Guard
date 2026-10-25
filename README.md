# Hybrid-Guard

Hybrid-Guard is a phishing URL risk analyzer that combines machine learning predictions with external checks (DNS, SSL, domain age, and HTTP/content signals).

## Main Features

- Multi-model support:
  - LGB (LightGBM: Gradient Boosted Decision Trees - fast, high accuracy on large datasets)
  - RF (Random Forest)
  - DT (Decision Tree)
- Hybrid risk scoring pipeline:
  - Base model probability
  - Policy/risk adjustment
  - External enrichment checks
- Web UI + REST API for interactive analysis + QR Code / Quishing Scanner
- Benchmark script for model evaluation (accuracy, precision, recall, F1, TP/TN/FP/FN)

## Project Structure

- `url_ml/`: feature engineering components
- `webapp/`: FastAPI backend + Jinja UI
- `train_url_model.py`: train LightGBM / RF / DT model
- `train_lgb_model.py`: convenience entrypoint for LightGBM mode
- `train_all_models.py`: train & benchmark all models (LGB, RF, DT) in sequence
- `debug_url_model.py`: inspect one or multiple URLs with explainable output
- `benchmark_url_detector.py`: benchmark model or hybrid mode
- `requirements.txt`: Python dependencies

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Training

### Train LightGBM (LGB) - Recommended

```bash
# Cách 1: Chạy script bash / bat riêng
# Trên Windows:
train_lgb_model.bat
# Hoặc trên Linux / Git Bash:
bash train_lgb_model.sh

# Cách 2: Chạy script Python riêng
python train_lgb_model.py --data Dataset.csv

# Cách 3: Chạy qua train_url_model
python train_url_model.py --data Dataset.csv --model lgb --output-dir artifacts/lgb_final
```

### Train RF (Random Forest)

```bash
python train_url_model.py --data Dataset.csv --model rf --output-dir artifacts/rf_final
```

### Train DT (Decision Tree)

```bash
python train_url_model.py --data Dataset.csv --model dt --max-depth 10 --output-dir artifacts/dt_final
```

### Train All Models (LGB, RF, DT) & Compare Benchmark

```bash
# Trên Windows:
train_all_models.bat

# Hoặc trên Linux / Git Bash:
bash train_all_models.sh

# Hoặc Python trực tiếp:
python train_all_models.py
```

## Debug / Analyze URL

Single URL (LightGBM model artifact):

```bash
python debug_url_model.py --url "github.com" --model-path artifacts/lgb_final/url_phishing_model.joblib --timeout 8 --json
```

Fresh check without relying on previous cache:

```bash
python debug_url_model.py --url "https://example.com" --model-path artifacts/lgb_final/url_phishing_model.joblib --force-enrichment --cache-file artifacts/tmp_cache_run.json --json
```

## Run API + Web UI

```bash
uvicorn webapp.app.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

- http://127.0.0.1:8000

## Benchmark

Model-only benchmark:

```bash
python benchmark_url_detector.py --data Dataset.csv --mode ml --model-path artifacts/lgb_final/url_phishing_model.joblib --output-dir artifacts/benchmark_lgb
```

Hybrid benchmark:

```bash
python benchmark_url_detector.py --data Dataset.csv --mode hybrid --sample-size 500 --output-dir artifacts/benchmark_hybrid
```

## Notes for This Repository

This repository is source-focused:

- Included: source code, configs, scripts
- Excluded by `.gitignore`:
  - local dataset file (`Dataset.csv` / `Dataset.csv.csv`)
  - generated artifacts/reports/benchmarks (`artifacts/`)
  - virtual environment and caches

If you clone this repo on another machine, provide your own dataset and train models before running full analysis.
