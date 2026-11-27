# Hybrid-Guard

Hybrid-Guard is a phishing URL risk analyzer that combines machine learning predictions with external checks (DNS, SSL, domain age, and HTTP/content signals).

## Main Features

- Multi-model support (calibrated thesis models, equivalent on the test set, see report): LGB (LightGBM), XGB (XGBoost), RF (Random Forest)
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

## Dataset

`Dataset_Hybrid.csv` (650,770 URLs, 43 columns): 19 URL features, identifier/partition/label columns and 14 QR columns
(12 non-constant). `partition` assigns every registered domain wholly to `train` (520,616) or `test` (130,154).
The shipped models use only the 19 URL features (`FEATURES` in `train_rf_thesis.py`); the QR columns are evaluated in experiment E8.

## Training (thesis protocol)

Grouped 5-fold CV, Optuna TPE (50 trials, no pruner), isotonic calibration on OOF predictions, frozen thresholds, single test pass.

```bash
python train_rf_thesis.py --data Dataset_Hybrid.csv --output-dir artifacts/rf_thesis
python train_boost_thesis.py --model lgb --data Dataset_Hybrid.csv --output-dir artifacts/lgb_thesis
python train_boost_thesis.py --model xgb --data Dataset_Hybrid.csv --output-dir artifacts/xgb_thesis
python boost_extras.py lgb 20 5000      # report tables R1-R13 / figures (lgb or xgb)
```

Experiments: `ablation_e2.py` (E2), `e4_collect.py` + `e4_analyze.py` (E4), `e6_robustness.py` (E6), `e7_features25.py` (E7),
`e8_qr_vs_url.py` + `e8b_len_check.py` (E8: Random Forest on URL only / QR only / URL+QR, plus baselines),
`compare_models_thesis.py` (pairwise tests).

Reports: `make_rf_report.py`, `make_boost_report.py lgb|xgb`, `make_e_report.py`, `make_items_report.py`, `make_e8_report.py`,
then `merge_reports.py` builds `Hybrid_Guard_Full_Report.docx`. The older `train_url_model.py` / `train_all_models.py`
(LGB/RF/DT, fixed thresholds, random split) are legacy and not used by the webapp.

## Debug / Analyze URL

Single URL (LightGBM model artifact):

```bash
python debug_url_model.py --url "github.com" --model-path artifacts/lgb_thesis/url_phishing_model.joblib --timeout 8 --json
```

Fresh check without relying on previous cache:

```bash
python debug_url_model.py --url "https://example.com" --model-path artifacts/lgb_thesis/url_phishing_model.joblib --force-enrichment --cache-file artifacts/tmp_cache_run.json --json
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
python benchmark_url_detector.py --data Dataset_Hybrid.csv --mode ml --model-path artifacts/lgb_thesis/url_phishing_model.joblib --output-dir artifacts/benchmark_lgb
```

Hybrid benchmark:

```bash
python benchmark_url_detector.py --data Dataset_Hybrid.csv --mode hybrid --sample-size 500 --output-dir artifacts/benchmark_hybrid
```

## Notes for This Repository

This repository is source-focused:

- Included: source code, configs, scripts
- Excluded by `.gitignore`:
  - local dataset file (`Dataset_Hybrid.csv`)
  - generated artifacts/reports/benchmarks (`artifacts/`)
  - virtual environment and caches

If you clone this repo on another machine, provide your own dataset and train models before running full analysis.
