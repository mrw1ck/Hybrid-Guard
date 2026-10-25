#!/usr/bin/env python3
"""Master script to train and benchmark URL phishing detection models in sequence:

- LGB (LightGBM Gradient Boosting)
- RF (Random Forest)
- DT (Decision Tree)
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd


MODELS_CONFIG = [
    {
        "name": "lgb",
        "display": "LightGBM (Gradient Boosting)",
        "output_dir": "artifacts/lgb_final",
        "cmd": [
            "--model", "lgb",
            "--n-estimators", "300",
            "--output-dir", "artifacts/lgb_final",
        ],
    },
    {
        "name": "rf",
        "display": "Random Forest (RF)",
        "output_dir": "artifacts/rf_final",
        "cmd": [
            "--model", "rf",
            "--n-estimators", "100",
            "--max-depth", "25",
            "--output-dir", "artifacts/rf_final",
        ],
    },
    {
        "name": "dt",
        "display": "Decision Tree (DT)",
        "output_dir": "artifacts/dt_final",
        "cmd": [
            "--model", "dt",
            "--max-depth", "20",
            "--output-dir", "artifacts/dt_final",
        ],
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and benchmark models in Hybrid-Guard (LGB, RF, DT).",
    )
    parser.add_argument("--data", default="Dataset.csv", help="Path to the dataset CSV.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lgb", "rf", "dt"],
        choices=["lgb", "rf", "dt"],
        help="List of models to run (default: lgb rf dt).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip training for models that already have saved artifacts in artifacts/<model>_final.",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Do not retrain; only aggregate and display metrics of already trained models.",
    )
    return parser.parse_args()


def get_python_executable():
    venv_py_win = Path(".venv/Scripts/python.exe")
    venv_py_unix = Path(".venv/bin/python")
    if venv_py_win.exists():
        return str(venv_py_win)
    if venv_py_unix.exists():
        return str(venv_py_unix)
    return sys.executable


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s}s"


def resolve_data_path(data_arg: str) -> Path:
    data_path = Path(data_arg)
    if data_path.exists():
        return data_path
    if (data_path.parent / f"{data_path.name}.csv").exists():
        return data_path.parent / f"{data_path.name}.csv"
    if data_arg == "Dataset.csv" and Path("Dataset.csv.csv").exists():
        return Path("Dataset.csv.csv")
    return data_path


def main():
    args = parse_args()
    python_bin = get_python_executable()
    data_path = resolve_data_path(args.data)

    if not data_path.exists():
        print(f"[ERROR] Dataset file not found: {data_path}")
        sys.exit(1)

    print("=" * 80)
    print(" HYBRID-GUARD: MASTER TRAINING & BENCHMARK PIPELINE")
    print("=" * 80)
    print(f" Python Interpreter: {python_bin}")
    print(f" Dataset Path:       {data_path} ({data_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f" Target Models:      {', '.join(m.upper() for m in args.models)}")
    print(f" Mode:               {'Benchmark Only' if args.benchmark_only else 'Train & Benchmark'}")
    print("=" * 80)

    results = []
    total_start = time.time()

    for item in MODELS_CONFIG:
        model_name = item["name"]
        if model_name not in args.models:
            continue

        output_dir = Path(item["output_dir"])
        model_artifact = output_dir / "url_phishing_model.joblib"
        metrics_file = output_dir / "training_metrics.json"

        elapsed_str = "Loaded"
        print(f"\n>>> [{model_name.upper()}] {item['display']}")

        if not args.benchmark_only:
            if args.skip_existing and model_artifact.exists() and metrics_file.exists():
                print(f"    [SKIP] Found existing artifact at: {model_artifact}")
            else:
                print(f"    [TRAIN] Training {item['display']}...")
                cmd = [python_bin, "train_url_model.py", "--data", str(data_path)] + item["cmd"]
                t0 = time.time()
                ret = subprocess.run(cmd)
                t1 = time.time()
                if ret.returncode != 0:
                    print(f"    [ERROR] Failed to train {model_name} (exit code: {ret.returncode})")
                    continue
                elapsed_str = format_duration(t1 - t0)
                print(f"    [SUCCESS] Training finished in {elapsed_str}")

        # Read metrics
        if metrics_file.exists():
            try:
                with metrics_file.open("r", encoding="utf-8") as f:
                    metrics = json.load(f)

                acc = metrics.get("accuracy", 0.0)
                f1 = metrics.get("f1", 0.0)

                # Extract precision/recall for class 1 from confusion matrix if available
                cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
                tn, fp = cm[0][0], cm[0][1]
                fn, tp = cm[1][0], cm[1][1]
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

                results.append({
                    "Model": model_name.upper(),
                    "Algorithm": item["display"],
                    "Accuracy": f"{acc * 100:.2f}%",
                    "F1-Score": f"{f1:.4f}",
                    "Precision (Phish)": f"{prec * 100:.2f}%",
                    "Recall (Phish)": f"{rec * 100:.2f}%",
                    "Status": "Ready",
                    "Train Time": elapsed_str,
                    "Artifact Path": str(model_artifact),
                })
            except Exception as e:
                print(f"    [WARN] Failed to parse metrics for {model_name}: {e}")
        else:
            print(f"    [WARN] No metrics file found at: {metrics_file}")

    total_time = time.time() - total_start

    print("\n" + "=" * 90)
    print("                     ALL MODELS BENCHMARK & PERFORMANCE SUMMARY")
    print("=" * 90)

    if results:
        df_summary = pd.DataFrame(results)
        # Display clean text table
        print(df_summary[["Model", "Accuracy", "F1-Score", "Precision (Phish)", "Recall (Phish)", "Train Time"]].to_string(index=False))

        # Save CSV summary
        summary_path = Path("artifacts/all_models_summary.csv")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print("=" * 90)
        print(f" Saved summary benchmark report to: {summary_path}")
    else:
        print(" No model metrics found.")

    print(f" Total Elapsed Time: {format_duration(total_time)}")
    print("=" * 90)
    print("\n[NOTE / HUONG DAN]")
    print("- All models above are active in Web UI: http://127.0.0.1:8000")
    print("- Switch the 'Model' dropdown to compare predictions across models in real time!\n")


if __name__ == "__main__":
    main()
