#!/usr/bin/env python3
"""Train a Random Forest on the pre-engineered URL dataset."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


FEATURE_COLUMNS = [
    "URLLength",
    "HostnameLength",
    "PathLength",
    "QueryLength",
    "NoOfDots",
    "NoOfHyphens",
    "NoOfDigits",
    "NoOfSlashes",
    "NoOfQueryParams",
    "DigitRatio",
    "SpecialCharRatio",
    "NoOfObfuscatedChar",
    "HostEntropy",
    "PathEntropy",
    "HasAtSymbol",
    "IsIPHost",
    "SubdomainDepth",
    "IsSuspiciousTLD",
    "IsPunycode",
    "HasHTTPS",
    "HasNonStandardPort",
    "NoOfSuspiciousKeywords",
    "HasBrandInSubdomainOrPath",
    "IsShortener",
    "HasEmbeddedURL",
]


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Random Forest using the dataset's train/test partitions.",
    )
    parser.add_argument(
        "--data",
        default=str(project_dir.parent / "Dataset.csv"),
        help="Path to the dataset CSV (default: ../Dataset.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_dir / "artifacts" / "rf_dataset"),
        help="Directory for the model, metrics, and feature importances.",
    )
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees.")
    parser.add_argument("--max-depth", type=int, default=25, help="Maximum tree depth.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if args.n_estimators < 1:
        raise ValueError("--n-estimators must be at least 1.")
    if args.max_depth < 1:
        raise ValueError("--max-depth must be at least 1.")

    required_columns = [*FEATURE_COLUMNS, "Label", "partition"]
    dataframe = pd.read_csv(data_path, usecols=required_columns)
    dataframe.columns = [str(column).strip().lstrip("\ufeff") for column in dataframe.columns]
    if dataframe[required_columns].isna().any().any():
        missing = dataframe[required_columns].columns[
            dataframe[required_columns].isna().any()
        ].tolist()
        raise ValueError(f"Dataset contains missing values in required columns: {missing}")

    partitions = dataframe["partition"].astype(str).str.strip().str.casefold()
    unexpected_partitions = sorted(set(partitions) - {"train", "test"})
    if unexpected_partitions:
        raise ValueError(
            "Expected the dataset's partition column to contain only 'train' and 'test'; "
            f"found: {unexpected_partitions}",
        )

    labels = pd.to_numeric(dataframe["Label"], errors="raise")
    if set(labels.unique()) != {0, 1}:
        raise ValueError(
            "Expected Label to contain both binary classes 0 (benign) and 1 (malicious).",
        )

    features = dataframe[FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise").astype(float)
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Feature columns contain infinite or non-finite values.")

    train_mask = partitions == "train"
    test_mask = partitions == "test"
    X_train, y_train = features.loc[train_mask], labels.loc[train_mask].astype(int)
    X_test, y_test = features.loc[test_mask], labels.loc[test_mask].astype(int)
    if X_train.empty or X_test.empty:
        raise ValueError("Both train and test partitions must contain rows.")
    if set(y_train.unique()) != {0, 1} or set(y_test.unique()) != {0, 1}:
        raise ValueError("Both train and test partitions must contain both label classes.")

    print(f"Dataset: {data_path}")
    print(f"Rows: {len(dataframe):,} (train: {len(X_train):,}, test: {len(X_test):,})")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print("Training Random Forest...")

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "classification_report": classification_report(
            y_test,
            predictions,
            digits=4,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "rf_dataset_model.joblib"
    metrics_path = output_dir / "training_metrics.json"
    importance_path = output_dir / "feature_importance.csv"

    joblib.dump(
        {
            "model": model,
            "feature_names": FEATURE_COLUMNS,
            "metrics": metrics,
            "training": {
                "rows": len(dataframe),
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "model_type": "rf",
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "random_state": args.random_state,
                "split_column": "partition",
            },
        },
        model_path,
    )
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False).to_csv(importance_path, index=False)

    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved feature importances: {importance_path}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    print("Confusion matrix:")
    for row in metrics["confusion_matrix"]:
        print(row)


if __name__ == "__main__":
    main()
