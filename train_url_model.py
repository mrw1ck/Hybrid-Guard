import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from url_ml.feature_engineering import NUMERIC_FEATURES, extract_url_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a phishing URL classifier from a CSV dataset (LightGBM, Random Forest, Decision Tree).",
    )
    parser.add_argument("--data", default="Dataset.csv", help="Path to the CSV dataset.")
    parser.add_argument(
        "--model",
        default="lgb",
        choices=["lgb", "rf", "dt"],
        help="Model type: lgb = LightGBM (Gradient Boosting), rf = Random Forest, dt = Decision Tree.",
    )
    parser.add_argument("--label-col", default="label", help="Target label column.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/lgb_final",
        help="Directory where the trained model and metadata are saved.",
    )
    # Common hyperparams
    parser.add_argument("--n-estimators", type=int, default=300, help="Tree count for LightGBM or Random Forest.")
    parser.add_argument("--max-depth", type=int, default=None, help="Optional max tree depth.")
    # LightGBM specific hyperparams
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate for LightGBM.")
    parser.add_argument("--num-leaves", type=int, default=63, help="Max number of leaves for LightGBM.")
    parser.add_argument("--sample-limit", type=int, default=0, help="Optional maximum row limit for training.")
    return parser.parse_args()


def resolve_data_path(data_arg: str) -> Path:
    data_path = Path(data_arg)
    if data_path.exists():
        return data_path
    if (data_path.parent / f"{data_path.name}.csv").exists():
        return data_path.parent / f"{data_path.name}.csv"
    if data_arg == "Dataset.csv" and Path("Dataset.csv.csv").exists():
        return Path("Dataset.csv.csv")
    raise FileNotFoundError(f"Dataset not found at: {data_arg}")


def ensure_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    missing_features = [column for column in NUMERIC_FEATURES if column not in dataframe.columns]
    if not missing_features:
        return dataframe

    if "url" not in dataframe.columns:
        raise ValueError(
            "Dataset is missing required numeric feature columns and also has no 'url' column to regenerate them.",
        )

    generated_features = dataframe["url"].apply(extract_url_features).apply(pd.Series)
    for column in NUMERIC_FEATURES:
        dataframe[column] = generated_features[column]
    return dataframe


def build_model(args: argparse.Namespace):
    if args.model == "lgb":
        return LGBMClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            max_depth=args.max_depth if args.max_depth is not None else -1,
            random_state=args.random_state,
            class_weight="balanced",
            n_jobs=-1,
            verbose=-1,
        )

    if args.model == "rf":
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            class_weight="balanced",
            n_jobs=-1,
        )

    if args.model == "dt":
        return DecisionTreeClassifier(
            max_depth=args.max_depth,
            random_state=args.random_state,
            class_weight="balanced",
        )

    raise ValueError(f"Unknown model type: {args.model}")


def extract_feature_importances(model, feature_names: list) -> np.ndarray:
    if hasattr(model, "feature_importances_"):
        return np.array(model.feature_importances_, dtype=float)
    return np.zeros(len(feature_names), dtype=float)


def main() -> None:
    args = parse_args()
    data_path = resolve_data_path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from: {data_path}...")
    dataframe = pd.read_csv(data_path)
    if args.label_col not in dataframe.columns:
        raise ValueError(f"Missing label column: {args.label_col}")

    dataframe = ensure_features(dataframe.copy())
    dataframe = dataframe.dropna(subset=[args.label_col])

    if args.sample_limit and args.sample_limit > 0 and len(dataframe) > args.sample_limit:
        dataframe = dataframe.sample(n=args.sample_limit, random_state=args.random_state)
        print(f"Sampled {args.sample_limit} rows from dataset for training.")

    X = dataframe[NUMERIC_FEATURES].astype(float)
    y = dataframe[args.label_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    print(f"Building and training model [{args.model.upper()}] with {len(X_train)} train rows, {len(X_test)} test rows...")
    model = build_model(args)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions, average="binary")),
        "classification_report": classification_report(y_test, predictions, digits=4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "label_mapping": {"0": "benign", "1": "malicious"},
    }

    feature_stats = {
        feature: {
            "mean": float(X_train[feature].mean()),
            "std": float(X_train[feature].std(ddof=0)) if float(X_train[feature].std(ddof=0)) > 0 else 1.0,
        }
        for feature in NUMERIC_FEATURES
    }

    artifact = {
        "model": model,
        "feature_names": NUMERIC_FEATURES,
        "feature_stats": feature_stats,
        "metrics": metrics,
        "training": {
            "rows": int(len(dataframe)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "model_type": args.model,
            "random_state": args.random_state,
            "lgb_config": {
                "n_estimators": args.n_estimators,
                "learning_rate": args.learning_rate,
                "num_leaves": args.num_leaves,
                "max_depth": args.max_depth,
            }
            if args.model == "lgb"
            else None,
        },
    }

    model_path = output_dir / "url_phishing_model.joblib"
    metrics_path = output_dir / "training_metrics.json"
    importance_path = output_dir / "feature_importance.csv"

    joblib.dump(artifact, model_path)
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    importances = extract_feature_importances(model, NUMERIC_FEATURES)
    importance_frame = pd.DataFrame(
        {
            "feature": NUMERIC_FEATURES,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)
    importance_frame.to_csv(importance_path, index=False)

    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved importances to: {importance_path}")
    print("\nEvaluation summary")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    print(metrics["classification_report"])
    print("Confusion matrix:")
    for row in metrics["confusion_matrix"]:
        print(row)


if __name__ == "__main__":
    main()