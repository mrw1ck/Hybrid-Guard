"""Train Random Forest on Dataset_Hybrid.csv using its built-in domain-disjoint train/test partition."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)

URL_FEATURES = [
    "HostRatio", "PathRatio", "NoOfDots", "NoOfHyphens", "NoOfDigits", "NoOfSlashes", "NoOfQueryParams",
    "DigitRatio", "SpecialCharRatio", "NoOfObfuscatedChar", "HostEntropy", "PathEntropy", "HasAtSymbol",
    "SubdomainRatio", "IsPunycode", "HasNonStandardPort", "HasBrandInSubdomainOrPath", "IsShortener",
    "HasEmbeddedURL",
]
QR_FEATURES = [
    "QRVersion", "QRDimension", "QRTotalModules", "QRBlackModules", "QRWhiteModules", "QRDotDensity",
    "QRECCLevel", "QRDataMask", "QRUnusedECC", "QRPayloadLength", "QRCapacityRatio", "QRModulePixelSize",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../Dataset_Hybrid.csv")
    p.add_argument("--features", choices=["all", "url", "qr"], default="all")
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--min-samples-leaf", type=int, default=1)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--output-dir", default="artifacts/rf_hybrid")
    a = p.parse_args()

    feats = {"all": URL_FEATURES + QR_FEATURES, "url": URL_FEATURES, "qr": QR_FEATURES}[a.features]
    df = pd.read_csv(a.data)
    tr, te = df[df.partition == "train"], df[df.partition == "test"]
    Xtr, ytr = tr[feats].astype(float), tr["Label"].astype(int)
    Xte, yte = te[feats].astype(float), te["Label"].astype(int)
    print(f"features={len(feats)} train={len(Xtr)} test={len(Xte)}")

    model = RandomForestClassifier(
        n_estimators=a.n_estimators, max_depth=a.max_depth, min_samples_leaf=a.min_samples_leaf,
        class_weight="balanced", n_jobs=-1, random_state=a.random_state,
    ).fit(Xtr, ytr)

    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    metrics = {
        "accuracy": accuracy_score(yte, pred), "precision": precision_score(yte, pred),
        "recall": recall_score(yte, pred), "f1": f1_score(yte, pred),
        "roc_auc": roc_auc_score(yte, proba), "pr_auc": average_precision_score(yte, proba),
        "fpr": fp / (fp + tn), "fnr": fn / (fn + tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "classification_report": classification_report(yte, pred, digits=4),
        "label_mapping": {"0": "benign", "1": "malicious"},
    }
    metrics = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in metrics.items()}

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats = {f: {"mean": float(Xtr[f].mean()), "std": float(Xtr[f].std(ddof=0)) or 1.0} for f in feats}
    joblib.dump({
        "model": model, "feature_names": feats, "feature_stats": stats, "metrics": metrics,
        "training": {"model_type": "rf", "dataset": "Dataset_Hybrid.csv", "feature_set": a.features,
                     "train_rows": len(Xtr), "test_rows": len(Xte), "n_estimators": a.n_estimators,
                     "max_depth": a.max_depth, "random_state": a.random_state},
    }, out / "url_phishing_model.joblib", compress=3)
    (out / "training_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    imp = pd.DataFrame({"feature": feats, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    imp.to_csv(out / "feature_importance.csv", index=False)

    print({k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})
    print(metrics["classification_report"])
    print("confusion [[TN,FP],[FN,TP]]:", metrics["confusion_matrix"])
    print(imp.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
