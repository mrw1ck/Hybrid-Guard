"""QR feature ablation and source-artifact audit for the thesis-protocol LightGBM run.

Trains LightGBM with the tuned hyperparameters of a finished train_lgb_thesis.py run on
several feature sets, using the same domain-grouped train/test partition, and writes:
  tables/qr_ablation.csv      feature-set ablation on the locked test partition
  tables/qr_audit.csv         per-feature QR audit (unique values, single-feature AUC)
  tables/scheme_audit.csv     scheme presence in the raw source URL by class
  qr_ablation.json            the same numbers for the report builder
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from url_ml.feature_engineering import QR_FEATURES, URL_FEATURES  # noqa: E402

ENCODER_ARTIFACTS = ["QRDataMask", "QRModulePixelSize", "QRUnusedECC", "QRECCLevel"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="Dataset.csv")
    p.add_argument("--run-dir", default="artifacts/lgb_thesis_33")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def tpr_at(y, s, fpr_target):
    f, t, _ = roc_curve(y, s)
    return float(np.interp(fpr_target, f, t))


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    params = dict(results["tuning"]["best_params"])
    limit = params.pop("limit_max_depth", False)
    depth = params.pop("max_depth", -1) if limit else -1
    params.pop("max_depth", None)
    model_params = dict(
        n_estimators=params["n_estimators"], num_leaves=params["num_leaves"],
        learning_rate=params["learning_rate"], colsample_bytree=params["feature_fraction"],
        subsample=params["bagging_fraction"], subsample_freq=1,
        min_child_samples=params["min_child_samples"], max_depth=depth,
        class_weight="balanced", random_state=args.seed, n_jobs=-1, verbose=-1,
    )

    cols = URL_FEATURES + QR_FEATURES + ["url", "url_raw", "scheme_added", "Label", "partition"]
    df = pd.read_csv(args.data, usecols=cols)
    df["URLLength"] = df["url"].str.len()
    tr, te = df[df["partition"] == "train"], df[df["partition"] == "test"]
    y_tr, y_te = tr["Label"].to_numpy(), te["Label"].to_numpy()

    sets = [
        ("URL only (19)", URL_FEATURES),
        ("URL + QR (33, deployed)", URL_FEATURES + QR_FEATURES),
        ("URL + QRPayloadLength only", URL_FEATURES + ["QRPayloadLength"]),
        ("URL + normalised URL length", URL_FEATURES + ["URLLength"]),
        ("URL + QR without QRPayloadLength",
         URL_FEATURES + [c for c in QR_FEATURES if c != "QRPayloadLength"]),
        ("URL + QR encoder fields only", URL_FEATURES + ENCODER_ARTIFACTS),
    ]
    rows = []
    for name, feats in sets:
        m = lgb.LGBMClassifier(**model_params).fit(tr[feats].astype(float), y_tr)
        s = m.predict_proba(te[feats].astype(float))[:, 1]
        rows.append({
            "Feature set": name, "Features": len(feats),
            "ROC-AUC": roc_auc_score(y_te, s), "PR-AUC": average_precision_score(y_te, s),
            "TPR@1% FPR": tpr_at(y_te, s, 0.01), "F1@0.5": f1_score(y_te, s >= 0.5),
        })
        print(f"{name:38s} PR-AUC {rows[-1]['PR-AUC']:.4f}  TPR@1% {rows[-1]['TPR@1% FPR']:.4f}", flush=True)
    ablation = pd.DataFrame(rows)

    audit = []
    for c in QR_FEATURES:
        nun = int(tr[c].nunique())
        auc = roc_auc_score(y_tr, tr[c]) if nun > 1 else 0.5
        audit.append({"Feature": c, "Unique values": nun, "AUC": auc, "max(AUC, 1-AUC)": max(auc, 1 - auc)})
    audit = pd.DataFrame(audit)
    payload_match = float((df["url_raw"].str.len() == df["QRPayloadLength"]).mean())
    unused_ecc_lt1 = df.loc[df["QRUnusedECC"] < 1, "Label"].value_counts().to_dict()

    scheme = (df.groupby("Label")["scheme_added"].agg(["size", "mean"])
              .rename(index={0: "Benign", 1: "Malicious"}).reset_index())
    scheme.columns = ["Class", "URLs", "Scheme added during normalisation"]
    scheme["Scheme present in raw source URL"] = 1 - scheme["Scheme added during normalisation"]

    out_t = run_dir / "tables"
    out_t.mkdir(parents=True, exist_ok=True)
    ablation.to_csv(out_t / "qr_ablation.csv", index=False)
    audit.to_csv(out_t / "qr_audit.csv", index=False)
    scheme.to_csv(out_t / "scheme_audit.csv", index=False)
    (run_dir / "qr_ablation.json").write_text(json.dumps({
        "params": model_params,
        "ablation": ablation.to_dict(orient="records"),
        "qr_audit": audit.to_dict(orient="records"),
        "scheme_audit": scheme.to_dict(orient="records"),
        "payload_equals_raw_length_share": payload_match,
        "payload_equals_normalised_length_share": float((df["URLLength"] == df["QRPayloadLength"]).mean()),
        "unused_ecc_below_one_by_label": {str(k): int(v) for k, v in unused_ecc_lt1.items()},
    }, indent=2), encoding="utf-8")
    print("Wrote", run_dir / "qr_ablation.json")


if __name__ == "__main__":
    main()
