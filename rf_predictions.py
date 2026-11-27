"""Regenerate OOF + test predictions of the Random Forest (same folds, seed, best params as train_rf_thesis.py)
so that compare_models_thesis.py can pair them with LightGBM / XGBoost. Deterministic (random_state=42)."""
import json, sys
import joblib, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from train_rf_thesis import FEATURES, SEED, make_rf, risk_levels
threads = int(sys.argv[1]) if len(sys.argv) > 1 else 10
out = "artifacts/rf_thesis"
df = pd.read_csv("Dataset.csv")
tr = df[df.partition == "train"].reset_index(drop=True); te = df[df.partition == "test"].reset_index(drop=True)
Xtr, ytr = tr[FEATURES].astype("float32").values, tr.Label.values.astype(int)
Xte, yte = te[FEATURES].astype("float32").values, te.Label.values.astype(int)
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Xtr, ytr, tr.registered_domain.values))
bp = json.load(open(f"{out}/best_params.json"))
oof = np.zeros(len(ytr)); fold_id = np.zeros(len(ytr), int)
for k, (a, b) in enumerate(folds):
    oof[b] = make_rf(bp, n_jobs=threads).fit(Xtr[a], ytr[a]).predict_proba(Xtr[b])[:, 1]; fold_id[b] = k; print("fold", k, flush=True)
art = joblib.load(f"{out}/url_phishing_model.joblib")
iso = art["calibrator"]; th = art["thresholds"]
raw = art["model"].predict_proba(Xte)[:, 1]; cal = iso.predict(raw)
hard = ((te.IsShortener == 1) | (te.HasEmbeddedURL == 1)).astype(int).values
_, dom_idx = np.unique(te.registered_domain.values, return_inverse=True)
np.savez_compressed(f"{out}/predictions.npz", fold_id=fold_id, y_train=ytr, oof_raw=oof, oof_cal=iso.predict(oof), y_test=yte,
    test_raw=raw, test_cal=cal, test_pred=(cal >= th["tau_high"]).astype(int), level=risk_levels(cal, hard, th["tau_high"], th["tau_low"]),
    test_domain_idx=dom_idx, tau_high=th["tau_high"], tau_low=th["tau_low"])
from sklearn.metrics import average_precision_score
print("test PR-AUC reproduced:", average_precision_score(yte, raw), "(expected 0.959877)")
