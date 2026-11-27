"""Train RF (best params, thesis protocol) on Dataset_Hybrid.csv and export Dataset_Hybrid_RF.csv with RF scores."""
import json, sys, numpy as np, pandas as pd, joblib
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from train_rf_thesis import FEATURES, SEED, make_rf, risk_levels
threads = int(sys.argv[1]) if len(sys.argv) > 1 else 18
df = pd.read_csv("Dataset_Hybrid.csv")
trm = (df.partition == "train").values
tr, te = df[trm], df[~trm]
Xtr, ytr = tr[FEATURES].astype("float32").values, tr.Label.values.astype(int)
Xte, yte = te[FEATURES].astype("float32").values, te.Label.values.astype(int)
bp = json.load(open("artifacts/rf_thesis/best_params.json"))
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Xtr, ytr, tr.registered_domain.values))
oof = np.zeros(len(ytr)); fid = np.zeros(len(ytr), int)
for k, (a, b) in enumerate(folds):
    oof[b] = make_rf(bp, n_jobs=threads).fit(Xtr[a], ytr[a]).predict_proba(Xtr[b])[:, 1]; fid[b] = k; print("fold", k, flush=True)
iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof, ytr)
oc = iso.predict(oof)
neg, pos = np.sort(oc[ytr == 0]), oc[ytr == 1]
cand = np.unique(oc)
th_hi = min(c for c in cand if (oc[ytr == 0] >= c).mean() <= 0.01)
th_lo = max(c for c in cand if (oc[ytr == 1] < c).mean() <= 0.02)
model = make_rf(bp, n_jobs=threads).fit(Xtr, ytr)
raw_te = model.predict_proba(Xte)[:, 1]; cal_te = iso.predict(raw_te)
print("test PR-AUC", average_precision_score(yte, raw_te), "ROC-AUC", roc_auc_score(yte, raw_te), "tau", th_hi, th_lo)
hard = lambda d: ((d.IsShortener == 1) | (d.HasEmbeddedURL == 1)).astype(int).values
out = df.copy()
out["rf_prob_raw"] = np.nan; out["rf_prob_cal"] = np.nan; out["rf_oof"] = 0
out.loc[trm, "rf_prob_raw"] = oof; out.loc[trm, "rf_prob_cal"] = oc; out.loc[trm, "rf_oof"] = 1
out.loc[~trm, "rf_prob_raw"] = raw_te; out.loc[~trm, "rf_prob_cal"] = cal_te
out["rf_pred"] = (out.rf_prob_cal >= th_hi).astype(int)
out["rf_risk_level"] = risk_levels(out.rf_prob_cal.values, hard(out), th_hi, th_lo)
out.to_csv("Dataset_Hybrid_RF.csv", index=False)
joblib.dump({"model": model, "calibrator": iso, "thresholds": {"tau_high": float(th_hi), "tau_low": float(th_lo)}, "features": FEATURES}, "artifacts/rf_thesis/url_phishing_model_retrain.joblib")
print("done", out.shape)
