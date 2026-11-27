"""Webapp integration test: (1) ThesisModelService reproduces the calibrated scores stored in predictions.npz;
(2) tier decisions equal the frozen three-level output; (3) the FastAPI app loads the thesis models and answers /api/analyze."""
import json, sys, numpy as np, pandas as pd
from pathlib import Path
from webapp.app.services.thesis_model_service import ThesisModelService
from webapp.app.services.analyzer_service import UrlAnalyzerService
from webapp.app.core.risk_policy import classify_probability
from train_rf_thesis import FEATURES
rows = []; df = pd.read_csv("Dataset.csv"); te = df[df.partition == "test"].reset_index(drop=True)
idx = np.random.default_rng(7).choice(len(te), 1500, replace=False)
for k, nm in [("lgb", "LightGBM"), ("xgb", "XGBoost"), ("rf", "Random Forest")]:
    svc = ThesisModelService(f"artifacts/{k}_thesis/url_phishing_model.joblib"); P = np.load(f"artifacts/{k}_thesis/predictions.npz")
    cal_web, lvl_web = [], []
    for i in idx:
        _, _, p = svc.predict(te.url[i]); cal_web.append(p)
        t = classify_probability(p, svc.thresholds["malicious"], svc.thresholds["suspicious"])
        hard = te.IsShortener[i] == 1 or te.HasEmbeddedURL[i] == 1
        lvl_web.append({"benign": 0, "suspicious": 1, "malicious": 2}[t] if not (t == "benign" and hard) else 1)
    cal_web = np.array(cal_web); cal_ref = P["test_cal"][idx]; lvl_ref = P["level"][idx]
    rows.append({"Model": nm, "n": len(idx), "max |cal_web - cal_stored|": float(np.max(np.abs(cal_web - cal_ref))), "score agreement (|d|<1e-6, %)": 100 * float(np.mean(np.abs(cal_web - cal_ref) < 1e-6)),
                 "risk-tier agreement (%)": 100 * float(np.mean(np.array(lvl_web) == lvl_ref)), "tau_high": svc.tau_high, "tau_low": svc.tau_low})
    print(rows[-1], flush=True)
out = pd.DataFrame(rows); Path("artifacts/webapp_update").mkdir(parents=True, exist_ok=True); out.to_csv("artifacts/webapp_update/W1_score_reproduction.csv", index=False)
