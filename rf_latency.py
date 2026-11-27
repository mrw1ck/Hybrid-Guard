"""RF: re-measure R11 (SHAP on 5,000 URLs) and R12 (latency incl. feature extraction); redraw figR6/figR7. Other RF tables untouched."""
import time, json, numpy as np, pandas as pd, joblib, shap, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import average_precision_score
from train_rf_thesis import FEATURES, SEED, risk_levels
from url_ml.thesis_features import extract
plt.rcParams.update({"font.family": "serif", "figure.dpi": 150, "axes.grid": True, "grid.alpha": .3})
out = Path("artifacts/rf_thesis"); rep = out / "report"; T0 = time.time(); log = lambda m: print(f"[{(time.time()-T0)/60:5.1f} min] {m}", flush=True)
art = joblib.load(out / "url_phishing_model.joblib"); model, iso = art["model"], art["calibrator"]; S = json.load(open(rep / "summary.json"))
te = pd.read_csv("Dataset.csv"); te = te[te.partition == "test"].reset_index(drop=True); Xte = te[FEATURES].astype("float32").values; yte = te.Label.values.astype(int)
tau_high, tau_low = S["tau_high"], S["tau_low"]; urls = te.url.values[:1050]; fe = []
for u in urls[50:]: t0 = time.perf_counter(); extract(u); fe.append((time.perf_counter() - t0) * 1e3)
model.set_params(n_jobs=1); li, lc = [], []
for i in range(1050):
    x = Xte[i:i + 1]; t0 = time.perf_counter(); p = model.predict_proba(x)[:, 1]; t1 = time.perf_counter(); c = iso.predict(p); risk_levels(c, np.zeros(1), tau_high, tau_low); t2 = time.perf_counter()
    if i >= 50: li.append((t1 - t0) * 1e3); lc.append((t2 - t1) * 1e3)
q = lambda v: [np.median(v), np.percentile(v, 95), np.percentile(v, 99)]; tot = np.array(fe) + np.array(li) + np.array(lc)
pd.DataFrame([["Feature extraction (re-implemented extractor)", *q(fe)], ["Random Forest inference", *q(li)], ["Calibration + risk level", *q(lc)], ["Total", *q(tot)]],
             columns=["Stage", "Median ms", "p95 ms", "p99 ms"]).to_csv(rep / "R12_latency.csv", index=False)
log("DONE")
