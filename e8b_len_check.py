"""E8b: is the URL+QR gain explained by one absolute-length column (QRPayloadLength)? Same protocol as e8_qr_vs_url.py."""
import json, numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedGroupKFold
from train_rf_thesis import FEATURES, SEED, Curve, make_rf, threshold_metrics, pick_thresholds
df = pd.read_csv("Dataset_Hybrid.csv", usecols=["url_raw","registered_domain","partition","Label","QRPayloadLength"]+FEATURES)
tr = df[df.partition=="train"].reset_index(drop=True); te = df[df.partition=="test"].reset_index(drop=True)
ytr, yte = tr.Label.values.astype(int), te.Label.values.astype(int)
print("payload==len(url_raw):", float((df.QRPayloadLength==df.url_raw.str.len()).mean()))
bp = json.load(open("artifacts/rf_thesis/best_params.json"))
_, dom = np.unique(te.registered_domain.values, return_inverse=True); nd = dom.max()+1
rng = np.random.default_rng(SEED); W = [rng.multinomial(nd, np.full(nd,1/nd))[dom].astype(float) for _ in range(2000)]
cols = FEATURES+["QRPayloadLength"]
m = make_rf(bp, 16).fit(tr[cols].astype("float32").values, ytr); raw = m.predict_proba(te[cols].astype("float32").values)[:,1]
c = Curve(raw, yte); p = c.stats(np.ones(len(yte))); bs = pd.DataFrame([c.stats(w) for w in W])
base = pd.read_csv("artifacts/e8_qr/E8_paired_delta.csv")
out = {k:{"value":p[k],"lo":bs[k].quantile(.025),"hi":bs[k].quantile(.975)} for k in ["ROC-AUC","PR-AUC","TPR@1%FPR","TPR@0.1%FPR"]}
json.dump(out, open("artifacts/e8_qr/E8b_url_plus_payload.json","w"), indent=1); print(out)
