"""Item 10: Table 3.5 lists 25 features, the implemented/stored set has 19. The 25-feature set is reconstructible from the report text:
16 shared + {URLLength, HostnameLength, PathLength, QueryLength, SubdomainDepth, IsIPHost, IsSuspiciousTLD, HasHTTPS, NoOfSuspiciousKeywords}
replacing {HostRatio, PathRatio, SubdomainRatio}. Sensitivity test: refit each model on 25 (and on 24 = without HasHTTPS) with the SAME tuned
hyperparameters (no re-tuning), evaluate once on the locked test set, paired domain-cluster bootstrap against the stored 19-feature scores."""
import json, re, sys, time
import numpy as np, pandas as pd
from urllib.parse import urlsplit
from sklearn.ensemble import RandomForestClassifier
from train_rf_thesis import FEATURES, SEED, Curve
from train_boost_thesis import build
from webapp.app.config import RISKY_TLDS, SUSPICIOUS_KEYWORDS
TH = int(sys.argv[1]); B = 1000
EXTRA = ["URLLength", "HostnameLength", "PathLength", "QueryLength", "SubdomainDepth", "IsIPHost", "IsSuspiciousTLD", "HasHTTPS", "NoOfSuspiciousKeywords"]
DROP = ["HostRatio", "PathRatio", "SubdomainRatio"]
F25 = [f for f in FEATURES if f not in DROP] + EXTRA; F24 = [f for f in F25 if f != "HasHTTPS"]
assert len(F25) == 25
ip = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
def extra(url, rd):
    s = urlsplit(url); h = (s.hostname or "").lower(); sub = h[:-len(rd) - 1] if h.endswith("." + rd) else ""; txt = (h + s.path + " " + s.query).lower()
    tld = h.rsplit(".", 1)[-1] if h else ""
    return [len(url), len(h), len(s.path), len(s.query), len([x for x in sub.split(".") if x]), int(bool(ip.match(h)) or h.startswith("[")),
            int(tld in RISKY_TLDS), int(s.scheme == "https"), sum(k in txt for k in SUSPICIOUS_KEYWORDS)]
df = pd.read_csv("Dataset_Hybrid.csv"); X = np.array([extra(u, r) for u, r in zip(df.url, df.registered_domain)], dtype="float32")
for i, c in enumerate(EXTRA): df[c] = X[:, i]
tr = df[df.partition == "train"].reset_index(drop=True); te = df[df.partition == "test"].reset_index(drop=True)
ytr, yte = tr.Label.values.astype(int), te.Label.values.astype(int); pw = float((ytr == 0).sum() / (ytr == 1).sum())
prev = pd.DataFrame({"Feature": EXTRA, "Mean (benign)": [df.loc[df.Label == 0, c].mean() for c in EXTRA], "Mean (malicious)": [df.loc[df.Label == 1, c].mean() for c in EXTRA]})
from sklearn.metrics import roc_auc_score
prev["Single-feature AUC (train)"] = [roc_auc_score(ytr, tr[c]) for c in EXTRA]; prev["max(AUC,1-AUC)"] = np.maximum(prev.iloc[:, -1], 1 - prev.iloc[:, -1])
prev["Scheme added by dataset builder (share)"] = np.nan
prev.loc[prev.Feature == "HasHTTPS", "Scheme added by dataset builder (share)"] = float(df.scheme_added.mean())
import os; os.makedirs("artifacts/e7_features25", exist_ok=True); prev.to_csv("artifacts/e7_features25/E7_new_feature_stats.csv", index=False)
_, dom = np.unique(te.registered_domain.values, return_inverse=True); nd = dom.max() + 1
rng = np.random.default_rng(SEED); W = [rng.multinomial(nd, np.full(nd, 1 / nd))[dom].astype(float) for _ in range(B)]
keys = ["ROC-AUC", "PR-AUC", "TPR@1%FPR", "TPR@0.1%FPR"]
def mk(name, p):
    if name == "LightGBM": return build("lgb", p, TH, pw)
    if name == "XGBoost": return build("xgb", p, TH, pw)
    mf = p["max_features"]; mf = float(mf) if mf not in ("sqrt", "log2") else mf
    return RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"], min_samples_split=p["min_samples_split"], max_features=mf, class_weight="balanced", n_jobs=TH, random_state=SEED)
P = {"LightGBM": json.load(open("artifacts/lgb_thesis/report/summary.json"))["best_params"], "XGBoost": json.load(open("artifacts/xgb_thesis/report/summary.json"))["best_params"], "Random Forest": json.load(open("artifacts/rf_thesis/best_params.json"))}
D = {"LightGBM": "lgb", "XGBoost": "xgb", "Random Forest": "rf"}; rows = []; T0 = time.time()
for name in P:
    base = np.load(f"artifacts/{D[name]}_thesis/predictions.npz")["test_raw"]; cb = Curve(base, yte); bb = np.array([[cb.stats(w)[k] for k in keys] for w in W]); pb = np.array([cb.stats(np.ones(len(yte)))[k] for k in keys])
    for vname, cols in [("19 features (stored)", FEATURES), ("25 features (Table 3.5)", F25), ("24 features (25 without HasHTTPS)", F24)]:
        if cols is FEATURES: pt, bs = pb, bb
        else:
            m = mk(name, P[name]).fit(tr[cols].astype("float32").values, ytr); s = m.predict_proba(te[cols].astype("float32").values)[:, 1]; del m
            c = Curve(s, yte); pt = np.array([c.stats(np.ones(len(yte)))[k] for k in keys]); bs = np.array([[c.stats(w)[k] for k in keys] for w in W])
        for j, k in enumerate(keys):
            d = bs[:, j] - bb[:, j]
            rows.append({"Model": name, "Feature set": vname, "n features": len(cols), "Metric": k, "Value": pt[j], "CI lower": np.quantile(bs[:, j], .025), "CI upper": np.quantile(bs[:, j], .975),
                         "Delta vs 19": pt[j] - pb[j], "Delta CI lower": np.quantile(d, .025) if cols is not FEATURES else np.nan, "Delta CI upper": np.quantile(d, .975) if cols is not FEATURES else np.nan})
        print(f"[{(time.time()-T0)/60:5.1f} min] {name} {vname}", flush=True)
pd.DataFrame(rows).to_csv("artifacts/e7_features25/E7_features25.csv", index=False)
json.dump({"F25": F25, "F24": F24, "dropped": DROP, "added": EXTRA}, open("artifacts/e7_features25/meta.json", "w"), indent=1); print("DONE")
