"""E2: feature-group ablation for LightGBM, XGBoost and Random Forest.
Hyperparameters are the ones tuned on the full feature set (not re-tuned per subset), all models are refit on the
training partition and evaluated once on the locked test partition. Ranking metrics are used (no threshold needed).
Differences against the full model carry paired domain-cluster bootstrap CIs (same resampling weights for all configs)."""
import json, sys, time
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from train_rf_thesis import FEATURES, SEED, Curve
from train_boost_thesis import build

TH = int(sys.argv[1]) if len(sys.argv) > 1 else 16
B = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
GROUPS = {
 "G1 Length ratios (HostRatio, PathRatio, SubdomainRatio)": ["HostRatio", "PathRatio", "SubdomainRatio"],
 "G2 Structure counts (Dots, Hyphens, Slashes, QueryParams)": ["NoOfDots", "NoOfHyphens", "NoOfSlashes", "NoOfQueryParams"],
 "G3 Character composition (Digits, DigitRatio, SpecialCharRatio, ObfuscatedChar)": ["NoOfDigits", "DigitRatio", "SpecialCharRatio", "NoOfObfuscatedChar"],
 "G4 Entropy (HostEntropy, PathEntropy)": ["HostEntropy", "PathEntropy"],
 "G5 Phishing indicators (At, Punycode, NonStdPort, Brand, Shortener, EmbeddedURL)": ["HasAtSymbol", "IsPunycode", "HasNonStandardPort", "HasBrandInSubdomainOrPath", "IsShortener", "HasEmbeddedURL"]}
assert sorted(sum(GROUPS.values(), [])) == sorted(FEATURES)

df = pd.read_csv("Dataset_Hybrid.csv")
tr = df[df.partition == "train"].reset_index(drop=True); te = df[df.partition == "test"].reset_index(drop=True)
ytr, yte = tr.Label.values.astype(int), te.Label.values.astype(int)
pw = float((ytr == 0).sum() / (ytr == 1).sum())
_, dom = np.unique(te.registered_domain.values, return_inverse=True); nd = dom.max() + 1
rng = np.random.default_rng(SEED); W = [rng.multinomial(nd, np.full(nd, 1 / nd))[dom].astype(float) for _ in range(B)]

params = {"Random Forest": json.load(open("artifacts/rf_thesis/best_params.json")),
          "LightGBM": json.load(open("artifacts/lgb_thesis/report/summary.json"))["best_params"],
          "XGBoost": json.load(open("artifacts/xgb_thesis/report/summary.json"))["best_params"]}
def make(name):
    p = params[name]
    if name == "LightGBM": return build("lgb", p, TH, pw)
    if name == "XGBoost": return build("xgb", p, TH, pw)
    mf = p["max_features"]; mf = float(mf) if mf not in ("sqrt", "log2") else mf
    return RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"],
        min_samples_split=p["min_samples_split"], max_features=mf, class_weight="balanced", n_jobs=TH, random_state=SEED)

configs = [("Full model (19 features)", FEATURES)]
for g, cols in GROUPS.items(): configs.append((f"Drop {g.split(' (')[0]}", [f for f in FEATURES if f not in cols]))
for g, cols in GROUPS.items(): configs.append((f"Only {g.split(' (')[0]}", cols))
keys = ["ROC-AUC", "PR-AUC", "TPR@1%FPR", "TPR@0.1%FPR"]
T0 = time.time(); rows = []; scores = {}
for name in params:
    for cname, cols in configs:
        m = make(name).fit(tr[cols].astype("float32").values, ytr)
        s = m.predict_proba(te[cols].astype("float32").values)[:, 1]; c = Curve(s, yte)
        point = c.stats(np.ones(len(yte))); bs = np.array([[c.stats(w)[k] for k in keys] for w in W])
        scores[(name, cname)] = (np.array([point[k] for k in keys]), bs)
        print(f"[{(time.time()-T0)/60:5.1f} min] {name:14s} {cname:28s} PR-AUC {point['PR-AUC']:.4f} TPR@1% {point['TPR@1%FPR']:.4f}", flush=True)
        del m
    pf, bf = scores[(name, configs[0][0])]
    for cname, cols in configs:
        p, b = scores[(name, cname)]
        for j, k in enumerate(keys):
            d = b[:, j] - bf[:, j]
            rows.append({"Model": name, "Configuration": cname, "n features": len(cols), "Metric": k, "Value": p[j],
                         "CI lower": np.quantile(b[:, j], .025), "CI upper": np.quantile(b[:, j], .975),
                         "Delta vs full": p[j] - pf[j], "Delta CI lower": np.quantile(d, .025) if cname != configs[0][0] else np.nan,
                         "Delta CI upper": np.quantile(d, .975) if cname != configs[0][0] else np.nan})
import os; os.makedirs("artifacts/e2_ablation", exist_ok=True)
pd.DataFrame(rows).to_csv("artifacts/e2_ablation/E2_ablation.csv", index=False)
json.dump({"groups": GROUPS, "params": params, "bootstrap": B}, open("artifacts/e2_ablation/meta.json", "w"), indent=1, default=str)
print("DONE", (time.time() - T0) / 60)
