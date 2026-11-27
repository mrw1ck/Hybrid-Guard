"""E8: Random Forest trained on URL features only, QR features only and URL+QR.

Same protocol as train_rf_thesis.py: grouped 5-fold OOF -> isotonic calibration -> frozen tau_high/tau_low
-> single test pass -> domain-cluster bootstrap. Hyper-parameters are the ones tuned for the 19 URL features
(artifacts/rf_thesis/best_params.json), NOT re-tuned per configuration (same convention as E2).
Usage: python e8_qr_vs_url.py [n_jobs] [bootstrap]
"""
import json, os, sys, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from urllib.parse import urlsplit
from train_rf_thesis import (FEATURES, SEED, Curve, make_rf, threshold_metrics, ece_score,
                             pick_thresholds, risk_levels)

TH = int(sys.argv[1]) if len(sys.argv) > 1 else 16
B = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
OUT = "artifacts/e8_qr"; os.makedirs(OUT, exist_ok=True)
QR_ALL = ["QRVersion", "QRDimension", "QRTotalModules", "QRBlackModules", "QRWhiteModules", "QRDotDensity", "QRECCLevel",
          "QRDataMask", "QRUnusedECC", "QRPayloadLength", "QRCapacityRatio", "QRModulePixelSize", "QRImageAspectRatio", "HasValidQR"]

df = pd.read_csv("Dataset_Hybrid.csv")
tr = df[df.partition == "train"].reset_index(drop=True); te = df[df.partition == "test"].reset_index(drop=True)
assert not (set(tr.registered_domain) & set(te.registered_domain))
QR = [c for c in QR_ALL if tr[c].nunique() > 1]
CONST = [c for c in QR_ALL if c not in QR]
print("QR used:", len(QR), "constant (dropped):", CONST, flush=True)
ytr, yte = tr.Label.values.astype(int), te.Label.values.astype(int)
hard_te = ((te.IsShortener == 1) | (te.HasEmbeddedURL == 1)).astype(int).values
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(tr, ytr, tr.registered_domain.values))
bp = json.load(open("artifacts/rf_thesis/best_params.json"))
_, dom = np.unique(te.registered_domain.values, return_inverse=True); nd = dom.max() + 1
rng = np.random.default_rng(SEED); W = [rng.multinomial(nd, np.full(nd, 1 / nd))[dom].astype(float) for _ in range(B)]
KEYS = ["ROC-AUC", "PR-AUC", "TPR@1%FPR", "TPR@0.1%FPR"]

def len_feats(urls):
    p = [urlsplit(u) for u in urls]
    return np.c_[[len(u) for u in urls], [len(x.path) for x in p], [u.count("/") for u in urls], [1 if x.query else 0 for x in p]]

CONFIGS = [  # name, columns, model kind
    ("URL only (19)", FEATURES, "rf"),
    ("QR only (12)", QR, "rf"),
    ("URL + QR (31)", FEATURES + QR, "rf"),
    ("QRPayloadLength only (1)", ["QRPayloadLength"], "rf"),
]
T0 = time.time(); res = {}; rows = []; risk_rows = []
def log(m): print(f"[{(time.time()-T0)/60:6.1f} min] {m}", flush=True)

def evaluate(name, Xtr, Xte, fit):
    oof = np.zeros(len(ytr))
    for tri, vai in folds:
        oof[vai] = fit().fit(Xtr[tri], ytr[tri]).predict_proba(Xtr[vai])[:, 1]
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(oof, ytr)
    th, tl = pick_thresholds(ytr, iso.predict(oof))
    m = fit().fit(Xtr, ytr); raw = m.predict_proba(Xte)[:, 1]; cal = iso.predict(raw)
    pred = (cal >= th).astype(int); lvl = risk_levels(cal, hard_te, th, tl)
    c = Curve(raw, yte)
    def allm(w):
        d = c.stats(w); d.update(threshold_metrics(yte, pred, cal, w)); return d
    point = allm(np.ones(len(yte))); bs = pd.DataFrame([allm(w) for w in W])
    res[name] = dict(raw=raw, cal=cal, point=point, bs=bs, model=m, th=th, tl=tl)
    oof_ps = Curve(oof, ytr).stats(np.ones(len(ytr)))
    for k in ["ROC-AUC", "PR-AUC", "TPR@1%FPR", "TPR@0.1%FPR", "Precision", "Recall (TPR)", "F1-score", "MCC", "FPR"]:
        rows.append({"Configuration": name, "Metric": k, "Value": point[k], "CI lower": bs[k].quantile(.025), "CI upper": bs[k].quantile(.975)})
    rows.append({"Configuration": name, "Metric": "CV PR-AUC (OOF, pooled)", "Value": oof_ps["PR-AUC"], "CI lower": np.nan, "CI upper": np.nan})
    rows.append({"Configuration": name, "Metric": "ECE (calibrated)", "Value": ece_score(yte, cal), "CI lower": np.nan, "CI upper": np.nan})
    rows.append({"Configuration": name, "Metric": "Coverage (Safe+Malicious share)", "Value": float(np.mean(lvl != 1)), "CI lower": np.nan, "CI upper": np.nan})
    for cl, nm in [(0, "Benign"), (1, "Malicious")]:
        mk = yte == cl
        risk_rows.append({"Configuration": name, "True class": nm, **{n: float(np.mean(lvl[mk] == k)) for k, n in enumerate(["Safe", "Suspicious", "Malicious"])}})
    log(f"{name}: ROC {point['ROC-AUC']:.4f} PR {point['PR-AUC']:.4f} TPR@1% {point['TPR@1%FPR']:.4f} tau_high {th:.3f}")

for name, cols, _ in CONFIGS:
    Xtr, Xte = tr[cols].astype("float32").values, te[cols].astype("float32").values
    evaluate(name, Xtr, Xte, lambda: make_rf(bp, n_jobs=TH))

# reference baselines (cheap)
Ltr, Lte = len_feats(tr.url.values), len_feats(te.url.values)
evaluate("URL-length-only tree (depth 2)", Ltr, Lte, lambda: DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=SEED))
Xtr19, Xte19 = tr[FEATURES].astype("float32").values, te[FEATURES].astype("float32").values
evaluate("Logistic regression (19 URL)", Xtr19, Xte19, lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight="balanced")))

R = pd.DataFrame(rows); R.to_csv(f"{OUT}/E8_metrics.csv", index=False)
pd.DataFrame(risk_rows).to_csv(f"{OUT}/E8_risk_levels.csv", index=False)

# paired differences vs URL only
base = res["URL only (19)"]["bs"]; bp0 = res["URL only (19)"]["point"]; dr = []
for name in res:
    if name == "URL only (19)": continue
    for k in KEYS:
        d = res[name]["bs"][k].values - base[k].values
        dr.append({"Configuration": name, "Metric": k, "Delta vs URL only": res[name]["point"][k] - bp0[k],
                   "CI lower": np.quantile(d, .025), "CI upper": np.quantile(d, .975), "Excludes 0": bool(np.quantile(d, .025) > 0 or np.quantile(d, .975) < 0)})
pd.DataFrame(dr).to_csv(f"{OUT}/E8_paired_delta.csv", index=False)

# QR feature diagnostics: class means, single-feature AUC, correlation with URL length
ulen_tr = tr.url.str.len().values
diag = []
for c in QR_ALL:
    a = roc_auc_score(ytr, tr[c].values) if tr[c].nunique() > 1 else np.nan
    sp = pd.Series(tr[c].values).corr(pd.Series(ulen_tr), method="spearman") if tr[c].nunique() > 1 else np.nan
    diag.append({"Feature": c, "Mean benign": tr.loc[ytr == 0, c].mean(), "Mean malicious": tr.loc[ytr == 1, c].mean(),
                 "Distinct values": tr[c].nunique(), "Single-feature AUC": a, "max(AUC,1-AUC)": max(a, 1 - a) if a == a else np.nan,
                 "Spearman vs URL length": sp})
D = pd.DataFrame(diag); D.to_csv(f"{OUT}/E8_qr_feature_diagnostics.csv", index=False)
log("QR diagnostics done")

# permutation importance on the URL+QR model (test PR-AUC drop, 3 repeats)
m = res["URL + QR (31)"]["model"]; cols = FEATURES + QR; Xte = te[cols].astype("float32").values
base_pr = Curve(m.predict_proba(Xte)[:, 1], yte).stats(np.ones(len(yte)))["PR-AUC"]; prng = np.random.default_rng(SEED); imp = []
for j, c in enumerate(cols):
    drops = []
    for _ in range(3):
        Xp = Xte.copy(); Xp[:, j] = Xp[prng.permutation(len(Xp)), j]
        drops.append(base_pr - Curve(m.predict_proba(Xp)[:, 1], yte).stats(np.ones(len(yte)))["PR-AUC"])
    imp.append({"Feature": c, "Group": "QR" if c in QR else "URL", "Permutation dPR-AUC": np.mean(drops), "SD": np.std(drops),
                "Native importance": m.feature_importances_[j]})
I = pd.DataFrame(imp).sort_values("Permutation dPR-AUC", ascending=False); I.to_csv(f"{OUT}/E8_importance_url_qr.csv", index=False)
log("permutation done")

# ----------------------------------------------------------------------------- figures
plt.rcParams.update({"font.family": "serif", "figure.dpi": 150, "axes.grid": True, "grid.alpha": .3})
names = list(res); colr = dict(zip(names, ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#7f7f7f", "#8c564b"]))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for n in names:
    s = res[n]["raw"]; o = np.argsort(-s, kind="stable"); y = yte[o]; tp = np.cumsum(y); fp = np.cumsum(1 - y)
    tpr, fpr, prec = tp / tp[-1], fp / fp[-1], tp / (tp + fp)
    ax[0].plot(fpr, tpr, color=colr[n], label=n, lw=1.3); ax[1].plot(tpr, prec, color=colr[n], lw=1.3)
ax[0].plot([0, 1], [0, 1], "k:", lw=.8); ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (test)")
ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall (test)", ylim=(0.3, 1.01)); ax[0].legend(fontsize=7, loc="lower right")
plt.tight_layout(); plt.savefig(f"{OUT}/figE8_1_roc_pr.png"); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for a_, k in zip(ax, ["PR-AUC", "TPR@1%FPR"]):
    v = [res[n]["point"][k] for n in names]; lo = [res[n]["point"][k] - res[n]["bs"][k].quantile(.025) for n in names]
    hi = [res[n]["bs"][k].quantile(.975) - res[n]["point"][k] for n in names]
    a_.barh(range(len(names)), v, xerr=[lo, hi], color=[colr[n] for n in names], capsize=3)
    a_.set_yticks(range(len(names))); a_.set_yticklabels(names, fontsize=7); a_.invert_yaxis(); a_.set(xlabel=k, title=f"{k} on test (95% CI)")
    for i, x in enumerate(v): a_.text(x + 0.01, i, f"{x:.3f}", va="center", fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/figE8_2_configs.png"); plt.close()

d2 = D.dropna(subset=["Single-feature AUC"]).sort_values("max(AUC,1-AUC)")
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].barh(d2.Feature, d2["max(AUC,1-AUC)"], color="#d62728"); ax[0].axvline(.5, color="k", ls=":"); ax[0].set(xlim=(.45, .8), xlabel="max(AUC, 1-AUC)", title="QR features: single-feature AUC (train)")
ax[1].barh(d2.Feature, d2["Spearman vs URL length"], color="#1f77b4"); ax[1].set(xlim=(-0.1, 1.05), xlabel="Spearman correlation with URL length", title="QR features vs URL length (train)")
for a_ in ax: a_.tick_params(axis="y", labelsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/figE8_3_qr_features.png"); plt.close()

top = I.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 5)); ax.barh(top.Feature, top["Permutation dPR-AUC"], xerr=top.SD, color=["#d62728" if g == "QR" else "#1f77b4" for g in top.Group])
ax.set(xlabel="Drop in test PR-AUC when permuted", title="URL + QR model: permutation importance (top 15)\nblue = URL, red = QR"); ax.tick_params(axis="y", labelsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/figE8_4_importance.png"); plt.close()
json.dump({"qr_used": QR, "qr_constant_dropped": CONST, "params": bp, "bootstrap": B, "minutes": (time.time() - T0) / 60}, open(f"{OUT}/meta.json", "w"), indent=1)
log("DONE")
