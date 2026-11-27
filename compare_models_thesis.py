"""Paired comparison of LightGBM, XGBoost and Random Forest (Chapter 3 protocol).
Tests: (1) Nadeau-Bengio corrected resampled t-test on the 5 grouped CV folds (PR-AUC),
       (2) DeLong test for ROC-AUC on the locked test set,
       (3) McNemar test on decisions at each model's frozen tau_high,
       (4) paired domain-cluster bootstrap CI for the difference in PR-AUC / ROC-AUC / TPR@1%FPR.
Holm correction is applied within each test family across the 3 model pairs."""
import itertools, json
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score
from train_rf_thesis import Curve

M = {"LightGBM": "artifacts/lgb_thesis", "XGBoost": "artifacts/xgb_thesis", "Random Forest": "artifacts/rf_thesis"}
P = {k: np.load(f"{v}/predictions.npz") for k, v in M.items()}
names = list(M); pairs = list(itertools.combinations(names, 2))
y = P[names[0]]["y_test"]; ytr = P[names[0]]["y_train"]; fid = P[names[0]]["fold_id"]
for k in names:
    assert (P[k]["y_test"] == y).all() and (P[k]["fold_id"] == fid).all(), f"{k}: test order / folds differ"

def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj

# ---------- (1) corrected resampled t-test on grouped CV folds
fold_ap = {k: np.array([average_precision_score(ytr[fid == f], P[k]["oof_raw"][fid == f]) for f in range(5)]) for k in names}
rows = []
for a, b in pairs:
    d = fold_ap[a] - fold_ap[b]; kf = len(d); var = d.var(ddof=1)
    se = np.sqrt((1 / kf + 1 / (kf - 1)) * var)           # n_test/n_train = 1/(k-1)
    t = d.mean() / se if se > 0 else 0.0; p = 2 * stats.t.sf(abs(t), kf - 1)
    rows.append({"Pair": f"{a} vs {b}", "Mean dPR-AUC": d.mean(), "SD of fold differences": d.std(ddof=1), "t (df=4)": t, "p": p})
t1 = pd.DataFrame(rows); t1["p (Holm)"] = holm(t1.p)
t1["Folds won by first"] = [f"{int((fold_ap[a] > fold_ap[b]).sum())}/5" for a, b in pairs]

# ---------- (2) DeLong (fast, Sun & Xu 2014)
def midrank(x):
    o = np.argsort(x); z = x[o]; n = len(x); t = np.zeros(n); i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]: j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1; i = j
    out = np.empty(n); out[o] = t; return out
def delong_components(s):
    pos = s[y == 1]; neg = s[y == 0]; m, n = len(pos), len(neg)
    tx, ty, tz = midrank(pos), midrank(neg), midrank(np.r_[pos, neg])
    return (tz[:m].sum() / m / n - (m + 1) / 2 / n), (tz[:m] - tx) / n, 1 - (tz[m:] - ty) / m
comp = {k: delong_components(P[k]["test_raw"]) for k in names}
rows = []
for a, b in pairs:
    (aa, v10a, v01a), (ab, v10b, v01b) = comp[a], comp[b]
    m, n = len(v10a), len(v01a)
    S10 = np.cov(np.vstack([v10a, v10b])); S01 = np.cov(np.vstack([v01a, v01b]))
    S = S10 / m + S01 / n; L = np.array([1, -1])
    var = L @ S @ L; z = (aa - ab) / np.sqrt(var); p = 2 * stats.norm.sf(abs(z))
    rows.append({"Pair": f"{a} vs {b}", "AUC first": aa, "AUC second": ab, "dAUC": aa - ab, "z": z, "p": p})
t2 = pd.DataFrame(rows); t2["p (Holm)"] = holm(t2.p)

# ---------- (3) McNemar at each model's frozen tau_high
rows = []
for a, b in pairs:
    ca = P[a]["test_pred"] == y; cb = P[b]["test_pred"] == y
    n01 = int((ca & ~cb).sum()); n10 = int((~ca & cb).sum())     # only a right / only b right
    p = stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    rows.append({"Pair": f"{a} vs {b}", "Only first correct": n01, "Only second correct": n10,
                 "Acc first": ca.mean(), "Acc second": cb.mean(), "p (exact)": p})
t3 = pd.DataFrame(rows); t3["p (Holm)"] = holm(t3["p (exact)"])

# ---------- (4) paired domain-cluster bootstrap
dom = P[names[0]]["test_domain_idx"]; nd = dom.max() + 1
curves = {k: Curve(P[k]["test_raw"], y) for k in names}
B = 2000; rng = np.random.default_rng(42); keys = ["ROC-AUC", "PR-AUC", "TPR@1%FPR"]
boot = {k: [] for k in names}
for _ in range(B):
    w = rng.multinomial(nd, np.full(nd, 1 / nd))[dom].astype(float)
    for k in names:
        s = curves[k].stats(w); boot[k].append([s[x] for x in keys])
boot = {k: np.array(v) for k, v in boot.items()}
pt = {k: np.array([curves[k].stats(np.ones(len(y)))[x] for x in keys]) for k in names}
rows = []
for a, b in pairs:
    for j, mname in enumerate(keys):
        dd = boot[a][:, j] - boot[b][:, j]
        p = min(1.0, 2 * min((dd <= 0).mean(), (dd >= 0).mean()) + 1 / B)
        rows.append({"Pair": f"{a} vs {b}", "Metric": mname, "Diff (first - second)": pt[a][j] - pt[b][j],
                     "CI lower": np.quantile(dd, .025), "CI upper": np.quantile(dd, .975), "Excludes 0": bool(np.quantile(dd, .025) > 0 or np.quantile(dd, .975) < 0), "p (bootstrap)": p})
t4 = pd.DataFrame(rows)
t4["p (Holm)"] = np.concatenate([holm(t4[t4.Metric == m]["p (bootstrap)"].values) for m in keys])[np.argsort(np.concatenate([np.flatnonzero(t4.Metric == m) for m in keys]))]

# ---------- ranking table
summ = {k: json.load(open(f"{v}/report/summary.json")) for k, v in M.items()}
rows = []
for k in names:
    t = summ[k]["test"]
    rows.append({"Model": k, "CV PR-AUC": summ[k]["best_cv_pr_auc"], "ROC-AUC": t["ROC-AUC"], "PR-AUC": t["PR-AUC"], "TPR@1%FPR": t["TPR@1%FPR"],
                 "TPR@0.1%FPR": t["TPR@0.1%FPR"], "Precision": t["Precision"], "Recall": t["Recall (TPR)"], "F1": t["F1-score"],
                 "FPR": t["FPR"], "ECE": summ[k]["ece_cal"], "Coverage": summ[k]["coverage"]})
t0 = pd.DataFrame(rows)

import os; os.makedirs("artifacts/comparison", exist_ok=True)
for n, t in [("C0_model_summary", t0), ("C1_corrected_t_test", t1), ("C2_delong", t2), ("C3_mcnemar", t3), ("C4_paired_bootstrap", t4)]:
    t.to_csv(f"artifacts/comparison/{n}.csv", index=False); print("\n==", n); print(t.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
