"""Random Forest under the Chapter-3 protocol used in LightGBM_Model1_Report.docx.

Pipeline: partition check -> grouped 5-fold CV -> Optuna TPE (50 trials, no pruner)
-> refit -> isotonic calibration on OOF -> frozen thresholds -> single test pass
-> domain-cluster bootstrap -> report tables R1-R13 and figures R1-R7.

Usage:  python train_rf_thesis.py --data Dataset_Hybrid.csv --output-dir artifacts/rf_thesis
Smoke:  python train_rf_thesis.py --smoke --output-dir /tmp/rf_smoke
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier

FEATURES = [
    "HostRatio", "PathRatio", "NoOfDots", "NoOfHyphens", "NoOfDigits", "NoOfSlashes",
    "NoOfQueryParams", "DigitRatio", "SpecialCharRatio", "NoOfObfuscatedChar", "HostEntropy",
    "PathEntropy", "HasAtSymbol", "SubdomainRatio", "IsPunycode", "HasNonStandardPort",
    "HasBrandInSubdomainOrPath", "IsShortener", "HasEmbeddedURL",
]
QR_FEATURES = ["QRVersion", "QRDimension", "QRTotalModules", "QRBlackModules", "QRWhiteModules", "QRDotDensity", "QRECCLevel",
               "QRDataMask", "QRUnusedECC", "QRPayloadLength", "QRCapacityRatio", "QRModulePixelSize"]
URL_FEATURES = list(FEATURES)
FEATURES = URL_FEATURES + QR_FEATURES   # URL + QR (31)
SEED = 42
plt.rcParams.update({"font.family": "serif", "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})


# ----------------------------------------------------------------------------- metrics
class Curve:
    """Pre-sorted structure so that weighted (bootstrap) metrics cost O(n)."""

    def __init__(self, score, y):
        self.order = np.argsort(-score, kind="stable")
        s = score[self.order]
        self.y = y[self.order].astype(float)
        last = np.r_[s[1:] != s[:-1], True]
        self.idx = np.flatnonzero(last)

    def stats(self, w):
        w = w[self.order]
        tp = np.cumsum(w * self.y)[self.idx]
        fp = np.cumsum(w * (1 - self.y))[self.idx]
        P, N = tp[-1], fp[-1]
        tpr = np.r_[0.0, tp / P]
        fpr = np.r_[0.0, fp / N]
        roc = float(np.trapezoid(tpr, fpr))
        prec = tp / np.maximum(tp + fp, 1e-12)
        ap = float(np.sum(np.diff(tpr) * prec))

        def tpr_at(x):
            ok = fpr <= x
            return float(tpr[ok].max()) if ok.any() else 0.0

        return {"ROC-AUC": roc, "PR-AUC": ap, "TPR@1%FPR": tpr_at(0.01), "TPR@0.1%FPR": tpr_at(0.001)}


def threshold_metrics(y, pred, cal, w):
    tp = np.sum(w * y * pred); fp = np.sum(w * (1 - y) * pred)
    fn = np.sum(w * y * (1 - pred)); tn = np.sum(w * (1 - y) * (1 - pred))
    prec = tp / max(tp + fp, 1e-12); rec = tp / max(tp + fn, 1e-12)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    den = np.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1e-12))
    return {
        "Precision": prec, "Recall (TPR)": rec, "F1-score": f1, "MCC": (tp * tn - fp * fn) / den,
        "FPR": fp / max(fp + tn, 1e-12), "FNR": fn / max(fn + tp, 1e-12),
        "Accuracy": (tp + tn) / (tp + tn + fp + fn),
        "Brier score (calibrated)": float(np.sum(w * (cal - y) ** 2) / np.sum(w)),
    }


def ece_score(y, p, bins=15):
    ids = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = ids == b
        if m.any():
            tot += m.sum() * abs(y[m].mean() - p[m].mean())
    return tot / len(y)


def reliability(y, p, bins=15):
    ids = np.minimum((p * bins).astype(int), bins - 1)
    xs, ys = [], []
    for b in range(bins):
        m = ids == b
        if m.sum() > 0:
            xs.append(p[m].mean()); ys.append(y[m].mean())
    return np.array(xs), np.array(ys)


def pick_thresholds(y, cal, fpr_target=0.01, fnr_target=0.02):
    """tau_high: lowest calibrated score with FPR(score>=t) <= target.
    tau_low: highest calibrated score with FNR(score<=t) <= target."""
    neg = np.sort(cal[y == 0]); pos = np.sort(cal[y == 1])
    uniq = np.unique(cal)
    fpr = 1 - np.searchsorted(neg, uniq, side="left") / len(neg)
    ok = np.flatnonzero(fpr <= fpr_target)
    tau_high = float(uniq[ok[0]]) if len(ok) else 1.0
    fnr = np.searchsorted(pos, uniq, side="right") / len(pos)
    ok = np.flatnonzero(fnr <= fnr_target)
    tau_low = float(uniq[ok[-1]]) if len(ok) else 0.0
    return tau_high, tau_low


def risk_levels(cal, hard, tau_high, tau_low):
    lvl = np.full(len(cal), 1)                     # 0 Safe, 1 Suspicious, 2 Malicious
    lvl[cal >= tau_high] = 2
    lvl[(cal <= tau_low) & (hard == 0)] = 0
    return lvl


# ----------------------------------------------------------------------------- main
def make_rf(p, n_jobs=-1):
    mf = p["max_features"]
    mf = float(mf) if mf not in ("sqrt", "log2") else mf
    return RandomForestClassifier(
        n_estimators=p["n_estimators"], max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"],
        min_samples_split=p["min_samples_split"], max_features=mf, class_weight="balanced",
        n_jobs=n_jobs, random_state=SEED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Dataset_Hybrid.csv")
    ap.add_argument("--output-dir", default="artifacts/rf_qr_thesis")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--max-trees", type=int, default=300)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--shap-n", type=int, default=1000)
    ap.add_argument("--perm-repeats", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.trials, a.max_trees, a.boot, a.shap_n, a.perm_repeats = 3, 60, 100, 100, 1
    out = Path(a.output_dir); rep = out / "report"; rep.mkdir(parents=True, exist_ok=True)
    T0 = time.time(); tables = {}

    def log(m): print(f"[{(time.time()-T0)/60:6.1f} min] {m}", flush=True)
    def save(name, df): tables[name] = df; df.to_csv(rep / f"{name}.csv", index=False)

    # ---- A. data
    df = pd.read_csv(a.data)
    if a.smoke:
        df = pd.concat([g.sample(min(len(g), 15000 if n == "train" else 4000), random_state=1)
                        for n, g in df.groupby("partition")])
    tr = df[df.partition == "train"].reset_index(drop=True)
    te = df[df.partition == "test"].reset_index(drop=True)
    shared = set(tr.registered_domain) & set(te.registered_domain)
    assert not shared, f"{len(shared)} registered domains shared between train and test"
    log(f"data ok: train {len(tr)}, test {len(te)}, shared domains 0")
    Xtr, ytr = tr[FEATURES].astype("float32").values, tr.Label.values.astype(int)
    Xte, yte = te[FEATURES].astype("float32").values, te.Label.values.astype(int)
    gtr = tr.registered_domain.values
    hard_te = ((te.IsShortener == 1) | (te.HasEmbeddedURL == 1)).astype(int).values

    r1 = pd.DataFrame({"Partition": ["Training", "Test (locked)", "Total"],
        "URLs": [len(tr), len(te), len(df)],
        "Malicious URLs": [ytr.sum(), yte.sum(), ytr.sum() + yte.sum()],
        "Malicious share": [ytr.mean(), yte.mean(), (ytr.sum() + yte.sum()) / len(df)],
        "Registered domains": [tr.registered_domain.nunique(), te.registered_domain.nunique(), df.registered_domain.nunique()]})
    save("R1_partitions", r1)

    # ---- B1. folds (shared by every trial and by OOF)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Xtr, ytr, gtr))

    # ---- B2. Optuna, no pruner
    def suggest(trial):
        mf = trial.suggest_categorical("max_features", ["sqrt", "log2", "0.3", "0.5", "0.8"])
        unlimited = trial.suggest_categorical("depth_unlimited", [True, False])
        depth = None if unlimited else trial.suggest_int("max_depth", 8, 40)
        return {"n_estimators": trial.suggest_int("n_estimators", min(100, a.max_trees), a.max_trees),
                "max_depth": depth, "max_features": mf,
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20)}

    def objective(trial):
        p = suggest(trial); sc = []
        for tri, vai in folds:
            m = make_rf(p).fit(Xtr[tri], ytr[tri])
            sc.append(average_precision_score(ytr[vai], m.predict_proba(Xtr[vai])[:, 1])); del m
        return float(np.mean(sc))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    db = out / "optuna_study.db"
    if db.exists(): db.unlink()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.NopPruner(), study_name="rf_qr_thesis",
                                storage=f"sqlite:///{db.as_posix()}")
    t_tune = time.time()
    study.optimize(objective, n_trials=a.trials,
                   callbacks=[lambda s, t: log(f"trial {t.number} PR-AUC={t.value:.4f} best={s.best_value:.4f}")])
    tune_min = (time.time() - t_tune) / 60
    best = study.best_trial; bp = suggest(optuna.trial.FixedTrial(best.params))
    log(f"best trial #{best.number}: {bp}")

    # ---- B3-B5. OOF with best config, calibration, thresholds
    oof = np.zeros(len(ytr)); rows = []
    for k, (tri, vai) in enumerate(folds, 1):
        m = make_rf(bp).fit(Xtr[tri], ytr[tri]); oof[vai] = m.predict_proba(Xtr[vai])[:, 1]; del m
        st = Curve(oof[vai], ytr[vai]).stats(np.ones(len(vai)))
        rows.append({"Fold": k, "Validation URLs": len(vai), **st})
    r3 = pd.DataFrame(rows)
    r3.loc[len(r3)] = {"Fold": "Mean ± SD", "Validation URLs": np.nan,
                       **{c: f"{r3[c].mean():.4f} ± {r3[c].std(ddof=1):.4f}" for c in ["ROC-AUC", "PR-AUC", "TPR@1%FPR", "TPR@0.1%FPR"]}}
    save("R3_oof_folds", r3)
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(oof, ytr)
    oof_cal = iso.predict(oof)
    tau_high, tau_low = pick_thresholds(ytr, oof_cal)
    oof_fpr = float(np.mean(oof_cal[ytr == 0] >= tau_high)); oof_fnr = float(np.mean(oof_cal[ytr == 1] <= tau_low))
    log(f"tau_high={tau_high:.4f} (OOF FPR {oof_fpr:.4%}), tau_low={tau_low:.4f} (OOF FNR {oof_fnr:.4%})")

    # ---- B4 refit on all train; B6 single test pass
    t = time.time(); model = make_rf(bp).fit(Xtr, ytr); log(f"refit {time.time()-t:.0f}s")
    raw = model.predict_proba(Xte)[:, 1]; cal = iso.predict(raw)
    pred = (cal >= tau_high).astype(int)
    lvl = risk_levels(cal, hard_te, tau_high, tau_low)

    # ---- C3 domain-cluster bootstrap
    curve = Curve(raw, yte)
    dom_codes, dom_idx = np.unique(te.registered_domain.values, return_inverse=True); nd = len(dom_codes)

    def all_metrics(w):
        d = curve.stats(w); d.update(threshold_metrics(yte, pred, cal, w)); return d

    point = all_metrics(np.ones(len(yte)))
    rng = np.random.default_rng(SEED); samples = []
    for _ in range(a.boot):
        cnt = rng.multinomial(nd, np.full(nd, 1 / nd)); samples.append(all_metrics(cnt[dom_idx].astype(float)))
    bs = pd.DataFrame(samples)
    order = ["ROC-AUC", "PR-AUC", "TPR@1%FPR", "TPR@0.1%FPR", "Precision", "Recall (TPR)", "F1-score", "MCC",
             "FPR", "FNR", "Accuracy", "Brier score (calibrated)"]
    r4 = pd.DataFrame({"Metric": order, "Value": [point[k] for k in order],
                       "CI lower": [bs[k].quantile(0.025) for k in order], "CI upper": [bs[k].quantile(0.975) for k in order]})
    save("R4_test_metrics", r4)
    tp = int(((pred == 1) & (yte == 1)).sum()); fp = int(((pred == 1) & (yte == 0)).sum())
    fn = int(((pred == 0) & (yte == 1)).sum()); tn = int(((pred == 0) & (yte == 0)).sum())
    p05 = (raw >= 0.5).astype(int)
    unc05 = threshold_metrics(yte, p05, raw, np.ones(len(yte)))
    log(f"TEST ROC {point['ROC-AUC']:.4f} PR {point['PR-AUC']:.4f} TPR@1% {point['TPR@1%FPR']:.4f}")

    # calibration (ECE/Brier)
    ece_raw, ece_cal = ece_score(yte, raw), ece_score(yte, cal)
    brier_raw = float(np.mean((raw - yte) ** 2)); brier_cal = float(np.mean((cal - yte) ** 2))
    save("R2b_calibration", pd.DataFrame({"": ["raw", "isotonic"], "ECE": [ece_raw, ece_cal], "Brier": [brier_raw, brier_cal]}))

    # ---- R5 risk levels
    names = ["Safe", "Suspicious", "Malicious"]
    r5 = []
    for c, nm in [(0, "Benign"), (1, "Malicious")]:
        m = yte == c; cnt = [int((lvl[m] == k).sum()) for k in range(3)]
        r5.append({"True class": nm, **{names[k]: f"{cnt[k]:,} ({cnt[k]/m.sum():.1%})" for k in range(3)}, "Total": int(m.sum())})
    save("R5_risk_levels", pd.DataFrame(r5))
    coverage = float(np.mean(lvl != 1))
    definite = lvl != 1
    f1_def = threshold_metrics(yte[definite], (lvl[definite] == 2).astype(int), cal[definite], np.ones(definite.sum()))["F1-score"]

    # ---- R6 base rates
    tpr_h, fpr_h = point["Recall (TPR)"], point["FPR"]; pos_n, neg_n = yte.sum(), (yte == 0).sum()
    rows = []
    for b in [0.4, 0.1, 0.05, 0.01]:
        w = np.where(yte == 1, 1.0, (1 - b) / b * pos_n / neg_n)
        rows.append({"Phishing base rate": f"{b:.0%}", "Precision at tau_high": tpr_h * b / (tpr_h * b + fpr_h * (1 - b)),
                     "PR-AUC": curve.stats(w)["PR-AUC"], "False alarms / 10,000 benign scans": fpr_h * 1e4})
    save("R6_base_rates", pd.DataFrame(rows))

    # ---- R7/R8 shortcut audit + length-only reference
    auc = {f: roc_auc_score(ytr, Xtr[:, i]) for i, f in enumerate(FEATURES)}
    r8 = pd.DataFrame({"Feature": list(auc), "AUC": list(auc.values())})
    r8["max(AUC,1-AUC)"] = np.maximum(r8.AUC, 1 - r8.AUC); r8["Flag (>0.95)"] = np.where(r8["max(AUC,1-AUC)"] > 0.95, "Yes", "No")
    r8 = r8.sort_values("max(AUC,1-AUC)", ascending=False); save("R8_single_feature_auc", r8)

    def len_feats(urls):
        parts = [urlsplit(u) for u in urls]
        return np.c_[[len(u) for u in urls], [len(p.path) for p in parts], [u.count("/") for u in urls], [1 if p.query else 0 for p in parts]]
    dt = DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=SEED).fit(len_feats(tr.url.values), ytr)
    ref = dt.predict_proba(len_feats(te.url.values))[:, 1]
    ref_stats = Curve(ref, yte).stats(np.ones(len(yte)))
    ref_f1 = threshold_metrics(yte, (ref >= 0.5).astype(int), ref, np.ones(len(yte)))["F1-score"]
    maj = threshold_metrics(yte, np.zeros(len(yte), int), np.zeros(len(yte)), np.ones(len(yte)))
    r7 = pd.DataFrame([
        {"Model": "Majority class (always benign)", "ROC-AUC": 0.5, "PR-AUC": yte.mean(), "TPR@1%FPR": 0.0, "F1-score": 0.0},
        {"Model": "URL-length-only tree (depth 2)", "ROC-AUC": ref_stats["ROC-AUC"], "PR-AUC": ref_stats["PR-AUC"], "TPR@1%FPR": ref_stats["TPR@1%FPR"], "F1-score": ref_f1},
        {"Model": "Random Forest (19 URL + 12 QR features)", "ROC-AUC": point["ROC-AUC"], "PR-AUC": point["PR-AUC"], "TPR@1%FPR": point["TPR@1%FPR"], "F1-score": point["F1-score"]}])
    save("R7_reference_models", r7)

    # ---- E3 split sensitivity
    rs = np.random.default_rng(SEED).permutation(len(df)); nte = len(te)
    all_X = df[FEATURES].astype("float32").values; all_y = df.Label.values.astype(int); all_d = df.registered_domain.values
    rte, rtr = rs[:nte], rs[nte:]
    seen = float(np.mean(np.isin(all_d[rte], np.unique(all_d[rtr]))))
    m_rand = make_rf(bp).fit(all_X[rtr], all_y[rtr]); rscore = m_rand.predict_proba(all_X[rte])[:, 1]; del m_rand
    rst = Curve(rscore, all_y[rte]).stats(np.ones(nte))
    r9 = pd.DataFrame([
        {"Split": "Grouped by registered domain (thesis)", "Test URLs on training domains": 0.0, **{k: point[k] for k in ["ROC-AUC", "PR-AUC", "TPR@1%FPR", "TPR@0.1%FPR"]}},
        {"Split": "Random URL-level split", "Test URLs on training domains": seen, **rst}])
    save("R9_split_sensitivity", r9)
    log(f"split sensitivity done (seen domains {seen:.1%})")

    # ---- R10 strata
    depth = te.NoOfSlashes.values - 2 if False else None
    from_url = np.array([max(0, len([s for s in urlsplit(u).path.split("/") if s])) for u in te.url.values])
    strata = {"Hard URLs": hard_te == 1, "Other URLs": hard_te == 0, "Path depth 0-1": from_url <= 1,
              "Path depth 2-3": (from_url >= 2) & (from_url <= 3), "Path depth >=4": from_url >= 4}
    rows = []
    for nm, m in strata.items():
        ym = yte[m]; w1 = np.ones(m.sum())
        tm = threshold_metrics(ym, pred[m], cal[m], w1)
        pr = Curve(raw[m], ym).stats(w1)["PR-AUC"] if 0 < ym.sum() < len(ym) else np.nan
        rows.append({"Stratum": nm, "URLs": int(m.sum()), "Malicious share": ym.mean(), "Recall": tm["Recall (TPR)"],
                     "FPR": tm["FPR"], "FNR": tm["FNR"], "PR-AUC": pr, "Safe share": float(np.mean(lvl[m] == 0))})
    save("R10_strata", pd.DataFrame(rows))

    # ---- R11 importance
    def perm_imp():
        rng = np.random.default_rng(SEED); base = average_precision_score(yte, raw); res = []
        for i in range(len(FEATURES)):
            d = []
            for _ in range(a.perm_repeats):
                Xp = Xte.copy(); Xp[:, i] = rng.permutation(Xp[:, i])
                d.append(base - average_precision_score(yte, model.predict_proba(Xp)[:, 1]))
            res.append((np.mean(d), np.std(d)))
        return res
    pi = perm_imp(); log("permutation importance done")
    shap_vals = None; shap_mean = np.full(len(FEATURES), np.nan); shap_idx = None
    try:
        import shap
        shap_idx = np.random.default_rng(SEED).choice(len(yte), min(a.shap_n, len(yte)), replace=False)
        t = time.time(); sv = shap.TreeExplainer(model).shap_values(Xte[shap_idx], check_additivity=False)
        sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
        shap_vals = sv; shap_mean = np.abs(sv).mean(0); log(f"shap {time.time()-t:.0f}s")
    except Exception as e:
        log(f"SHAP skipped: {e}")
    r11 = pd.DataFrame({"Feature": FEATURES, "Permutation dPR-AUC": [p[0] for p in pi], "Permutation SD": [p[1] for p in pi],
                        "Gini share": model.feature_importances_, "Mean |SHAP| (prob. units)": shap_mean}).sort_values("Permutation dPR-AUC", ascending=False)
    save("R11_feature_importance", r11)

    # ---- R12 latency (batch size 1, 50 warm-up + 1000 calls, n_jobs=1)
    model.set_params(n_jobs=1); xs = Xte[:1050]; lat_inf, lat_cal = [], []
    for i in range(1050):
        x = xs[i:i + 1]; t0 = time.perf_counter(); p = model.predict_proba(x)[:, 1]; t1 = time.perf_counter()
        c = iso.predict(p); _ = risk_levels(c, np.zeros(1), tau_high, tau_low); t2 = time.perf_counter()
        if i >= 50: lat_inf.append((t1 - t0) * 1e3); lat_cal.append((t2 - t1) * 1e3)
    model.set_params(n_jobs=-1)
    tot = np.array(lat_inf) + np.array(lat_cal)
    q = lambda v: [np.median(v), np.percentile(v, 95), np.percentile(v, 99)]
    r12 = pd.DataFrame([["Feature extraction (not measured: extractor not in repo)", np.nan, np.nan, np.nan],
                        ["Random Forest inference", *q(lat_inf)], ["Calibration + risk level", *q(lat_cal)], ["Total (excl. feature extraction)", *q(tot)]],
                       columns=["Stage", "Median ms", "p95 ms", "p99 ms"])
    save("R12_latency", r12)

    # ---- R13 environment
    save("R13_environment", pd.DataFrame({"Component": ["Python", "scikit-learn", "Optuna", "pandas / NumPy", "Platform"],
        "Version": [platform.python_version(), sklearn.__version__, optuna.__version__, f"{pd.__version__} / {np.__version__}", platform.platform()]}))

    # ================= figures
    th = study.trials
    fig, ax = plt.subplots(figsize=(6.5, 3.6)); vals = [t.value for t in th]
    ax.scatter(range(len(vals)), vals, s=14, label="Completed trial"); ax.plot(np.maximum.accumulate(vals), color="tab:red", label="Best so far")
    ax.set(xlabel="Trial", ylabel="Mean CV PR-AUC", title="Optuna TPE optimisation history (Random Forest)"); ax.legend(); fig.tight_layout(); fig.savefig(rep / "figR1_optuna_history.png"); plt.close()

    fig, ax = plt.subplots(figsize=(5, 4.6))
    for s, nm, e in [(raw, "Raw Random Forest", ece_raw), (cal, "Isotonic-calibrated", ece_cal)]:
        x, y_ = reliability(yte, s); ax.plot(x, y_, "o-", ms=3, label=f"{nm} (ECE {e:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set(xlabel="Mean predicted probability", ylabel="Observed malicious share", title="Reliability diagram, test set"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(rep / "figR2_reliability.png"); plt.close()

    def roc_pr(s):
        o = np.argsort(-s, kind="stable"); ss = s[o]; yy = yte[o]; last = np.r_[ss[1:] != ss[:-1], True]
        tp_ = np.cumsum(yy)[last]; fp_ = np.cumsum(1 - yy)[last]
        return np.r_[0, fp_ / fp_[-1]], np.r_[0, tp_ / tp_[-1]], tp_ / (tp_ + fp_), tp_ / tp_[-1]
    f1_, t1_, p1_, r1_ = roc_pr(raw); f2_, t2_, p2_, r2_ = roc_pr(ref)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    axs[0].plot(f1_, t1_, lw=2, label=f"Random Forest (AUC {point['ROC-AUC']:.4f})"); axs[0].plot(f2_, t2_, "--", c="gray", label="URL-length-only tree")
    axs[0].scatter([point["FPR"]], [point["Recall (TPR)"]], c="k", zorder=5, label=r"$\tau_{high}$ (frozen)")
    for x in (1e-3, 1e-2): axs[0].axvline(x, c="red", ls=":", lw=1)
    axs[0].set(xscale="log", xlim=(1e-4, 1), ylim=(0, 1), xlabel="False positive rate (log scale)", ylabel="True positive rate", title="(a) ROC curve, locked test set"); axs[0].legend(loc="lower right", fontsize=8)
    axs[1].plot(r1_, p1_, lw=2, label=f"Random Forest (AP {point['PR-AUC']:.4f})"); axs[1].plot(r2_, p2_, "--", c="gray", label="URL-length-only tree")
    axs[1].scatter([point["Recall (TPR)"]], [point["Precision"]], c="k", zorder=5); axs[1].axhline(yte.mean(), c="k", ls=":", lw=1, label=f"Prevalence {yte.mean():.2f}")
    axs[1].set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision", title="(b) Precision-recall curve"); axs[1].legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(rep / "figR3_roc_pr.png"); plt.close()

    fig, ax = plt.subplots(figsize=(7, 3.6)); cols = ["#4c9a2a", "#e3a21a", "#c0392b"]
    for yi, (c, nm) in enumerate([(1, "Malicious URL"), (0, "Benign URL")]):
        m = yte == c; left = 0
        for k in range(3):
            sh = np.mean(lvl[m] == k); ax.barh(yi, sh, left=left, color=cols[k], label=names[k] if yi == 0 else None)
            if sh > 0.05: ax.text(left + sh / 2, yi, f"{sh:.1%}", ha="center", va="center", color="w", fontsize=8)
            left += sh
    ax.set(yticks=[0, 1], yticklabels=["Malicious URL", "Benign URL"], xlim=(0, 1), xlabel="Share of test URLs", title="Three-level risk output on the test set")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2)); fig.tight_layout(); fig.savefig(rep / "figR4_risk_levels.png"); plt.close()

    bb = np.logspace(-3, np.log10(0.5), 200); pp = tpr_h * bb / (tpr_h * bb + fpr_h * (1 - bb))
    fig, ax = plt.subplots(figsize=(5.6, 3.6)); ax.plot(bb, pp, lw=2)
    for b in [0.4, 0.1, 0.05, 0.01]:
        v = tpr_h * b / (tpr_h * b + fpr_h * (1 - b)); ax.scatter([b], [v], c="k", zorder=5); ax.annotate(f"{v:.3f}", (b, v), textcoords="offset points", xytext=(6, -12), fontsize=8)
    ax.set(xscale="log", ylim=(0, 1.02), xlabel="Phishing base rate (log scale)", ylabel=r"Precision at $\tau_{high}$", title="Precision under different base rates")
    fig.tight_layout(); fig.savefig(rep / "figR5_base_rate.png"); plt.close()

    s = r11.sort_values("Permutation dPR-AUC")
    fig, ax = plt.subplots(figsize=(7, 5.5)); ax.barh(s.Feature, s["Permutation dPR-AUC"], xerr=s["Permutation SD"])
    ax.set(xlabel="Drop in PR-AUC when the feature is permuted", title="Permutation importance on held-out test URLs"); fig.tight_layout(); fig.savefig(rep / "figR6_permutation.png"); plt.close()
    if shap_vals is not None:
        shap.summary_plot(shap_vals, Xte[shap_idx], feature_names=FEATURES, show=False, max_display=19)
        plt.title(f"SHAP summary on {len(shap_idx):,} test URLs", fontsize=10); plt.tight_layout(); plt.savefig(rep / "figR7_shap.png"); plt.close()

    # ================= artifacts
    version = hashlib.sha256((json.dumps(bp, sort_keys=True, default=str) + str(SEED) + str(len(tr))).encode()).hexdigest()[:12]
    mu, sd = tr[FEATURES].mean(), tr[FEATURES].std()
    artifact = {
        "model": model, "calibrator": iso, "thresholds": {"tau_high": tau_high, "tau_low": tau_low},
        "feature_names": FEATURES,
        "feature_stats": {f: {"mean": float(mu[f]), "std": float(sd[f])} for f in FEATURES},
        "training": {"model_type": "rf_qr", "random_state": SEED, "best_params": bp, "version": version,
                     "cv": "StratifiedGroupKFold(5) by registered_domain", "tuning": f"Optuna TPE {a.trials} trials, no pruner"}}
    joblib.dump(artifact, out / "url_phishing_model.joblib", compress=3)
    (out / "feature_schema.json").write_text(json.dumps({"features": FEATURES, "label": "Label", "version": version,
        "thresholds": {"tau_high": tau_high, "tau_low": tau_low}, "never_safe_if": ["IsShortener==1", "HasEmbeddedURL==1"]}, indent=2))
    (out / "best_params.json").write_text(json.dumps(bp, indent=2, default=str))
    summary = {
        "version": version, "best_trial": best.number, "best_cv_pr_auc": best.value, "best_params": bp,
        "tau_high": tau_high, "tau_low": tau_low, "oof_fpr_at_tau_high": oof_fpr, "oof_fnr_at_tau_low": oof_fnr,
        "test": {k: float(point[k]) for k in order}, "confusion_at_tau_high": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "uncalibrated_at_0.5": {k: float(unc05[k]) for k in ["Precision", "Recall (TPR)", "F1-score", "FPR"]},
        "coverage": coverage, "f1_on_definite": float(f1_def), "ece_raw": ece_raw, "ece_cal": ece_cal, "brier_raw": brier_raw, "brier_cal": brier_cal,
        "n_hard_test": int(hard_te.sum()), "split_sensitivity_seen_domains": seen, "tuning_minutes": tune_min,
        "total_minutes": (time.time() - T0) / 60, "bootstrap": a.boot}
    (rep / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"DONE. artifacts in {out}")


if __name__ == "__main__":
    main()
