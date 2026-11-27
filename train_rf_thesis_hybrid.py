"""Random Forest trained under the Chapter-3 protocol (same as the LightGBM/RF thesis reports)
on Dataset_Hybrid.csv (19 URL features + 12 QR features).

Stages: Optuna TPE tuning (grouped 5-fold, PR-AUC) -> OOF -> refit -> isotonic calibration ->
frozen thresholds -> single test evaluation with domain-cluster bootstrap -> tables/figures.
Everything is written to --output-dir (results.json, tables/*.csv, figures/*.png, model joblib).
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier

URL_FEATURES = [
    "HostRatio", "PathRatio", "NoOfDots", "NoOfHyphens", "NoOfDigits", "NoOfSlashes", "NoOfQueryParams",
    "DigitRatio", "SpecialCharRatio", "NoOfObfuscatedChar", "HostEntropy", "PathEntropy", "HasAtSymbol",
    "SubdomainRatio", "IsPunycode", "HasNonStandardPort", "HasBrandInSubdomainOrPath", "IsShortener",
    "HasEmbeddedURL",
]
QR_FEATURES = [
    "QRVersion", "QRDimension", "QRTotalModules", "QRBlackModules", "QRWhiteModules", "QRDotDensity",
    "QRECCLevel", "QRDataMask", "QRUnusedECC", "QRPayloadLength", "QRCapacityRatio", "QRModulePixelSize",
]


def tpr_at_fpr(fpr, tpr, x):
    i = np.searchsorted(fpr, x, side="right") - 1
    return float(tpr[max(i, 0)])


def curve_metrics(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return {
        "roc_auc": float(np.trapezoid(tpr, fpr)),
        "pr_auc": float(average_precision_score(y, s)),
        "tpr_1": tpr_at_fpr(fpr, tpr, 0.01),
        "tpr_01": tpr_at_fpr(fpr, tpr, 0.001),
    }


def thresh_metrics(y, s, thr):
    p = (s >= thr).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "mcc": float(mcc), "fpr": fp / max(fp + tn, 1),
            "fnr": fn / max(fn + tp, 1), "accuracy": (tp + tn) / len(y), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def ece_score(y, s, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(s, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - s[m].mean())
    return float(e)


def all_metrics(y, s, thr):
    d = curve_metrics(y, s)
    d.update(thresh_metrics(y, s, thr))
    d["brier"] = float(brier_score_loss(y, s))
    return d


def bootstrap_ci(y, s, groups, thr, reps, seed, n_jobs):
    codes, uniq = pd.factorize(groups)
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    nd = len(uniq)
    seeds = np.random.SeedSequence(seed).spawn(reps)
    keys = ["roc_auc", "pr_auc", "tpr_1", "tpr_01", "precision", "recall", "f1", "mcc", "fpr", "fnr",
            "accuracy", "brier"]

    def one(ss):
        rng = np.random.default_rng(ss)
        d = rng.integers(0, nd, nd)
        rows = order[np.concatenate([np.arange(starts[i], starts[i] + counts[i]) for i in d])]
        m = all_metrics(y[rows], s[rows], thr)
        return [m[k] for k in keys]

    res = np.array(Parallel(n_jobs=n_jobs, batch_size=8)(delayed(one)(ss) for ss in seeds))
    return {k: [float(np.percentile(res[:, i], 2.5)), float(np.percentile(res[:, i], 97.5))]
            for i, k in enumerate(keys)}


def params_from_trial(t):
    depth = None if t.suggest_categorical("depth_mode", ["unlimited", "limited"]) == "unlimited" \
        else t.suggest_int("max_depth", 8, 40)
    return dict(
        n_estimators=t.suggest_int("n_estimators", 100, 300),
        max_depth=depth,
        max_features=t.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.8]),
        min_samples_leaf=t.suggest_int("min_samples_leaf", 3, 30),
        min_samples_split=t.suggest_int("min_samples_split", 2, 20),
    )


def make_rf(p, seed):
    return RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=seed, **p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../Dataset_Hybrid.csv")
    ap.add_argument("--features", choices=["all", "url"], default="all")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--sample", type=int, default=0, help="debug: subsample training rows")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="artifacts/rf_thesis_hybrid")
    a = ap.parse_args()

    out = Path(a.output_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    feats = URL_FEATURES + (QR_FEATURES if a.features == "all" else [])
    t_all = time.time()
    R = {"features": feats, "seed": a.seed}

    df = pd.read_csv(a.data)
    df["url_len"] = df["url"].str.len()
    df["hard"] = ((df.IsShortener == 1) | (df.HasEmbeddedURL == 1)).astype(int)
    tr = df[df.partition == "train"].reset_index(drop=True)
    te = df[df.partition == "test"].reset_index(drop=True)
    if a.sample:
        tr = tr.sample(a.sample, random_state=a.seed).reset_index(drop=True)
        te = te.sample(max(a.sample // 4, 1000), random_state=a.seed).reset_index(drop=True)
    assert not set(tr.registered_domain) & set(te.registered_domain), "domain leakage"
    Xtr, ytr, gtr = tr[feats].astype(float).values, tr.Label.values, tr.registered_domain.values
    Xte, yte, gte = te[feats].astype(float).values, te.Label.values, te.registered_domain.values

    # R1 partitions
    r1 = pd.DataFrame([
        ["Training", len(tr), int(ytr.sum()), ytr.mean(), tr.registered_domain.nunique()],
        ["Test (locked)", len(te), int(yte.sum()), yte.mean(), te.registered_domain.nunique()],
        ["Total", len(tr) + len(te), int(ytr.sum() + yte.sum()), (ytr.sum() + yte.sum()) / (len(tr) + len(te)),
         pd.concat([tr, te]).registered_domain.nunique()]],
        columns=["Partition", "URLs", "Malicious URLs", "Malicious share", "Registered domains"])
    r1.to_csv(out / "tables/R1_partitions.csv", index=False)
    R["domain_overlap"] = 0

    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=a.seed).split(Xtr, ytr, gtr))

    # ---- tuning
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name="rf_hybrid", direction="maximize", sampler=optuna.samplers.TPESampler(seed=a.seed),
        pruner=optuna.pruners.NopPruner(), storage=f"sqlite:///{(out / 'optuna_study.db').as_posix()}",
        load_if_exists=True)

    def objective(t):
        p = params_from_trial(t)
        sc = []
        for tri, vai in folds:
            m = make_rf(p, a.seed).fit(Xtr[tri], ytr[tri])
            sc.append(average_precision_score(ytr[vai], m.predict_proba(Xtr[vai])[:, 1]))
        print(f"trial {t.number} PR-AUC={np.mean(sc):.4f} {p}", flush=True)
        return float(np.mean(sc))

    t0 = time.time()
    done = len([t for t in study.trials if t.state.name == "COMPLETE"])
    if done < a.trials:
        study.optimize(objective, n_trials=a.trials - done)
    R["tuning_minutes"] = (time.time() - t0) / 60
    best = study.best_trial
    bp = params_from_trial(optuna.trial.FixedTrial(best.params))
    R["best_trial"] = best.number
    R["best_cv_pr_auc"] = best.value
    R["best_params"] = bp
    vals = np.array([t.value for t in study.trials if t.value is not None])
    R["tuning_summary"] = {"n_trials": len(vals), "median": float(np.median(vals)), "best": float(vals.max()),
                           "within_0.002": int((vals >= vals.max() - 0.002).sum()), "worst": float(vals.min())}
    pd.DataFrame([{"Hyperparameter": "n_estimators", "Search range": "100-300", "Selected": bp["n_estimators"]},
                  {"Hyperparameter": "max_depth", "Search range": "unlimited or 8-40", "Selected": bp["max_depth"] or "unlimited"},
                  {"Hyperparameter": "max_features", "Search range": "sqrt, log2, 0.3, 0.5, 0.8", "Selected": bp["max_features"]},
                  {"Hyperparameter": "min_samples_leaf", "Search range": "3-30", "Selected": bp["min_samples_leaf"]},
                  {"Hyperparameter": "min_samples_split", "Search range": "2-20", "Selected": bp["min_samples_split"]}]
                 ).to_csv(out / "tables/R2_hyperparameters.csv", index=False)
    plt.figure(figsize=(6, 3.6))
    plt.plot([t.number for t in study.trials], [t.value for t in study.trials], "o", ms=4, label="trial")
    plt.plot([t.number for t in study.trials], np.maximum.accumulate([t.value for t in study.trials]), "r-", label="best so far")
    plt.xlabel("Trial"); plt.ylabel("Mean CV PR-AUC"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "figures/R1_optuna_history.png", dpi=150); plt.close()

    # ---- OOF with best params
    oof = np.zeros(len(ytr)); rows = []
    for k, (tri, vai) in enumerate(folds, 1):
        m = make_rf(bp, a.seed).fit(Xtr[tri], ytr[tri])
        oof[vai] = m.predict_proba(Xtr[vai])[:, 1]
        c = curve_metrics(ytr[vai], oof[vai])
        rows.append({"Fold": k, "Validation URLs": len(vai), "ROC-AUC": c["roc_auc"], "PR-AUC": c["pr_auc"],
                     "TPR@1%FPR": c["tpr_1"], "TPR@0.1%FPR": c["tpr_01"]})
    r3 = pd.DataFrame(rows)
    r3.loc[len(r3)] = ["Mean", np.nan] + [r3[c].mean() for c in r3.columns[2:]]
    r3.loc[len(r3)] = ["SD", np.nan] + [r3.iloc[:5][c].std(ddof=1) for c in r3.columns[2:]]
    r3.to_csv(out / "tables/R3_oof_folds.csv", index=False)

    # ---- refit, calibrate, thresholds
    model = make_rf(bp, a.seed).fit(Xtr, ytr)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof, ytr)
    oof_c = iso.predict(oof)
    cand = np.unique(oof_c)
    neg, pos = (ytr == 0), (ytr == 1)
    fpr_c = np.array([(oof_c[neg] >= c).mean() for c in cand])
    tau_high = float(cand[np.argmax(fpr_c <= 0.01)]) if (fpr_c <= 0.01).any() else float(cand[-1])
    fnr_c = np.array([(oof_c[pos] <= c).mean() for c in cand])
    ok = fnr_c <= 0.02
    tau_low = float(cand[ok][-1]) if ok.any() else float(cand[0])
    R["tau_high"], R["tau_low"] = tau_high, tau_low
    R["oof_fpr_at_tau_high"] = float((oof_c[neg] >= tau_high).mean())
    R["oof_fnr_at_tau_low"] = float((oof_c[pos] <= tau_low).mean())

    # ---- locked test (used once)
    raw = model.predict_proba(Xte)[:, 1]
    cal = iso.predict(raw)
    R["ece_raw"], R["ece_cal"] = ece_score(yte, raw), ece_score(yte, cal)
    R["brier_raw"], R["brier_cal"] = float(brier_score_loss(yte, raw)), float(brier_score_loss(yte, cal))
    M = all_metrics(yte, cal, tau_high)
    ci = bootstrap_ci(yte, cal, gte, tau_high, a.boot, a.seed, n_jobs=-1)
    R["test"], R["test_ci"], R["bootstrap_reps"] = M, ci, a.boot
    R["uncalibrated_at_0.5"] = thresh_metrics(yte, raw, 0.5)
    names = [("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"), ("TPR@1%FPR", "tpr_1"), ("TPR@0.1%FPR", "tpr_01"),
             ("Precision", "precision"), ("Recall (TPR)", "recall"), ("F1-score", "f1"), ("MCC", "mcc"),
             ("FPR", "fpr"), ("FNR", "fnr"), ("Accuracy", "accuracy"), ("Brier score (calibrated)", "brier")]
    pd.DataFrame([{"Metric": n, "Value": M[k], "CI low": ci[k][0], "CI high": ci[k][1]} for n, k in names]
                 ).to_csv(out / "tables/R4_test_metrics.csv", index=False)

    # three-level
    hard = te.hard.values == 1
    lvl = np.where(cal >= tau_high, "Malicious", np.where((cal <= tau_low) & ~hard, "Safe", "Suspicious"))
    r5 = pd.crosstab(pd.Series(np.where(yte == 1, "Malicious", "Benign"), name="True class"),
                     pd.Series(lvl, name="Level")).reindex(columns=["Safe", "Suspicious", "Malicious"], fill_value=0)
    r5.to_csv(out / "tables/R5_risk_levels.csv")
    definite = lvl != "Suspicious"
    pred_def = (lvl[definite] == "Malicious").astype(int)
    tp = ((pred_def == 1) & (yte[definite] == 1)).sum(); fp = ((pred_def == 1) & (yte[definite] == 0)).sum()
    fn = ((pred_def == 0) & (yte[definite] == 1)).sum()
    R["coverage"] = float(definite.mean())
    R["f1_definite"] = float(2 * tp / max(2 * tp + fp + fn, 1))
    (r5.div(r5.sum(1), axis=0) * 100).plot(kind="bar", stacked=True, figsize=(5, 3.6), color=["#4caf50", "#ffc107", "#e53935"])
    plt.ylabel("Share (%)"); plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig(out / "figures/R4_risk_levels.png", dpi=150); plt.close()

    # calibration plot
    bins = np.linspace(0, 1, 11); idx = np.clip(np.digitize(cal, bins[1:-1]), 0, 9)
    idr = np.clip(np.digitize(raw, bins[1:-1]), 0, 9)
    plt.figure(figsize=(4.2, 4.2)); plt.plot([0, 1], [0, 1], "k--")
    for sc, ii, lab in ((raw, idr, "raw"), (cal, idx, "isotonic")):
        xs = [sc[ii == b].mean() for b in range(10) if (ii == b).any()]
        ys = [yte[ii == b].mean() for b in range(10) if (ii == b).any()]
        plt.plot(xs, ys, "o-", label=lab)
    plt.xlabel("Predicted"); plt.ylabel("Observed"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "figures/R2_reliability.png", dpi=150); plt.close()

    # ROC / PR
    fpr, tpr, _ = roc_curve(yte, cal)
    from sklearn.metrics import precision_recall_curve
    pr_, rc_, _ = precision_recall_curve(yte, cal)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    ax[0].semilogx(np.maximum(fpr, 1e-5), tpr); ax[0].plot(M["fpr"], M["recall"], "ro")
    ax[0].set_xlabel("FPR (log)"); ax[0].set_ylabel("TPR"); ax[0].set_title("ROC")
    ax[1].plot(rc_, pr_); ax[1].plot(M["recall"], M["precision"], "ro")
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title("PR")
    plt.tight_layout(); plt.savefig(out / "figures/R3_roc_pr.png", dpi=150); plt.close()

    # R6 base rates
    fprv, tprv = M["fpr"], M["recall"]
    rows6, xs = [], [0.4, 0.1, 0.05, 0.01]
    for pi in xs:
        prec = tprv * pi / max(tprv * pi + fprv * (1 - pi), 1e-12)
        w = np.where(yte == 1, pi / yte.mean(), (1 - pi) / (1 - yte.mean()))
        rows6.append({"Phishing base rate": f"{int(pi*100)}%", "Precision at tau_high": prec,
                      "PR-AUC": float(average_precision_score(yte, cal, sample_weight=w)),
                      "False alarms / 10,000 benign scans": fprv * 10000})
    pd.DataFrame(rows6).to_csv(out / "tables/R6_base_rates.csv", index=False)
    grid = np.linspace(0.005, 0.5, 100)
    plt.figure(figsize=(5, 3.6))
    plt.semilogx(grid, tprv * grid / (tprv * grid + fprv * (1 - grid))); plt.xlabel("Phishing base rate")
    plt.ylabel("Precision at tau_high"); plt.tight_layout()
    plt.savefig(out / "figures/R5_precision_base_rate.png", dpi=150); plt.close()

    # R7 reference, R8 single feature
    lt = DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=a.seed).fit(
        tr[["url_len"]].values, ytr)
    ls = lt.predict_proba(te[["url_len"]].values)[:, 1]
    lm = curve_metrics(yte, ls)
    f1_len = thresh_metrics(yte, ls, 0.5)["f1"]
    pd.DataFrame([
        ["Majority class (always benign)", 0.5, yte.mean(), 0.0, 0.0],
        ["URL-length-only tree (depth 2)", lm["roc_auc"], lm["pr_auc"], lm["tpr_1"], f1_len],
        [f"Random Forest ({len(feats)} features)", M["roc_auc"], M["pr_auc"], M["tpr_1"], M["f1"]]],
        columns=["Model", "ROC-AUC", "PR-AUC", "TPR@1%FPR", "F1-score"]).to_csv(out / "tables/R7_reference.csv", index=False)
    sf = []
    for j, f in enumerate(feats):
        auc = roc_auc_score(ytr, Xtr[:, j]) if np.ptp(Xtr[:, j]) > 0 else 0.5
        sf.append({"Feature": f, "AUC": auc, "max(AUC,1-AUC)": max(auc, 1 - auc), "Flag (>0.95)": "Yes" if max(auc, 1 - auc) > 0.95 else "No"})
    pd.DataFrame(sf).sort_values("max(AUC,1-AUC)", ascending=False).head(10).to_csv(out / "tables/R8_single_feature_auc.csv", index=False)
    R["max_single_feature_auc"] = float(max(x["max(AUC,1-AUC)"] for x in sf))

    # R9 split sensitivity (random URL-level split, same sizes & params)
    rng = np.random.default_rng(a.seed)
    allX = np.vstack([Xtr, Xte]); ally = np.concatenate([ytr, yte]); alld = np.concatenate([gtr, gte])
    perm = rng.permutation(len(ally)); ntr = len(ytr)
    ri_tr, ri_te = perm[:ntr], perm[ntr:]
    rm = make_rf(bp, a.seed).fit(allX[ri_tr], ally[ri_tr])
    rs = iso.predict(rm.predict_proba(allX[ri_te])[:, 1])  # indicative: reuse calibrator
    rcm = curve_metrics(ally[ri_te], rs)
    seen = float(np.isin(alld[ri_te], alld[ri_tr]).mean())
    pd.DataFrame([
        ["Grouped by registered domain (thesis)", 0.0, M["roc_auc"], M["pr_auc"], M["tpr_1"], M["tpr_01"]],
        ["Random URL-level split", seen, rcm["roc_auc"], rcm["pr_auc"], rcm["tpr_1"], rcm["tpr_01"]]],
        columns=["Split", "Test URLs on training domains", "ROC-AUC", "PR-AUC", "TPR@1%FPR", "TPR@0.1%FPR"]
    ).to_csv(out / "tables/R9_split_sensitivity.csv", index=False)

    # R10 strata
    pdepth = te.NoOfSlashes.values  # proxy not used; use PathRatio-free depth from url
    depth = te.url.str.replace(r"^[a-zA-Z]+://", "", regex=True).str.split("?").str[0].str.rstrip("/").str.count("/").values
    strata = {"Hard URLs": hard, "Other URLs": ~hard, "Path depth 0-1": depth <= 1,
              "Path depth 2-3": (depth >= 2) & (depth <= 3), "Path depth >=4": depth >= 4}
    r10 = []
    for n, m in strata.items():
        y_, s_ = yte[m], cal[m]
        tm = thresh_metrics(y_, s_, tau_high)
        r10.append({"Stratum": n, "URLs": int(m.sum()), "Malicious share": float(y_.mean()), "Recall": tm["recall"],
                    "FPR": tm["fpr"], "FNR": tm["fnr"],
                    "PR-AUC": float(average_precision_score(y_, s_)) if 0 < y_.sum() < len(y_) else np.nan,
                    "Safe share": float(((s_ <= tau_low) & ~hard[m]).mean())})
    pd.DataFrame(r10).to_csv(out / "tables/R10_strata.csv", index=False)

    # R11 permutation importance on test (calibrated PR-AUC drop); 40k sample, 3 repeats
    sub = np.random.default_rng(a.seed).choice(len(yte), min(40000, len(yte)), replace=False)
    pi = permutation_importance(model, Xte[sub], yte[sub], scoring="average_precision", n_repeats=3,
                                random_state=a.seed, n_jobs=1)
    r11 = pd.DataFrame({"Feature": feats, "Permutation dPR-AUC": pi.importances_mean, "Permutation SD": pi.importances_std,
                        "Gini share": model.feature_importances_}).sort_values("Permutation dPR-AUC", ascending=False)
    r11.to_csv(out / "tables/R11_feature_importance.csv", index=False)
    plt.figure(figsize=(6, 6)); top = r11.head(20)[::-1]
    plt.barh(top.Feature, top["Permutation dPR-AUC"]); plt.xlabel("dPR-AUC when permuted"); plt.tight_layout()
    plt.savefig(out / "figures/R6_permutation_importance.png", dpi=150); plt.close()

    # R12 latency (model + calibration only; QR features need an image so extraction is not included)
    one = Xte[:1]; lat = []
    m1 = RandomForestClassifier  # noqa
    model.set_params(n_jobs=1)
    for _ in range(50): iso.predict(model.predict_proba(one)[:, 1])
    for _ in range(300):
        t = time.perf_counter(); p_ = model.predict_proba(one)[:, 1]; t1 = time.perf_counter()
        iso.predict(p_); t2 = time.perf_counter(); lat.append(((t1 - t) * 1e3, (t2 - t1) * 1e3))
    lat = np.array(lat); tot = lat.sum(1)
    pd.DataFrame([["RF inference", *np.percentile(lat[:, 0], [50, 95, 99])],
                  ["Calibration + risk level", *np.percentile(lat[:, 1], [50, 95, 99])],
                  ["Total (no feature extraction)", *np.percentile(tot, [50, 95, 99])]],
                 columns=["Stage", "Median ms", "p95 ms", "p99 ms"]).to_csv(out / "tables/R12_latency.csv", index=False)
    model.set_params(n_jobs=-1)

    # ---- save model
    version = hashlib.sha256(json.dumps([bp, a.seed, feats], default=str).encode()).hexdigest()[:12]
    R["model_version"] = version
    R["total_minutes"] = (time.time() - t_all) / 60
    stats = {f: {"mean": float(Xtr[:, j].mean()), "std": float(Xtr[:, j].std()) or 1.0} for j, f in enumerate(feats)}
    joblib.dump({"model": model, "calibrator": iso, "tau_high": tau_high, "tau_low": tau_low, "feature_names": feats,
                 "feature_stats": stats, "metrics": {k: v for k, v in M.items()},
                 "training": {"model_type": "rf", "dataset": "Dataset_Hybrid.csv", "version": version,
                              "params": bp, "seed": a.seed}},
                out / "url_phishing_model.joblib", compress=3)
    (out / "feature_schema.json").write_text(json.dumps({"features": feats}, indent=2))
    (out / "results.json").write_text(json.dumps(R, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps({k: R[k] for k in ["best_cv_pr_auc", "tau_high", "tau_low", "coverage", "total_minutes"]}, indent=1))
    print({k: round(M[k], 4) for k in ["roc_auc", "pr_auc", "tpr_1", "tpr_01", "precision", "recall", "f1", "fpr"]})


if __name__ == "__main__":
    main()
