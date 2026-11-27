"""LightGBM / XGBoost under the same Chapter-3 protocol as train_rf_thesis.py.

Same data, same grouped folds (StratifiedGroupKFold, seed 42), Optuna TPE 50 trials with NO pruner,
isotonic calibration on OOF, frozen thresholds, one test pass, domain-cluster bootstrap.
Predictions (OOF + test) are saved so that compare_models_thesis.py can run the paired tests.

Usage: python train_boost_thesis.py --model lgb --output-dir artifacts/lgb_thesis --threads 10
       python train_boost_thesis.py --model xgb --output-dir artifacts/xgb_thesis --threads 10
Smoke: add --smoke
"""
import argparse, hashlib, json, platform, time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from train_rf_thesis import (FEATURES, SEED, Curve, ece_score, pick_thresholds, risk_levels,
                             threshold_metrics)


def build(kind, p, threads, pos_weight):
    if kind == "lgb":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=p["n_estimators"], num_leaves=p["num_leaves"], max_depth=p["max_depth"],
                              learning_rate=p["learning_rate"], colsample_bytree=p["feature_fraction"],
                              subsample=p["bagging_fraction"], subsample_freq=1, min_child_samples=p["min_child_samples"],
                              class_weight="balanced", n_jobs=threads, random_state=SEED, verbose=-1)
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"], learning_rate=p["learning_rate"],
                         subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
                         min_child_weight=p["min_child_weight"], reg_lambda=p["reg_lambda"],
                         scale_pos_weight=pos_weight, tree_method="hist", n_jobs=threads,
                         random_state=SEED, eval_metric="logloss", verbosity=0)


def suggest(kind, trial, smoke):
    hi = 60 if smoke else 1000
    if kind == "lgb":                      # Table 3.7 search space
        unl = trial.suggest_categorical("depth_unlimited", [True, False])
        return {"n_estimators": trial.suggest_int("n_estimators", min(100, hi), hi),
                "num_leaves": trial.suggest_int("num_leaves", 15, 255),
                "max_depth": -1 if unl else trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100)}
    return {"n_estimators": trial.suggest_int("n_estimators", min(100, hi), hi),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lgb", "xgb"], required=True)
    ap.add_argument("--data", default="Dataset_Hybrid.csv")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke: a.trials, a.boot = 3, 100
    kind = a.model
    out = Path(a.output_dir); rep = out / "report"; rep.mkdir(parents=True, exist_ok=True)
    T0 = time.time()
    def log(m): print(f"[{(time.time()-T0)/60:6.1f} min] {m}", flush=True)

    df = pd.read_csv(a.data)
    if a.smoke:
        df = pd.concat([g.sample(min(len(g), 15000 if n == "train" else 4000), random_state=1) for n, g in df.groupby("partition")])
    tr = df[df.partition == "train"].reset_index(drop=True); te = df[df.partition == "test"].reset_index(drop=True)
    assert not (set(tr.registered_domain) & set(te.registered_domain))
    Xtr, ytr = tr[FEATURES].astype("float32").values, tr.Label.values.astype(int)
    Xte, yte = te[FEATURES].astype("float32").values, te.Label.values.astype(int)
    gtr = tr.registered_domain.values
    hard_te = ((te.IsShortener == 1) | (te.HasEmbeddedURL == 1)).astype(int).values
    pos_weight = float((ytr == 0).sum() / (ytr == 1).sum())
    log(f"data ok: train {len(tr)}, test {len(te)}, shared domains 0")

    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Xtr, ytr, gtr))
    fold_id = np.zeros(len(ytr), int)
    for k, (_, vai) in enumerate(folds): fold_id[vai] = k

    def objective(trial):
        p = suggest(kind, trial, a.smoke); sc = []
        for tri, vai in folds:
            m = build(kind, p, a.threads, pos_weight).fit(Xtr[tri], ytr[tri])
            sc.append(average_precision_score(ytr[vai], m.predict_proba(Xtr[vai])[:, 1])); del m
        return float(np.mean(sc))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    db = out / "optuna_study.db"
    if db.exists(): db.unlink()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.NopPruner(), study_name=f"{kind}_thesis",
                                storage=f"sqlite:///{db.as_posix()}")
    t_tune = time.time()
    study.optimize(objective, n_trials=a.trials,
                   callbacks=[lambda s, t: log(f"trial {t.number} PR-AUC={t.value:.4f} best={s.best_value:.4f}")])
    tune_min = (time.time() - t_tune) / 60
    best = study.best_trial; bp = suggest(kind, optuna.trial.FixedTrial(best.params), a.smoke)
    log(f"best trial #{best.number}: {bp}")

    oof = np.zeros(len(ytr)); rows = []
    for k, (tri, vai) in enumerate(folds, 1):
        m = build(kind, bp, a.threads, pos_weight).fit(Xtr[tri], ytr[tri]); oof[vai] = m.predict_proba(Xtr[vai])[:, 1]; del m
        rows.append({"Fold": k, "Validation URLs": len(vai), **Curve(oof[vai], ytr[vai]).stats(np.ones(len(vai)))})
    r3 = pd.DataFrame(rows)
    r3.loc[len(r3)] = {"Fold": "Mean ± SD", "Validation URLs": np.nan,
                       **{c: f"{r3[c].mean():.4f} ± {r3[c].std(ddof=1):.4f}" for c in ["ROC-AUC", "PR-AUC", "TPR@1%FPR", "TPR@0.1%FPR"]}}
    r3.to_csv(rep / "R3_oof_folds.csv", index=False)
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(oof, ytr)
    oof_cal = iso.predict(oof); tau_high, tau_low = pick_thresholds(ytr, oof_cal)
    oof_fpr = float(np.mean(oof_cal[ytr == 0] >= tau_high)); oof_fnr = float(np.mean(oof_cal[ytr == 1] <= tau_low))
    log(f"tau_high={tau_high:.4f} (OOF FPR {oof_fpr:.4%}), tau_low={tau_low:.4f} (OOF FNR {oof_fnr:.4%})")

    t = time.time(); model = build(kind, bp, a.threads, pos_weight).fit(Xtr, ytr); log(f"refit {time.time()-t:.0f}s")
    raw = model.predict_proba(Xte)[:, 1]; cal = iso.predict(raw)
    pred = (cal >= tau_high).astype(int); lvl = risk_levels(cal, hard_te, tau_high, tau_low)

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
    pd.DataFrame({"Metric": order, "Value": [point[k] for k in order],
                  "CI lower": [bs[k].quantile(0.025) for k in order], "CI upper": [bs[k].quantile(0.975) for k in order]}
                 ).to_csv(rep / "R4_test_metrics.csv", index=False)
    log(f"TEST ROC {point['ROC-AUC']:.4f} PR {point['PR-AUC']:.4f} TPR@1% {point['TPR@1%FPR']:.4f}")

    names = ["Safe", "Suspicious", "Malicious"]; r5 = []
    for c, nm in [(0, "Benign"), (1, "Malicious")]:
        m = yte == c; cnt = [int((lvl[m] == k).sum()) for k in range(3)]
        r5.append({"True class": nm, **{names[k]: f"{cnt[k]:,} ({cnt[k]/m.sum():.1%})" for k in range(3)}, "Total": int(m.sum())})
    pd.DataFrame(r5).to_csv(rep / "R5_risk_levels.csv", index=False)
    definite = lvl != 1
    f1_def = threshold_metrics(yte[definite], (lvl[definite] == 2).astype(int), cal[definite], np.ones(definite.sum()))["F1-score"]

    # predictions for the paired tests
    np.savez_compressed(out / "predictions.npz", fold_id=fold_id, y_train=ytr, oof_raw=oof, oof_cal=oof_cal,
                        y_test=yte, test_raw=raw, test_cal=cal, test_pred=pred, level=lvl,
                        test_domain_idx=dom_idx, tau_high=tau_high, tau_low=tau_low)
    version = hashlib.sha256((json.dumps(bp, sort_keys=True, default=str) + kind + str(SEED)).encode()).hexdigest()[:12]
    joblib.dump({"model": model, "calibrator": iso, "thresholds": {"tau_high": tau_high, "tau_low": tau_low},
                 "feature_names": FEATURES, "training": {"model_type": kind, "best_params": bp, "version": version}},
                out / "url_phishing_model.joblib", compress=3)
    import lightgbm, xgboost, sklearn
    summary = {"model": kind, "version": version, "best_trial": best.number, "best_cv_pr_auc": best.value, "best_params": bp,
               "tau_high": tau_high, "tau_low": tau_low, "oof_fpr_at_tau_high": oof_fpr, "oof_fnr_at_tau_low": oof_fnr,
               "test": {k: float(point[k]) for k in order},
               "confusion_at_tau_high": {"TP": int(((pred == 1) & (yte == 1)).sum()), "FP": int(((pred == 1) & (yte == 0)).sum()),
                                         "FN": int(((pred == 0) & (yte == 1)).sum()), "TN": int(((pred == 0) & (yte == 0)).sum())},
               "coverage": float(np.mean(lvl != 1)), "f1_on_definite": float(f1_def),
               "ece_raw": ece_score(yte, raw), "ece_cal": ece_score(yte, cal),
               "brier_raw": float(np.mean((raw - yte) ** 2)), "brier_cal": float(np.mean((cal - yte) ** 2)),
               "pruned_trials": sum(t.state.name == "PRUNED" for t in study.trials), "completed_trials": sum(t.state.name == "COMPLETE" for t in study.trials),
               "tuning_minutes": tune_min, "total_minutes": (time.time() - T0) / 60, "bootstrap": a.boot,
               "env": {"python": platform.python_version(), "lightgbm": lightgbm.__version__, "xgboost": xgboost.__version__,
                       "sklearn": sklearn.__version__, "platform": platform.platform()}}
    (rep / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"DONE. artifacts in {out}")


if __name__ == "__main__":
    main()
