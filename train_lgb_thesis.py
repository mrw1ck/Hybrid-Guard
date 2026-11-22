#!/usr/bin/env python3
"""Train and evaluate Model 1 (LightGBM) following the thesis protocol (Chapter 3).

- Split: the domain-grouped `partition` column of Dataset.csv (no registered domain shared
  between train and test, Section 3.3.3). The test partition is used once, at the end.
- Tuning: Optuna TPE, 50 trials, 5-fold StratifiedGroupKFold by registered domain,
  objective = mean PR-AUC, search space of Table 3.7 (Section 3.5.2).
- Calibration: isotonic regression on out-of-fold predictions (Section 3.5.3).
- Thresholds: tau_high at 1% FPR and tau_low at a fixed FNR on calibrated OOF scores;
  shorteners / embedded URLs are never labelled Safe (Section 3.5.3).
- Evaluation: metrics of Section 3.6.1 with domain-cluster bootstrap CIs, base rates,
  reference models and shortcut audit (3.6.3), split sensitivity (E3), latency (E7).

Outputs go to --output-dir (default artifacts/lgb_thesis); artifacts/lgb_final is not touched.
The Optuna study is stored in SQLite, so an interrupted run resumes where it stopped.
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
import lightgbm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

from url_ml.feature_engineering import FEATURES, LISTS_DIR, extract_static_features

FPR_LEVELS = (0.01, 0.001)
BASE_RATES = (0.10, 0.05, 0.01)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thesis-protocol training of the LightGBM URL model.")
    parser.add_argument("--data", default="Dataset.csv")
    parser.add_argument("--output-dir", default="artifacts/lgb_thesis")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=2000, help="Domain-cluster bootstrap replicates.")
    parser.add_argument("--fpr-target", type=float, default=0.01, help="FPR that defines tau_high.")
    parser.add_argument("--fnr-target", type=float, default=0.02, help="FNR that defines tau_low.")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ----------------------------------------------------------------------------- metrics
class RankCurves:
    """Weighted ROC / PR curves on a fixed score order, fast enough for thousands of bootstrap replicates."""

    def __init__(self, y: np.ndarray, score: np.ndarray):
        self.order = np.argsort(-score, kind="mergesort")
        s_sorted = score[self.order]
        self.y_sorted = y[self.order].astype(float)
        self.cut = np.r_[np.where(np.diff(s_sorted))[0], len(s_sorted) - 1]

    def evaluate(self, weights: np.ndarray) -> dict:
        w = weights[self.order]
        tps = np.cumsum(w * self.y_sorted)[self.cut]
        fps = np.cumsum(w * (1.0 - self.y_sorted))[self.cut]
        tpr = np.r_[0.0, tps / tps[-1]]
        fpr = np.r_[0.0, fps / fps[-1]]
        with np.errstate(invalid="ignore", divide="ignore"):
            precision = np.where(tps + fps > 0, tps / (tps + fps), 1.0)
        recall = tps / tps[-1]
        result = {
            "ROC-AUC": float(np.trapezoid(tpr, fpr)),
            "PR-AUC": float(np.sum(np.diff(np.r_[0.0, recall]) * precision)),
        }
        for level in FPR_LEVELS:
            result[f"TPR@{level:.1%} FPR"] = float(np.interp(level, fpr, tpr))
        return result


def threshold_metrics(y: np.ndarray, pred: np.ndarray, w: np.ndarray | None = None) -> dict:
    w = np.ones(len(y)) if w is None else w
    tp = float(np.sum(w * (y == 1) * (pred == 1)))
    fp = float(np.sum(w * (y == 0) * (pred == 1)))
    fn = float(np.sum(w * (y == 1) * (pred == 0)))
    tn = float(np.sum(w * (y == 0) * (pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "Accuracy": (tp + tn) / (tp + tn + fp + fn),
        "Precision": precision,
        "Recall (TPR)": recall,
        "F1-score": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "MCC": (tp * tn - fp * fn) / den if den else 0.0,
        "FPR": fp / (fp + tn) if fp + tn else 0.0,
        "FNR": fn / (fn + tp) if fn + tp else 0.0,
    }


def ece_score(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    count = np.bincount(bins, minlength=n_bins)
    conf = np.bincount(bins, weights=p, minlength=n_bins)
    true = np.bincount(bins, weights=y, minlength=n_bins)
    mask = count > 0
    return float(np.sum(np.abs(conf[mask] - true[mask])) / len(p))


def pick_thresholds(y: np.ndarray, score: np.ndarray, fpr_target: float, fnr_target: float) -> tuple[float, float]:
    """tau_high: lowest score with FPR <= target; tau_low: highest score with FNR <= target (ties respected)."""
    candidates = np.unique(score)
    neg = np.sort(score[y == 0])
    pos = np.sort(score[y == 1])
    fpr_at = 1.0 - np.searchsorted(neg, candidates, side="left") / len(neg)  # P(neg >= t)
    fnr_at = np.searchsorted(pos, candidates, side="right") / len(pos)  # P(pos <= t)
    tau_high = float(candidates[fpr_at <= fpr_target].min())
    tau_low = float(candidates[fnr_at <= fnr_target].max())
    return tau_high, tau_low


def risk_levels(cal: np.ndarray, tau_low: float, tau_high: float, never_safe: np.ndarray) -> np.ndarray:
    levels = np.full(len(cal), "Suspicious", dtype=object)
    levels[cal >= tau_high] = "Malicious"
    levels[(cal <= tau_low) & ~never_safe] = "Safe"
    return levels


# ----------------------------------------------------------------------------- tuning
def lgb_params_from_trial(trial: optuna.Trial, seed: int) -> dict:
    limit_depth = trial.suggest_categorical("limit_max_depth", [False, True])
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 12) if limit_depth else -1,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "colsample_bytree": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "subsample": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "subsample_freq": 1,
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "class_weight": "balanced",
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def best_params(study: optuna.Study, seed: int) -> dict:
    return lgb_params_from_trial(optuna.trial.FixedTrial(study.best_params), seed)


def run_tuning(X, y, folds, args, out_dir: Path) -> optuna.Study:
    storage = f"sqlite:///{(out_dir / 'optuna_study.db').as_posix()}"
    study = optuna.create_study(
        study_name="lgb_model1",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1),
    )
    done = len([t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)])
    remaining = max(args.n_trials - done, 0)
    log(f"Optuna: {done} trials already done, running {remaining} more")

    def objective(trial: optuna.Trial) -> float:
        params = lgb_params_from_trial(trial, args.seed)
        scores = []
        for k, (tr, va) in enumerate(folds):
            model = LGBMClassifier(**params).fit(X.iloc[tr], y[tr])
            scores.append(average_precision_score(y[va], model.predict_proba(X.iloc[va])[:, 1]))
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_pr_auc", scores)
        return float(np.mean(scores))

    def callback(study_: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        value = f"{trial.value:.5f}" if trial.value is not None else "pruned"
        log(f"  trial {trial.number:>2}: {value} | best {study_.best_value:.5f} (#{study_.best_trial.number})")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if remaining:
        study.optimize(objective, n_trials=remaining, callbacks=[callback])
    return study


# ----------------------------------------------------------------------------- plots
def setup_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
        }
    )


def savefig(fig, out_dir: Path, name: str) -> None:
    fig.savefig(out_dir / "figures" / f"{name}.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    setup_plots()
    t_start = time.time()
    results: dict = {"config": vars(args)}

    # ------------------------------------------------------------------ data
    log(f"Loading {args.data}")
    df = pd.read_csv(args.data, low_memory=False)
    label_col = "Label" if "Label" in df.columns else "label"
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    y_all = df[label_col].astype(int).to_numpy()
    X_all = df[FEATURES].astype(float)
    is_train = (df["partition"] == "train").to_numpy()
    is_test = (df["partition"] == "test").to_numpy()
    groups = df["registered_domain"].astype(str).to_numpy()

    shared = set(groups[is_train]) & set(groups[is_test])
    assert not shared, f"{len(shared)} registered domains appear in both partitions"
    X_tr, y_tr, g_tr = X_all[is_train].reset_index(drop=True), y_all[is_train], groups[is_train]
    X_te, y_te, g_te = X_all[is_test].reset_index(drop=True), y_all[is_test], groups[is_test]
    df_te = df[is_test].reset_index(drop=True)
    results["data"] = {
        "rows": int(len(df)),
        "train_rows": int(is_train.sum()),
        "test_rows": int(is_test.sum()),
        "train_malicious": int(y_tr.sum()),
        "test_malicious": int(y_te.sum()),
        "train_domains": int(len(set(g_tr))),
        "test_domains": int(len(set(g_te))),
        "shared_domains": 0,
        "hard_urls_test": int(df_te["is_hard_url"].sum()) if "is_hard_url" in df_te else None,
        "scheme_added_share": float(df["scheme_added"].mean()) if "scheme_added" in df else None,
        "features": FEATURES,
    }
    log(f"Train {len(y_tr):,} URLs / {len(set(g_tr)):,} domains | test {len(y_te):,} URLs / {len(set(g_te)):,} domains")

    log("Building StratifiedGroupKFold folds on the training partition")
    sgkf = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    folds = list(sgkf.split(X_tr, y_tr, g_tr))
    for k, (a, b) in enumerate(folds):
        assert not set(g_tr[a]) & set(g_tr[b]), f"fold {k} shares domains"

    # ------------------------------------------------------------------ shortcut audit (training partition)
    audit = []
    for feature in FEATURES:
        auc = roc_auc_score(y_tr, X_tr[feature])
        audit.append({"Feature": feature, "AUC": auc, "max(AUC, 1-AUC)": max(auc, 1 - auc)})
    audit_df = pd.DataFrame(audit).sort_values("max(AUC, 1-AUC)", ascending=False)
    audit_df["Flag (>0.95)"] = audit_df["max(AUC, 1-AUC)"] > 0.95
    audit_df.to_csv(out_dir / "tables" / "single_feature_auc.csv", index=False)
    results["shortcut_audit"] = audit_df.to_dict(orient="records")

    # ------------------------------------------------------------------ tuning
    t0 = time.time()
    study = run_tuning(X_tr, y_tr, folds, args, out_dir)
    params = best_params(study, args.seed)
    trials_df = study.trials_dataframe(attrs=("number", "value", "state", "params", "duration"))
    trials_df.to_csv(out_dir / "tables" / "optuna_trials.csv", index=False)
    results["tuning"] = {
        "best_trial": study.best_trial.number,
        "best_cv_pr_auc": study.best_value,
        "best_params": study.best_params,
        "n_complete": int((trials_df["state"] == "COMPLETE").sum()),
        "n_pruned": int((trials_df["state"] == "PRUNED").sum()),
        "minutes": (time.time() - t0) / 60,
    }
    log(f"Best CV PR-AUC {study.best_value:.5f} with {study.best_params}")

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    complete = trials_df[trials_df["state"] == "COMPLETE"]
    ax.scatter(complete["number"], complete["value"], s=14, label="Completed trial")
    pruned = trials_df[trials_df["state"] == "PRUNED"]
    ax.scatter(pruned["number"], np.full(len(pruned), complete["value"].min()), marker="x", color="grey", s=14, label="Pruned trial")
    best_so_far = complete.set_index("number")["value"].cummax().reindex(trials_df["number"]).ffill()
    ax.plot(trials_df["number"], best_so_far, color="tab:red", lw=1.5, label="Best so far")
    ax.set(xlabel="Trial", ylabel="Mean CV PR-AUC", title="Optuna TPE optimisation history (LightGBM)")
    ax.legend(fontsize=8)
    savefig(fig, out_dir, "fig_optuna_history")

    # ------------------------------------------------------------------ OOF predictions, calibration, thresholds
    log("Out-of-fold predictions with the best parameters")
    oof = np.zeros(len(y_tr))
    fold_rows = []
    for k, (a, b) in enumerate(folds, start=1):
        model = LGBMClassifier(**params).fit(X_tr.iloc[a], y_tr[a])
        oof[b] = model.predict_proba(X_tr.iloc[b])[:, 1]
        curves = RankCurves(y_tr[b], oof[b]).evaluate(np.ones(len(b)))
        fold_rows.append({"Fold": k, "Validation URLs": len(b), **curves})
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out_dir / "tables" / "cv_folds.csv", index=False)
    results["cv_folds"] = fold_df.to_dict(orient="records")

    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(oof, y_tr)
    oof_cal = calibrator.predict(oof)
    tau_high, tau_low = pick_thresholds(y_tr, oof_cal, args.fpr_target, args.fnr_target)
    results["thresholds"] = {
        "tau_high": tau_high,
        "tau_low": tau_low,
        "fpr_target": args.fpr_target,
        "fnr_target": args.fnr_target,
        "oof_fpr_at_tau_high": float(np.mean(oof_cal[y_tr == 0] >= tau_high)),
        "oof_fnr_at_tau_low": float(np.mean(oof_cal[y_tr == 1] <= tau_low)),
    }
    log(f"tau_high = {tau_high:.4f}, tau_low = {tau_low:.4f}")

    # ------------------------------------------------------------------ final model
    log("Refitting the final model on the whole training partition")
    t0 = time.time()
    final_model = LGBMClassifier(**params).fit(X_tr, y_tr)
    results["final_fit_seconds"] = time.time() - t0

    # ------------------------------------------------------------------ locked test set (used once)
    log("Evaluating on the locked test partition")
    raw = final_model.predict_proba(X_te)[:, 1]
    cal = calibrator.predict(raw)
    pred = (cal >= tau_high).astype(int)
    curves = RankCurves(y_te, raw)
    ones = np.ones(len(y_te))
    point = {**curves.evaluate(ones), **threshold_metrics(y_te, pred)}
    assert abs(point["ROC-AUC"] - roc_auc_score(y_te, raw)) < 1e-9
    assert abs(point["PR-AUC"] - average_precision_score(y_te, raw)) < 1e-9
    point["Brier (raw)"] = float(np.mean((raw - y_te) ** 2))
    point["Brier (calibrated)"] = float(np.mean((cal - y_te) ** 2))
    point["ECE (raw)"] = ece_score(y_te, raw)
    point["ECE (calibrated)"] = ece_score(y_te, cal)
    point_at_05 = threshold_metrics(y_te, (raw >= 0.5).astype(int))

    log(f"Domain-cluster bootstrap ({args.n_boot} replicates)")
    dom_codes, dom_index = pd.factorize(g_te)
    rng = np.random.default_rng(args.seed)
    boot_rows = []
    for _ in range(args.n_boot):
        counts = np.bincount(rng.integers(0, len(dom_index), len(dom_index)), minlength=len(dom_index))
        w = counts[dom_codes].astype(float)
        row = {**curves.evaluate(w), **threshold_metrics(y_te, pred, w)}
        row["Brier (calibrated)"] = float(np.sum(w * (cal - y_te) ** 2) / w.sum())
        boot_rows.append(row)
    boot = pd.DataFrame(boot_rows)
    metric_order = [
        "ROC-AUC", "PR-AUC", "TPR@1.0% FPR", "TPR@0.1% FPR", "Precision", "Recall (TPR)", "F1-score",
        "MCC", "FPR", "FNR", "Accuracy", "Brier (calibrated)",
    ]
    test_table = pd.DataFrame(
        {
            "Metric": metric_order,
            "Value": [point[m] for m in metric_order],
            "CI low": [boot[m].quantile(0.025) for m in metric_order],
            "CI high": [boot[m].quantile(0.975) for m in metric_order],
        }
    )
    test_table.to_csv(out_dir / "tables" / "test_metrics.csv", index=False)
    results["test"] = {
        "point": point,
        "ci95": {m: [float(boot[m].quantile(0.025)), float(boot[m].quantile(0.975))] for m in metric_order},
        "at_threshold_0.5_uncalibrated": point_at_05,
        "confusion_at_tau_high": {
            "tn": int(np.sum((y_te == 0) & (pred == 0))),
            "fp": int(np.sum((y_te == 0) & (pred == 1))),
            "fn": int(np.sum((y_te == 1) & (pred == 0))),
            "tp": int(np.sum((y_te == 1) & (pred == 1))),
        },
    }

    # Risk levels and coverage
    never_safe = ((df_te["IsShortener"] == 1) | (df_te["HasEmbeddedURL"] == 1)).to_numpy()
    levels = risk_levels(cal, tau_low, tau_high, never_safe)
    level_counts = pd.crosstab(pd.Series(np.where(y_te == 1, "Malicious URL", "Benign URL"), name="True class"),
                               pd.Series(levels, name="Risk level")).reindex(columns=["Safe", "Suspicious", "Malicious"], fill_value=0)
    level_counts.to_csv(out_dir / "tables" / "risk_levels.csv")
    decided = levels != "Suspicious"
    covered = threshold_metrics(y_te[decided], (levels[decided] == "Malicious").astype(int))
    results["risk_levels"] = {
        "counts": {k: {c: int(v) for c, v in row.items()} for k, row in level_counts.iterrows()},
        "coverage": float(decided.mean()),
        "metrics_on_covered": covered,
        "never_safe_urls": int(never_safe.sum()),
    }

    # Base rates (Section 3.6.1)
    pos_w = np.where(y_te == 1, 1.0, 0.0)
    base_rows = []
    for pi in (0.40,) + BASE_RATES:
        neg_scale = (y_te.sum() / (len(y_te) - y_te.sum())) * (1 - pi) / pi
        weights = pos_w + np.where(y_te == 0, neg_scale, 0.0)
        tpr, fpr = point["Recall (TPR)"], point["FPR"]
        base_rows.append(
            {
                "Phishing base rate": pi,
                "Precision at tau_high": tpr * pi / (tpr * pi + fpr * (1 - pi)),
                "PR-AUC": curves.evaluate(weights)["PR-AUC"],
                "False alarms per 10,000 benign scans": fpr * 10_000,
            }
        )
    base_df = pd.DataFrame(base_rows)
    base_df.to_csv(out_dir / "tables" / "base_rates.csv", index=False)
    results["base_rates"] = base_df.to_dict(orient="records")

    # Hard-URL stratum and simple strata (Section 3.6.3)
    strata = {}
    if "is_hard_url" in df_te:
        strata["Hard URLs"] = df_te["is_hard_url"].to_numpy() == 1
        strata["Other URLs"] = df_te["is_hard_url"].to_numpy() == 0
    slashes = df_te["NoOfSlashes"].to_numpy()
    strata["Path depth 0-1"] = slashes <= 1
    strata["Path depth 2-3"] = (slashes >= 2) & (slashes <= 3)
    strata["Path depth >=4"] = slashes >= 4
    strata["Shortener / embedded URL"] = never_safe
    strata_rows = []
    for name, mask in strata.items():
        ys, ps = y_te[mask], pred[mask]
        row = {"Stratum": name, "URLs": int(mask.sum()), "Malicious share": float(ys.mean()) if mask.any() else np.nan}
        row.update({k: threshold_metrics(ys, ps)[k] for k in ("Recall (TPR)", "FPR", "FNR", "F1-score")})
        row["PR-AUC"] = average_precision_score(ys, raw[mask]) if 0 < ys.sum() < len(ys) else np.nan
        row["Safe share"] = float(np.mean(levels[mask] == "Safe"))
        strata_rows.append(row)
    strata_df = pd.DataFrame(strata_rows)
    strata_df.to_csv(out_dir / "tables" / "strata.csv", index=False)
    results["strata"] = strata_df.to_dict(orient="records")

    # ------------------------------------------------------------------ reference models (Section 3.6.3)
    log("Reference models")

    def length_features(frame: pd.DataFrame) -> pd.DataFrame:
        raw_url = frame["url_raw"].astype(str) if "url_raw" in frame else frame["url"].astype(str)
        parts = (frame["url"].astype(str)).map(lambda u: urlsplit(u))
        return pd.DataFrame(
            {
                "URLLength": raw_url.str.len(),
                "PathLength": parts.map(lambda p: len(p.path)),
                "NoOfSlashes": frame["NoOfSlashes"],
                "HasQuery": parts.map(lambda p: int(bool(p.query))),
            }
        )

    len_tr = length_features(df[is_train].reset_index(drop=True))
    len_te = length_features(df_te)
    ref_tree = DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=args.seed).fit(len_tr, y_tr)
    ref_score = ref_tree.predict_proba(len_te)[:, 1]
    ref_rows = [
        {"Model": "Majority class (always benign)", "ROC-AUC": 0.5, "PR-AUC": float(y_te.mean()),
         "TPR@1.0% FPR": 0.0, "F1-score": 0.0},
        {"Model": "URL-length-only tree (depth 2)", **{k: v for k, v in RankCurves(y_te, ref_score).evaluate(ones).items() if k != "TPR@0.1% FPR"},
         "F1-score": threshold_metrics(y_te, (ref_score >= 0.5).astype(int))["F1-score"]},
        {"Model": f"LightGBM ({len(FEATURES)} features)", "ROC-AUC": point["ROC-AUC"], "PR-AUC": point["PR-AUC"],
         "TPR@1.0% FPR": point["TPR@1.0% FPR"], "F1-score": point["F1-score"]},
    ]
    ref_df = pd.DataFrame(ref_rows)
    ref_df.to_csv(out_dir / "tables" / "reference_models.csv", index=False)
    results["reference_models"] = ref_df.to_dict(orient="records")

    # ------------------------------------------------------------------ split sensitivity (E3)
    log("Split sensitivity: random URL-level split with the same parameters")
    idx = np.arange(len(df))
    r_tr, r_te = train_test_split(idx, test_size=is_test.mean(), random_state=args.seed, stratify=y_all)
    random_model = LGBMClassifier(**params).fit(X_all.iloc[r_tr], y_all[r_tr])
    r_score = random_model.predict_proba(X_all.iloc[r_te])[:, 1]
    r_curves = RankCurves(y_all[r_te], r_score).evaluate(np.ones(len(r_te)))
    shared_share = float(pd.Series(groups[r_te]).isin(set(groups[r_tr])).mean())
    split_df = pd.DataFrame(
        [
            {"Split": "Grouped by registered domain (thesis)", "Test URLs on training domains": 0.0,
             **{k: point[k] for k in ("ROC-AUC", "PR-AUC", "TPR@1.0% FPR", "TPR@0.1% FPR")}},
            {"Split": "Random URL-level split", "Test URLs on training domains": shared_share, **r_curves},
        ]
    )
    split_df.to_csv(out_dir / "tables" / "split_sensitivity.csv", index=False)
    results["split_sensitivity"] = split_df.to_dict(orient="records")

    # ------------------------------------------------------------------ explainability
    log("Permutation importance and SHAP")
    sample_idx, _ = train_test_split(np.arange(len(y_te)), train_size=min(20_000, len(y_te) - 1), stratify=y_te, random_state=args.seed)
    perm = permutation_importance(final_model, X_te.iloc[sample_idx], y_te[sample_idx], scoring="average_precision",
                                  n_repeats=5, random_state=args.seed, n_jobs=1)
    gain = final_model.booster_.feature_importance(importance_type="gain")
    importance_df = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Permutation ΔPR-AUC (mean)": perm.importances_mean,
            "Permutation ΔPR-AUC (std)": perm.importances_std,
            "Gain share": gain / gain.sum(),
        }
    )
    try:
        import shap

        shap_x = X_te.sample(n=min(5000, len(X_te)), random_state=args.seed)
        shap_values = np.asarray(shap.TreeExplainer(final_model).shap_values(shap_x))
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        importance_df["Mean |SHAP|"] = np.abs(shap_values).mean(axis=0)
        shap.summary_plot(shap_values, shap_x, max_display=len(FEATURES), show=False, plot_size=(6.4, max(5.6, 0.24 * len(FEATURES))))
        fig = plt.gcf()
        fig.suptitle("SHAP summary on 5,000 test URLs", y=1.0, fontsize=10)
        savefig(fig, out_dir, "fig_shap_beeswarm")
    except ImportError:
        log("shap not installed, skipping SHAP")
    importance_df = importance_df.sort_values("Permutation ΔPR-AUC (mean)", ascending=False)
    importance_df.to_csv(out_dir / "tables" / "feature_importance.csv", index=False)
    results["feature_importance"] = importance_df.to_dict(orient="records")

    fig, ax = plt.subplots(figsize=(6.4, max(4.6, 0.2 * len(FEATURES))))
    order = importance_df.iloc[::-1]
    ax.barh(order["Feature"], order["Permutation ΔPR-AUC (mean)"], xerr=order["Permutation ΔPR-AUC (std)"], color="#1f77b4")
    ax.set(xlabel="Drop in PR-AUC when the feature is permuted", title="Permutation importance on held-out test URLs")
    savefig(fig, out_dir, "fig_permutation_importance")

    # ------------------------------------------------------------------ latency (E7)
    log("Latency at batch size 1")
    lat_urls = df_te["url_raw"].astype(str).sample(n=min(1050, len(df_te)), random_state=args.seed).tolist()
    timings = {"Feature extraction": [], "LightGBM inference": [], "Calibration + risk level": [], "Total": []}
    for i, url in enumerate(lat_urls):
        t_a = time.perf_counter()
        feats = extract_static_features(url)
        t_b = time.perf_counter()
        frame = pd.DataFrame([{name: feats[name] for name in FEATURES}])
        p = final_model.predict_proba(frame)[:, 1]
        t_c = time.perf_counter()
        c = calibrator.predict(p)
        _ = "Malicious" if c[0] >= tau_high else ("Safe" if c[0] <= tau_low else "Suspicious")
        t_d = time.perf_counter()
        if i >= 50:  # warm-up
            timings["Feature extraction"].append((t_b - t_a) * 1000)
            timings["LightGBM inference"].append((t_c - t_b) * 1000)
            timings["Calibration + risk level"].append((t_d - t_c) * 1000)
            timings["Total"].append((t_d - t_a) * 1000)
    latency_df = pd.DataFrame(
        [{"Stage": k, "Median (ms)": np.median(v), "p95 (ms)": np.percentile(v, 95), "p99 (ms)": np.percentile(v, 99)} for k, v in timings.items()]
    )
    latency_df.to_csv(out_dir / "tables" / "latency.csv", index=False)
    results["latency"] = latency_df.to_dict(orient="records")
    results["latency_calls"] = len(timings["Total"])

    # ------------------------------------------------------------------ figures
    log("Figures")
    y_sorted = y_te[curves.order]
    tps = np.cumsum(y_sorted)[curves.cut]
    fps = np.cumsum(1 - y_sorted)[curves.cut]
    roc_fpr, roc_tpr = np.r_[0, fps / fps[-1]], np.r_[0, tps / tps[-1]]
    pr_p, pr_r = tps / (tps + fps), tps / tps[-1]
    ref_c = RankCurves(y_te, ref_score)
    ys2 = y_te[ref_c.order]
    rt, rf = np.cumsum(ys2)[ref_c.cut], np.cumsum(1 - ys2)[ref_c.cut]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    axes[0].semilogx(np.clip(roc_fpr, 1e-5, 1), roc_tpr, lw=2, label=f"LightGBM (AUC {point['ROC-AUC']:.4f})")
    axes[0].semilogx(np.clip(np.r_[0, rf / rf[-1]], 1e-5, 1), np.r_[0, rt / rt[-1]], lw=1.5, ls="--", color="grey",
                     label="URL-length-only tree")
    axes[0].scatter([point["FPR"]], [point["Recall (TPR)"]], color="black", zorder=5, label=r"$\tau_{high}$ (frozen)")
    for level in FPR_LEVELS:
        axes[0].axvline(level, color="tab:red", ls=":", lw=1)
    axes[0].set(xlim=(1e-4, 1), ylim=(0, 1.01), xlabel="False positive rate (log scale)", ylabel="True positive rate",
                title="(a) ROC curve, locked test set")
    axes[0].legend(fontsize=7.5, loc="lower right")
    axes[1].plot(pr_r, pr_p, lw=2, label=f"LightGBM (AP {point['PR-AUC']:.4f})")
    axes[1].plot(rt / rt[-1], rt / (rt + rf), lw=1.5, ls="--", color="grey", label="URL-length-only tree")
    axes[1].axhline(y_te.mean(), color="black", lw=1, ls=":", label=f"Prevalence {y_te.mean():.2f}")
    axes[1].scatter([point["Recall (TPR)"]], [point["Precision"]], color="black", zorder=5)
    axes[1].set(xlim=(0, 1), ylim=(0, 1.01), xlabel="Recall", ylabel="Precision", title="(b) Precision-recall curve")
    axes[1].legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    savefig(fig, out_dir, "fig_roc_pr")

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    edges = np.linspace(0, 1, 16)
    for scores, name in ((raw, "Raw LightGBM"), (cal, "Isotonic-calibrated")):
        b = np.clip(np.digitize(scores, edges) - 1, 0, 14)
        cnt = np.bincount(b, minlength=15)
        mask = cnt > 0
        ax.plot((np.bincount(b, weights=scores, minlength=15)[mask] / cnt[mask]),
                (np.bincount(b, weights=y_te, minlength=15)[mask] / cnt[mask]), "o-", ms=3, lw=1.5,
                label=f"{name} (ECE {ece_score(y_te, scores):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed malicious share", title="Reliability diagram, test set")
    ax.legend(fontsize=8)
    savefig(fig, out_dir, "fig_calibration")

    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    shares = level_counts.div(level_counts.sum(axis=1), axis=0)
    left = np.zeros(len(shares))
    for level, color in (("Safe", "#4c9a2a"), ("Suspicious", "#e3a21a"), ("Malicious", "#c0392b")):
        ax.barh(shares.index, shares[level], left=left, color=color, label=level)
        for j, v in enumerate(shares[level]):
            if v > 0.04:
                ax.text(left[j] + v / 2, j, f"{v:.1%}", ha="center", va="center", color="white", fontsize=8)
        left += shares[level].to_numpy()
    ax.set(xlim=(0, 1), xlabel="Share of test URLs", title="Three-level risk output on the test set")
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    savefig(fig, out_dir, "fig_risk_levels")

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    pis = np.logspace(-3, np.log10(0.4), 60)
    tpr, fpr = point["Recall (TPR)"], point["FPR"]
    ax.semilogx(pis, tpr * pis / (tpr * pis + fpr * (1 - pis)), lw=2)
    for pi in BASE_RATES + (0.40,):
        prec = tpr * pi / (tpr * pi + fpr * (1 - pi))
        ax.scatter([pi], [prec], color="black", zorder=5)
        ax.annotate(f"{prec:.3f}", (pi, prec), textcoords="offset points", xytext=(4, -10), fontsize=8)
    ax.set(xlabel="Phishing base rate (log scale)", ylabel=r"Precision at $\tau_{high}$", ylim=(0, 1.02),
           title="Precision under different base rates")
    savefig(fig, out_dir, "fig_base_rates")

    # ------------------------------------------------------------------ artifacts
    log("Saving model artifacts")
    stats = {f: {"mean": float(X_tr[f].mean()), "std": float(X_tr[f].std(ddof=0)) or 1.0} for f in FEATURES}
    list_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(LISTS_DIR.glob("*.txt"))}
    version = hashlib.sha256(json.dumps([study.best_params, list_hashes, FEATURES], sort_keys=True).encode()).hexdigest()[:12]
    artifact = {
        "model": final_model,
        "feature_names": FEATURES,
        "feature_stats": stats,
        "calibrator": calibrator,
        "thresholds": {"tau_high": tau_high, "tau_low": tau_low},
        "metrics": {"test": point, "confusion_matrix": [[results["test"]["confusion_at_tau_high"]["tn"], results["test"]["confusion_at_tau_high"]["fp"]],
                                                        [results["test"]["confusion_at_tau_high"]["fn"], results["test"]["confusion_at_tau_high"]["tp"]]]},
        "training": {"model_type": "lgb", "random_state": args.seed, "split": "partition (grouped by registered domain)",
                     "rows": int(len(df)), "train_rows": int(len(y_tr)), "test_rows": int(len(y_te)), "params": params,
                     "version": version},
    }
    joblib.dump(artifact, out_dir / "url_phishing_model.joblib")
    final_model.booster_.save_model(str(out_dir / "lgb_model1.txt"))
    (out_dir / "feature_schema.json").write_text(
        json.dumps({"version": version, "features": FEATURES, "lists_sha256": list_hashes,
                    "thresholds": {"tau_high": tau_high, "tau_low": tau_low}}, indent=2),
        encoding="utf-8",
    )

    results["environment"] = {
        "python": platform.python_version(), "lightgbm": lightgbm.__version__, "scikit-learn": sklearn.__version__,
        "optuna": optuna.__version__, "pandas": pd.__version__, "numpy": np.__version__, "platform": platform.platform(),
        "processor": platform.processor(),
    }
    results["model_version"] = version
    results["total_minutes"] = (time.time() - t_start) / 60
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    log(f"Done in {results['total_minutes']:.1f} min. Results in {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
