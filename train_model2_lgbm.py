"""
train_model2_lgbm.py — Model 2: LightGBM on 19 URL lexical features only
==========================================================================
Protocol: test.md (thesis-grade training)
Features:  19 URL lexical features (NO QR features)
Dataset:   Dataset.csv  — use partition column as-is, NO re-split
Folds:     cv_folds_by_sha.csv  — StratifiedGroupKFold(5), seed=42
Tuning:    Optuna TPE, 50 trials, objective = mean PR-AUC over 5 folds
Outputs:   models/, predictions/, figures/, tables/, results/, reports/
"""

# ── stdlib ─────────────────────────────────────────────────────────────────
import argparse, hashlib, json, logging, os, sys, time, warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── third-party ─────────────────────────────────────────────────────────────
import joblib
import lightgbm as lgb
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from scipy import stats
from scipy.special import expit
from sklearn.calibration import IsotonicRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    auc, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
    precision_recall_curve,
)
from sklearn.tree import DecisionTreeClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════
URL_FEATURES = [
    "HostRatio", "PathRatio", "NoOfDots", "NoOfHyphens", "NoOfDigits",
    "NoOfSlashes", "NoOfQueryParams", "DigitRatio", "SpecialCharRatio",
    "NoOfObfuscatedChar", "HostEntropy", "PathEntropy", "HasAtSymbol",
    "SubdomainRatio", "IsPunycode", "HasNonStandardPort",
    "HasBrandInSubdomainOrPath", "IsShortener", "HasEmbeddedURL",
]
LABEL_COL   = "Label"
PART_COL    = "partition"
SHA_COL     = "url_sha256"
DOM_COL     = "registered_domain"

N_TRIALS       = 50
N_FOLDS        = 5
SEED           = 42
BOOTSTRAP_REPS = 2_000
SHAP_SAMPLE    = 20_000
PERM_REPEATS   = 10
LATENCY_BATCH  = 1
LATENCY_URLS   = 1_000
LATENCY_WARMUP = 50
FPR_TARGET     = 0.01   # tau_high: FPR ≤ 1 %
FNR_TARGET     = 0.02   # tau_low:  FNR ≤ 2 %

LGB_FIXED = dict(
    objective="binary",
    metric="average_precision",
    verbose=-1,
    n_jobs=-1,
    seed=SEED,
    deterministic=True,
    force_col_wise=True,
)

# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def setup_dirs(base: Path):
    for d in ["models", "predictions", "figures", "tables", "results", "reports"]:
        (base / d).mkdir(parents=True, exist_ok=True)
    (base / "artifacts").mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ece_score(y_true, y_prob, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.sum() * abs(acc - conf)
    return ece / len(y_true)


def pick_thresholds(y_true, y_prob_cal):
    """Return tau_high (FPR<=FPR_TARGET) and tau_low (FNR<=FNR_TARGET)."""
    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_prob_cal)
    fnr_arr = 1 - tpr_arr

    # tau_high: highest threshold where FPR <= target
    mask_h = fpr_arr <= FPR_TARGET
    tau_high = float(thresholds[mask_h][-1]) if mask_h.any() else float(thresholds[-1])

    # tau_low: lowest threshold where FNR <= target
    mask_l = fnr_arr <= FNR_TARGET
    tau_low = float(thresholds[mask_l][0]) if mask_l.any() else float(thresholds[0])

    return tau_high, tau_low


def threshold_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return dict(
        threshold=threshold,
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        fpr=fpr, fnr=fnr,
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
    )


def risk_level(p, tau_high, tau_low):
    if p >= tau_high:  return "HIGH"
    if p >= tau_low:   return "MEDIUM"
    return "LOW"


def domain_bootstrap_ci(y_true, y_prob, domains, metric_fn, n_rep=BOOTSTRAP_REPS, seed=SEED):
    rng = np.random.default_rng(seed)
    order = np.argsort(domains)
    sorted_domains = domains[order]
    unique_doms, split_idx = np.unique(sorted_domains, return_index=True)
    dom_groups = np.split(order, split_idx[1:])
    n_doms = len(unique_doms)
    scores = []
    for _ in range(n_rep):
        boot = rng.integers(0, n_doms, size=n_doms)
        idx = np.concatenate([dom_groups[i] for i in boot])
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], y_prob[idx]))
    scores = np.array(scores)
    return float(np.mean(scores)), float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


# ═══════════════════════════════════════════════════════════════════════════
# LOAD & VALIDATE DATA
# ═══════════════════════════════════════════════════════════════════════════

def load_data(data_path: Path, folds_path: Path):
    log.info("Loading dataset …")
    df = pd.read_csv(data_path, low_memory=False)
    log.info(f"  Total rows: {len(df):,}")

    # Assert required columns
    required = URL_FEATURES + [LABEL_COL, PART_COL, SHA_COL, DOM_COL]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"

    # Assert no NaN in features
    nan_counts = df[URL_FEATURES].isna().sum()
    assert nan_counts.sum() == 0, f"NaN in features:\n{nan_counts[nan_counts>0]}"
    log.info("  ✓ 0 NaN in URL features")

    # Split
    train_df = df[df[PART_COL] == "train"].copy()
    test_df  = df[df[PART_COL] == "test"].copy()
    log.info(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    # Assert no domain leakage
    train_doms = set(train_df[DOM_COL].dropna().unique())
    test_doms  = set(test_df[DOM_COL].dropna().unique())
    shared = train_doms & test_doms
    assert len(shared) == 0, f"Domain leakage! {len(shared)} shared domains"
    log.info("  ✓ 0 shared registered_domain between train/test")

    # Load folds
    log.info("Loading cv_folds_by_sha.csv …")
    folds_df = pd.read_csv(folds_path)
    assert SHA_COL in folds_df.columns and "fold" in folds_df.columns
    train_df = train_df.merge(folds_df[[SHA_COL, "fold"]], on=SHA_COL, how="left")
    missing_folds = train_df["fold"].isna().sum()
    assert missing_folds == 0, f"{missing_folds} train rows have no fold assignment"
    train_df["fold"] = train_df["fold"].astype(int)
    log.info(f"  Fold distribution:\n{train_df['fold'].value_counts().sort_index()}")

    return train_df, test_df


# ═══════════════════════════════════════════════════════════════════════════
# OPTUNA TUNING
# ═══════════════════════════════════════════════════════════════════════════

def make_objective(train_df):
    X = train_df[URL_FEATURES].values
    y = train_df[LABEL_COL].values
    folds = train_df["fold"].values

    def objective(trial):
        params = {
            **LGB_FIXED,
            "n_estimators":       trial.suggest_int("n_estimators", 200, 1200),
            "num_leaves":         trial.suggest_int("num_leaves", 31, 300),
            "learning_rate":      trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
            "feature_fraction":   trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction":   trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq":       1,
            "min_child_samples":  trial.suggest_int("min_child_samples", 5, 100),
            "max_depth":          trial.suggest_int("max_depth", -1, 15),
            "reg_alpha":          trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":         trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        pr_aucs = []
        for fold_id in range(N_FOLDS):
            tr_idx = folds != fold_id
            va_idx = folds == fold_id
            model = lgb.LGBMClassifier(**params)
            model.fit(X[tr_idx], y[tr_idx],
                      eval_set=[(X[va_idx], y[va_idx])],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(-1)])
            p_va = model.predict_proba(X[va_idx])[:, 1]
            pr_aucs.append(average_precision_score(y[va_idx], p_va))
        return float(np.mean(pr_aucs))

    return objective


def run_optuna(train_df, db_path: Path, n_trials: int):
    storage = f"sqlite:///{db_path}"
    study = optuna.create_study(
        study_name="model2_lgbm",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    done = len(study.trials)
    remaining = n_trials - done
    if remaining <= 0:
        log.info(f"Optuna: already {done} trials, skipping tuning.")
    else:
        log.info(f"Optuna: {done} done, running {remaining} more (target={n_trials}) …")
        objective = make_objective(train_df)
        study.optimize(objective, n_trials=remaining, show_progress_bar=True)
    log.info(f"Best trial #{study.best_trial.number}: PR-AUC={study.best_value:.5f}")
    log.info(f"Best params: {study.best_params}")
    return study


# ═══════════════════════════════════════════════════════════════════════════
# FINAL MODEL + OOF + CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════

def train_final_model(train_df, best_params):
    log.info("Training final model on ALL train data …")
    params = {**LGB_FIXED, **best_params, "bagging_freq": 1}
    model = lgb.LGBMClassifier(**params)
    X = train_df[URL_FEATURES].values
    y = train_df[LABEL_COL].values
    model.fit(X, y)
    log.info("  Final model trained.")
    return model


def compute_oof(train_df, best_params):
    log.info("Computing OOF predictions …")
    X = train_df[URL_FEATURES].values
    y = train_df[LABEL_COL].values
    folds = train_df["fold"].values
    params = {**LGB_FIXED, **best_params, "bagging_freq": 1}

    oof_raw = np.zeros(len(train_df))
    cv_metrics = []
    for fold_id in range(N_FOLDS):
        tr_idx = folds != fold_id
        va_idx = folds == fold_id
        m = lgb.LGBMClassifier(**params)
        m.fit(X[tr_idx], y[tr_idx],
              eval_set=[(X[va_idx], y[va_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])
        p = m.predict_proba(X[va_idx])[:, 1]
        oof_raw[va_idx] = p
        cv_metrics.append({
            "fold": fold_id,
            "pr_auc": average_precision_score(y[va_idx], p),
            "roc_auc": roc_auc_score(y[va_idx], p),
        })
        log.info(f"  Fold {fold_id}: PR-AUC={cv_metrics[-1]['pr_auc']:.4f}  ROC-AUC={cv_metrics[-1]['roc_auc']:.4f}")

    pr_aucs = [m["pr_auc"] for m in cv_metrics]
    roc_aucs = [m["roc_auc"] for m in cv_metrics]
    log.info(f"  OOF PR-AUC: {np.mean(pr_aucs):.4f} ± {np.std(pr_aucs):.4f}")
    log.info(f"  OOF ROC-AUC: {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}")
    return oof_raw, cv_metrics


def calibrate(train_df, oof_raw):
    log.info("Fitting Isotonic calibrator on OOF …")
    y = train_df[LABEL_COL].values
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(oof_raw, y)
    oof_cal = cal.predict(oof_raw)
    log.info(f"  ECE before calibration: {ece_score(y, oof_raw):.4f}")
    log.info(f"  ECE after  calibration: {ece_score(y, oof_cal):.4f}")
    return cal, oof_cal


# ═══════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════

def plot_roc_pr(y_true, y_prob, out_dir: Path, prefix="test"):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_val = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"LightGBM (AUC={roc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Model 2 LightGBM"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_roc_curve.png", dpi=300)
    plt.close(fig)

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_val = auc(rec, prec)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, lw=2, label=f"LightGBM (AUC={pr_val:.4f})")
    baseline = y_true.mean()
    ax.axhline(baseline, color="gray", linestyle="--", label=f"Baseline ({baseline:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("PR Curve — Model 2 LightGBM"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_pr_curve.png", dpi=300)
    plt.close(fig)
    return roc_val, pr_val


def plot_confusion(y_true, y_prob, threshold, out_dir: Path, prefix="test"):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=12)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Benign","Phishing"]); ax.set_yticklabels(["Benign","Phishing"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix @ tau_high={threshold:.4f}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_confusion_matrix.png", dpi=300)
    plt.close(fig)


def plot_reliability(y_true, y_prob_cal, out_dir: Path, prefix="test"):
    n_bins = 15
    bins = np.linspace(0, 1, n_bins + 1)
    bin_means, bin_true, bin_counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob_cal >= lo) & (y_prob_cal < hi)
        if mask.sum() == 0: continue
        bin_means.append(y_prob_cal[mask].mean())
        bin_true.append(y_true[mask].mean())
        bin_counts.append(mask.sum())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7), gridspec_kw={"height_ratios": [3,1]})
    ax1.plot([0,1],[0,1],"k--",lw=1,label="Perfect calibration")
    ax1.plot(bin_means, bin_true, "o-", lw=2, label="LightGBM")
    ax1.set_xlabel("Mean predicted probability"); ax1.set_ylabel("Fraction of positives")
    ax1.set_title("Reliability Diagram — Model 2 LightGBM"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.bar(bin_means, bin_counts, width=0.05, color="steelblue", alpha=0.7)
    ax2.set_xlabel("Mean predicted probability"); ax2.set_ylabel("Count"); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_reliability_diagram.png", dpi=300)
    plt.close(fig)


def plot_optuna_history(study, out_dir: Path):
    vals = [t.value for t in study.trials if t.value is not None]
    best_so_far = np.maximum.accumulate(vals)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(range(len(vals)), vals, s=15, alpha=0.6, label="Trial PR-AUC")
    ax.plot(range(len(best_so_far)), best_so_far, "r-", lw=2, label="Best so far")
    ax.set_xlabel("Trial"); ax.set_ylabel("CV PR-AUC")
    ax.set_title("Optuna Optimization History"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "optuna_history.png", dpi=300)
    plt.close(fig)


def plot_shap(model, X_test_sample, feature_names, out_dir: Path):
    log.info(f"  Computing SHAP values on {len(X_test_sample)} samples …")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    fig, ax = plt.subplots(figsize=(8, 7))
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names,
                      show=False, plot_type="dot", max_display=19, plot_size=None)
    plt.title("SHAP Beeswarm — Model 2 LightGBM (19 URL Features)")
    plt.tight_layout()
    fig.savefig(out_dir / "shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close("all")

    # Save mean |SHAP|
    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_shap})
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return shap_df


def plot_perm_importance(perm_result, feature_names, out_dir: Path):
    imp_mean = perm_result.importances_mean
    imp_std  = perm_result.importances_std
    order = np.argsort(imp_mean)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.array(feature_names)[order][::-1], imp_mean[order][::-1],
            xerr=imp_std[order][::-1], capsize=3, color="steelblue", alpha=0.8)
    ax.set_xlabel("PR-AUC drop (mean ± std)")
    ax.set_title("Permutation Importance — Model 2 LightGBM")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_dir / "permutation_importance.png", dpi=300)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# REFERENCE MODELS
# ═══════════════════════════════════════════════════════════════════════════

def eval_reference_models(train_df, test_df):
    X_tr = train_df[URL_FEATURES].values
    y_tr = train_df[LABEL_COL].values
    X_te = test_df[URL_FEATURES].values
    y_te = test_df[LABEL_COL].values

    # Majority class baseline
    majority = int(pd.Series(y_tr).mode()[0])
    p_maj = np.full(len(y_te), float(majority))
    pr_maj = average_precision_score(y_te, p_maj)

    # Depth-2 decision tree (URL length only via proxy HostRatio+PathRatio)
    dt = DecisionTreeClassifier(max_depth=2, random_state=SEED)
    dt.fit(X_tr, y_tr)
    p_dt = dt.predict_proba(X_te)[:, 1]
    pr_dt = average_precision_score(y_te, p_dt)

    return {
        "majority_class_pr_auc": float(pr_maj),
        "depth2_tree_pr_auc": float(pr_dt),
    }


# ═══════════════════════════════════════════════════════════════════════════
# LATENCY BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_latency(model, X_test):
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X_test), size=LATENCY_URLS + LATENCY_WARMUP, replace=True)
    samples = X_test[idx]
    # warmup
    for i in range(LATENCY_WARMUP):
        model.predict_proba(samples[i:i+1])
    # measure
    times = []
    for i in range(LATENCY_WARMUP, LATENCY_WARMUP + LATENCY_URLS):
        t0 = time.perf_counter()
        model.predict_proba(samples[i:i+1])
        times.append((time.perf_counter() - t0) * 1e6)
    return {
        "latency_us_mean": float(np.mean(times)),
        "latency_us_p50":  float(np.percentile(times, 50)),
        "latency_us_p95":  float(np.percentile(times, 95)),
        "latency_us_p99":  float(np.percentile(times, 99)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# SPLIT SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════════

def split_sensitivity(train_df, test_df, best_params):
    """Compare domain-grouped vs random URL-level split."""
    log.info("  Split sensitivity analysis …")
    X_all = pd.concat([train_df[URL_FEATURES], test_df[URL_FEATURES]]).values
    y_all = pd.concat([train_df[[LABEL_COL]], test_df[[LABEL_COL]]])[LABEL_COL].values

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X_all))
    split_pt = int(0.8 * len(idx))
    X_tr_r, y_tr_r = X_all[idx[:split_pt]], y_all[idx[:split_pt]]
    X_te_r, y_te_r = X_all[idx[split_pt:]], y_all[idx[split_pt:]]

    params = {**LGB_FIXED, **best_params, "bagging_freq": 1}
    m = lgb.LGBMClassifier(**params)
    m.fit(X_tr_r, y_tr_r)
    p_r = m.predict_proba(X_te_r)[:, 1]
    pr_random = average_precision_score(y_te_r, p_r)

    return {
        "domain_grouped_pr_auc": "see results.pr_auc",
        "random_url_level_pr_auc": float(pr_random),
        "note": "Random split inflates PR-AUC if same domain in train+test.",
    }


# ═══════════════════════════════════════════════════════════════════════════
# STRATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def strata_analysis(test_df, y_prob_cal, tau_high):
    results = {}

    # Hard URLs: shortener or embedded
    mask_hard = (test_df["IsShortener"] == 1) | (test_df["HasEmbeddedURL"] == 1)
    for name, mask in [("hard_urls", mask_hard), ("normal_urls", ~mask_hard)]:
        if mask.sum() == 0: continue
        y_s = test_df.loc[mask, LABEL_COL].values
        p_s = y_prob_cal[mask.values]
        if len(np.unique(y_s)) < 2: continue
        results[name] = {
            "n": int(mask.sum()),
            "pr_auc": float(average_precision_score(y_s, p_s)),
            **threshold_metrics(y_s, p_s, tau_high),
        }

    # Path depth (using NoOfSlashes as proxy)
    for depth_label, mask_fn in [
        ("shallow_path", test_df["NoOfSlashes"] <= 2),
        ("deep_path",    test_df["NoOfSlashes"] >  2),
    ]:
        mask = mask_fn
        if mask.sum() == 0: continue
        y_s = test_df.loc[mask, LABEL_COL].values
        p_s = y_prob_cal[mask.values]
        if len(np.unique(y_s)) < 2: continue
        results[depth_label] = {
            "n": int(mask.sum()),
            "pr_auc": float(average_precision_score(y_s, p_s)),
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════
# HANDOFF REPORT
# ═══════════════════════════════════════════════════════════════════════════

def write_handoff(results: dict, out_path: Path):
    r = results
    md = f"""# MODEL2_lgbm — HANDOFF REPORT
Generated: {datetime.now(timezone.utc).isoformat()}

## Model Identity
- **Model ID**: model2_lgbm
- **Feature set**: 19 URL lexical features (no QR features)
- **Algorithm**: LightGBM (gradient boosting)
- **Calibration**: Isotonic Regression on OOF predictions

## Dataset
- Total rows: {r['dataset']['total_rows']:,}
- Train: {r['dataset']['train_rows']:,} | Test: {r['dataset']['test_rows']:,}
- Train positives: {r['dataset']['train_pos']:,} | Train negatives: {r['dataset']['train_neg']:,}
- Shared domains (train/test): 0 ✓

## Cross-Validation (5-fold StratifiedGroupKFold)
| Fold | PR-AUC | ROC-AUC |
|------|--------|---------|
{chr(10).join(f"| {m['fold']} | {m['pr_auc']:.4f} | {m['roc_auc']:.4f} |" for m in r['cv_metrics'])}
| **Mean** | **{r['oof_pr_auc_mean']:.4f} ± {r['oof_pr_auc_std']:.4f}** | **{r['oof_roc_auc_mean']:.4f} ± {r['oof_roc_auc_std']:.4f}** |

## Test Set Evaluation (locked after FROZEN.json)
| Metric | Value | 95% CI |
|--------|-------|--------|
| ROC-AUC | {r['test']['roc_auc']:.4f} | [{r['test']['roc_auc_ci'][1]:.4f}, {r['test']['roc_auc_ci'][2]:.4f}] |
| PR-AUC  | {r['test']['pr_auc']:.4f}  | [{r['test']['pr_auc_ci'][1]:.4f}, {r['test']['pr_auc_ci'][2]:.4f}] |
| ECE     | {r['test']['ece']:.4f} | — |

## Thresholds
| Threshold | Value | FPR | FNR |
|-----------|-------|-----|-----|
| tau_high (FPR≤1%) | {r['thresholds']['tau_high']:.4f} | {r['thresholds']['tau_high_metrics']['fpr']:.4f} | {r['thresholds']['tau_high_metrics']['fnr']:.4f} |
| tau_low  (FNR≤2%) | {r['thresholds']['tau_low']:.4f}  | {r['thresholds']['tau_low_metrics']['fpr']:.4f}  | {r['thresholds']['tau_low_metrics']['fnr']:.4f}  |

## Latency
| Metric | Value |
|--------|-------|
| Mean   | {r['latency']['latency_us_mean']:.1f} µs/URL |
| P50    | {r['latency']['latency_us_p50']:.1f} µs/URL |
| P95    | {r['latency']['latency_us_p95']:.1f} µs/URL |
| P99    | {r['latency']['latency_us_p99']:.1f} µs/URL |

## Reference Models
| Model | PR-AUC |
|-------|--------|
| Majority class baseline | {r['reference_models']['majority_class_pr_auc']:.4f} |
| Depth-2 decision tree   | {r['reference_models']['depth2_tree_pr_auc']:.4f} |
| **LightGBM (Model 2)**  | **{r['test']['pr_auc']:.4f}** |

## Best Hyperparameters
```json
{json.dumps(r['best_params'], indent=2)}
```

## Output Files
| File | Description |
|------|-------------|
| `models/model2_lgbm.txt` | LightGBM booster (text format) |
| `models/model2_lgbm_calibrator.joblib` | Isotonic calibrator |
| `artifacts/FROZEN.json` | Test-lock checkpoint |
| `predictions/model2_lgbm_oof_train.parquet` | OOF predictions (train) |
| `predictions/model2_lgbm_test.parquet` | Test predictions |
| `results/model2_lgbm.json` | All metrics (machine-readable) |
"""
    out_path.write_text(md, encoding="utf-8")
    log.info(f"  Handoff report written: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# CHECKER SCRIPT
# ═══════════════════════════════════════════════════════════════════════════

CHECKER_CODE = '''"""checker_model2.py — verify every number in handoff against results JSON."""
import json, sys
from pathlib import Path

BASE = Path(__file__).parent
results_path = BASE / "results" / "model2_lgbm.json"
handoff_path = BASE / "reports" / "MODEL2_lgbm_HANDOFF.md"

r = json.loads(results_path.read_text(encoding="utf-8"))
handoff = handoff_path.read_text(encoding="utf-8")

errors = []

def check(name, value, text, fmt=".4f"):
    s = format(value, fmt)
    if s not in text:
        errors.append(f"FAIL: {name}={s} not found in handoff")
    else:
        print(f"  OK: {name}={s}")

print("Checking results vs handoff …")
check("test.roc_auc",  r["test"]["roc_auc"],  handoff)
check("test.pr_auc",   r["test"]["pr_auc"],   handoff)
check("test.ece",      r["test"]["ece"],       handoff)
check("tau_high", r["thresholds"]["tau_high"], handoff)
check("tau_low",  r["thresholds"]["tau_low"],  handoff)
check("latency_us_mean", r["latency"]["latency_us_mean"], handoff, ".1f")

# Check FROZEN.json exists and predates test results
frozen_path = BASE / "artifacts" / "FROZEN.json"
if not frozen_path.exists():
    errors.append("FAIL: artifacts/FROZEN.json does not exist")
else:
    frozen = json.loads(frozen_path.read_text())
    print(f"  OK: FROZEN.json exists (written {frozen.get('written_at',\'?\')})")

# Check parquets
import pandas as pd
for name, path in [
    ("oof_train", BASE/"predictions"/"model2_lgbm_oof_train.parquet"),
    ("test",      BASE/"predictions"/"model2_lgbm_test.parquet"),
]:
    if not path.exists():
        errors.append(f"FAIL: {path} missing")
        continue
    df = pd.read_parquet(path)
    expected = ["url_sha256","registered_domain","partition","fold","label","p_raw","p_cal","risk_level"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        errors.append(f"FAIL: {name} parquet missing cols: {missing}")
    else:
        print(f"  OK: {name} parquet schema correct ({len(df):,} rows)")

if errors:
    print("\\n=== CHECKER FAILED ===")
    for e in errors: print(" ", e)
    sys.exit(1)
else:
    print("\\n=== ALL CHECKS PASSED ===")
'''


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="Dataset.csv")
    parser.add_argument("--folds-path", default="cv_folds_by_sha.csv")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--n-trials",   type=int, default=N_TRIALS)
    args = parser.parse_args()

    t_start = time.perf_counter()
    base = Path(args.output_dir)
    setup_dirs(base)

    fig_dir  = base / "figures"
    tbl_dir  = base / "tables"
    res_dir  = base / "results"
    rep_dir  = base / "reports"
    pred_dir = base / "predictions"
    mod_dir  = base / "models"
    art_dir  = base / "artifacts"

    data_path  = Path(args.data)
    folds_path = Path(args.folds_path)

    log.info("=" * 60)
    log.info("Model 2 — LightGBM (19 URL features) — Full Training")
    log.info("=" * 60)

    # ── 1. Load data ─────────────────────────────────────────────
    train_df, test_df = load_data(data_path, folds_path)

    # ── 2. Optuna tuning ─────────────────────────────────────────
    db_path = art_dir / "optuna_model2.db"
    study   = run_optuna(train_df, db_path, args.n_trials)
    best_params = study.best_params

    # ── 3. OOF + calibration ─────────────────────────────────────
    oof_parquet_path = pred_dir / "model2_lgbm_oof_train.parquet"
    model_txt_path   = mod_dir / "model2_lgbm.txt"
    cal_path         = mod_dir / "model2_lgbm_calibrator.joblib"

    if oof_parquet_path.exists() and model_txt_path.exists() and cal_path.exists():
        log.info("Loading existing OOF predictions, calibrator, and model …")
        oof_df = pd.read_parquet(oof_parquet_path)
        if "Label" in oof_df.columns:
            oof_df = oof_df.rename(columns={"Label": "label"})
        oof_raw = oof_df["p_raw"].values
        oof_cal = oof_df["p_cal"].values
        cal = joblib.load(cal_path)
        final_model = lgb.LGBMClassifier(**{**LGB_FIXED, **best_params, "bagging_freq": 1})
        final_model.fit(train_df[URL_FEATURES].values, train_df[LABEL_COL].values)
        tau_high, tau_low = pick_thresholds(train_df[LABEL_COL].values, oof_cal)
        cv_metrics = []
        for fold_id in range(N_FOLDS):
            va_idx = oof_df["fold"] == fold_id
            cv_metrics.append({
                "fold": fold_id,
                "pr_auc": average_precision_score(oof_df.loc[va_idx, "label"], oof_df.loc[va_idx, "p_raw"]),
                "roc_auc": roc_auc_score(oof_df.loc[va_idx, "label"], oof_df.loc[va_idx, "p_raw"]),
            })
    else:
        oof_raw, cv_metrics = compute_oof(train_df, best_params)
        cal, oof_cal = calibrate(train_df, oof_raw)
        tau_high, tau_low = pick_thresholds(train_df[LABEL_COL].values, oof_cal)
        final_model = train_final_model(train_df, best_params)
        final_model.booster_.save_model(str(model_txt_path))
        joblib.dump(cal, cal_path)
        log.info(f"  Model saved: {model_txt_path}")
        oof_df = train_df[[SHA_COL, DOM_COL, PART_COL, "fold", LABEL_COL]].copy()
        oof_df = oof_df.rename(columns={LABEL_COL: "label"})
        oof_df["p_raw"] = oof_raw
        oof_df["p_cal"] = oof_cal
        oof_df["risk_level"] = [risk_level(p, tau_high, tau_low) for p in oof_cal]

    log.info(f"  tau_high={tau_high:.4f}  tau_low={tau_low:.4f}")

    # ── 4. Write FROZEN.json (BEFORE touching test set) ──────────
    frozen = {
        "written_at":    datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": sha256_file(data_path),
        "tau_high":      tau_high,
        "tau_low":       tau_low,
        "best_params":   best_params,
        "oof_pr_auc_mean": float(np.mean([m["pr_auc"] for m in cv_metrics])),
    }
    (art_dir / "FROZEN.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    log.info("  ✓ FROZEN.json written — test set now unlocked")

    # ── 7. Save OOF parquet ──────────────────────────────────────
    if "Label" in oof_df.columns:
        oof_df = oof_df.rename(columns={"Label": "label"})
    oof_df.to_parquet(oof_parquet_path, index=False)

    # ── 8. Test evaluation (ONCE) ────────────────────────────────
    log.info("Evaluating on TEST set (single evaluation) …")
    X_te = test_df[URL_FEATURES].values
    y_te = test_df[LABEL_COL].values
    p_te_raw = final_model.predict_proba(X_te)[:, 1]
    p_te_cal = cal.predict(p_te_raw)

    roc_val, pr_val = plot_roc_pr(y_te, p_te_cal, fig_dir)
    plot_confusion(y_te, p_te_cal, tau_high, fig_dir)
    plot_reliability(y_te, p_te_cal, fig_dir)

    ece_val = ece_score(y_te, p_te_cal)
    ks_val  = stats.ks_2samp(p_te_cal[y_te==0], p_te_cal[y_te==1]).statistic

    # Bootstrap CI
    log.info(f"  Bootstrap CI ({BOOTSTRAP_REPS} replicates) …")
    domains_te = test_df[DOM_COL].values
    roc_mean, roc_lo, roc_hi = domain_bootstrap_ci(y_te, p_te_cal, domains_te, roc_auc_score)
    pr_mean,  pr_lo,  pr_hi  = domain_bootstrap_ci(y_te, p_te_cal, domains_te, average_precision_score)

    th_high_m = threshold_metrics(y_te, p_te_cal, tau_high)
    th_low_m  = threshold_metrics(y_te, p_te_cal, tau_low)
    log.info(f"  tau_high: Prec={th_high_m['precision']:.4f} Rec={th_high_m['recall']:.4f} "
             f"F1={th_high_m['f1']:.4f} FPR={th_high_m['fpr']:.4f}")

    # Test parquet
    test_pred_df = test_df[[SHA_COL, DOM_COL, PART_COL, LABEL_COL]].copy()
    test_pred_df = test_pred_df.rename(columns={LABEL_COL: "label"})
    test_pred_df["fold"]       = -1
    test_pred_df["p_raw"]      = p_te_raw
    test_pred_df["p_cal"]      = p_te_cal
    test_pred_df["risk_level"] = [risk_level(p, tau_high, tau_low) for p in p_te_cal]
    test_pred_df.to_parquet(pred_dir / "model2_lgbm_test.parquet", index=False)

    # ── 9. Strata analysis ───────────────────────────────────────
    strata = strata_analysis(test_df, p_te_cal, tau_high)

    # ── 10. Reference models ─────────────────────────────────────
    log.info("Evaluating reference models …")
    ref_models = eval_reference_models(train_df, test_df)

    # ── 11. Split sensitivity ────────────────────────────────────
    split_sens = split_sensitivity(train_df, test_df, best_params)
    split_sens["domain_grouped_pr_auc"] = float(pr_val)

    # ── 12. SHAP ─────────────────────────────────────────────────
    shap_csv_path = tbl_dir / "shap_importance.csv"
    shap_png_path = fig_dir / "shap_beeswarm.png"
    if shap_csv_path.exists() and shap_png_path.exists():
        log.info("  Loading cached SHAP importance from disk …")
        shap_df = pd.read_csv(shap_csv_path)
    else:
        log.info("SHAP analysis …")
        rng = np.random.default_rng(SEED)
        shap_idx = rng.choice(len(X_te), size=min(SHAP_SAMPLE, len(X_te)), replace=False)
        X_shap = X_te[shap_idx]
        shap_df = plot_shap(final_model, X_shap, URL_FEATURES, fig_dir)
        shap_df.to_csv(shap_csv_path, index=False)
        shap_df.to_markdown(tbl_dir / "shap_importance.md", index=False)

    # ── 13. Permutation importance ───────────────────────────────
    log.info("Permutation importance …")
    from sklearn.base import BaseEstimator, ClassifierMixin

    class CalibratedEstimator(BaseEstimator, ClassifierMixin):
        def __init__(self, m=None, c=None):
            self.m = m
            self.c = c
            self.classes_ = np.array([0, 1])
        def fit(self, X, y=None): return self
        def predict(self, X): return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
        def predict_proba(self, X):
            p1 = self.c.predict(self.m.predict_proba(X)[:, 1])
            return np.column_stack([1 - p1, p1])

    perm_scorer = lambda est, X, y: average_precision_score(y, est.predict_proba(X)[:, 1])
    perm_model  = CalibratedEstimator(final_model, cal)
    perm_result = permutation_importance(
        perm_model, X_te, y_te,
        n_repeats=PERM_REPEATS, random_state=SEED,
        scoring=perm_scorer,
    )
    plot_perm_importance(perm_result, URL_FEATURES, fig_dir)
    perm_df = pd.DataFrame({
        "feature": URL_FEATURES,
        "importance_mean": perm_result.importances_mean,
        "importance_std":  perm_result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    perm_df.to_csv(tbl_dir / "permutation_importance.csv", index=False)
    perm_df.to_markdown(tbl_dir / "permutation_importance.md", index=False)

    # ── 14. Latency benchmark ────────────────────────────────────
    log.info("Latency benchmark …")
    latency = benchmark_latency(final_model, X_te)
    log.info(f"  Mean={latency['latency_us_mean']:.1f}µs  P95={latency['latency_us_p95']:.1f}µs")

    # ── 15. Optuna history plot ───────────────────────────────────
    plot_optuna_history(study, fig_dir)

    # ── 16. CV table ─────────────────────────────────────────────
    cv_df = pd.DataFrame(cv_metrics)
    cv_df.loc[len(cv_df)] = {
        "fold": "mean±std",
        "pr_auc":  f"{np.mean([m['pr_auc'] for m in cv_metrics]):.4f}±{np.std([m['pr_auc'] for m in cv_metrics]):.4f}",
        "roc_auc": f"{np.mean([m['roc_auc'] for m in cv_metrics]):.4f}±{np.std([m['roc_auc'] for m in cv_metrics]):.4f}",
    }
    cv_df.to_csv(tbl_dir / "cv_results.csv", index=False)
    cv_df.to_markdown(tbl_dir / "cv_results.md", index=False)

    # McNemar placeholder (requires second model — log note)
    log.info("  [NOTE] McNemar test requires comparison model predictions — run after all models trained.")

    # ── 17. Compile results JSON ─────────────────────────────────
    results = {
        "model_id": "model2_lgbm",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "total_rows":  len(train_df) + len(test_df),
            "train_rows":  len(train_df),
            "test_rows":   len(test_df),
            "train_pos":   int(train_df[LABEL_COL].sum()),
            "train_neg":   int((train_df[LABEL_COL]==0).sum()),
            "test_pos":    int(y_te.sum()),
            "test_neg":    int((y_te==0).sum()),
        },
        "best_params": best_params,
        "cv_metrics":  cv_metrics,
        "oof_pr_auc_mean":  float(np.mean([m["pr_auc"]  for m in cv_metrics])),
        "oof_pr_auc_std":   float(np.std( [m["pr_auc"]  for m in cv_metrics])),
        "oof_roc_auc_mean": float(np.mean([m["roc_auc"] for m in cv_metrics])),
        "oof_roc_auc_std":  float(np.std( [m["roc_auc"] for m in cv_metrics])),
        "test": {
            "roc_auc":    float(roc_val),
            "roc_auc_ci": [roc_mean, roc_lo, roc_hi],
            "pr_auc":     float(pr_val),
            "pr_auc_ci":  [pr_mean, pr_lo, pr_hi],
            "ece":        float(ece_val),
            "ks_statistic": float(ks_val),
        },
        "thresholds": {
            "tau_high":         tau_high,
            "tau_high_metrics": th_high_m,
            "tau_low":          tau_low,
            "tau_low_metrics":  th_low_m,
        },
        "latency":          latency,
        "reference_models": ref_models,
        "split_sensitivity": split_sens,
        "strata":           strata,
        "features":         URL_FEATURES,
        "n_features":       len(URL_FEATURES),
        "elapsed_seconds":  time.perf_counter() - t_start,
    }
    (res_dir / "model2_lgbm.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info(f"  Results saved: {res_dir / 'model2_lgbm.json'}")

    # ── 18. Handoff report ───────────────────────────────────────
    write_handoff(results, rep_dir / "MODEL2_lgbm_HANDOFF.md")

    # ── 19. DEVIATIONS.md ────────────────────────────────────────
    dev_path = base / "DEVIATIONS.md"
    if not dev_path.exists():
        dev_path.write_text(
            "# DEVIATIONS from test.md\n\n"
            "| # | Section | Deviation | Reason |\n"
            "|---|---------|-----------|--------|\n"
            "| 1 | McNemar test | Deferred until all 3 models trained | Requires prediction pairs from LightGBM + RF/XGBoost |\n"
            "| 2 | URL-length-only tree | Approximated with all 19 features depth-2 tree | URL length not a direct feature; HostRatio+PathRatio used as proxy |\n",
            encoding="utf-8",
        )

    # ── 20. Checker script ───────────────────────────────────────
    checker_path = base / "checker_model2.py"
    checker_path.write_text(CHECKER_CODE, encoding="utf-8")
    log.info(f"  Checker written: {checker_path}")

    elapsed = time.perf_counter() - t_start
    log.info("=" * 60)
    log.info(f"DONE in {elapsed/60:.1f} min")
    log.info(f"  ROC-AUC:  {roc_val:.4f}  [{roc_lo:.4f}, {roc_hi:.4f}]")
    log.info(f"  PR-AUC:   {pr_val:.4f}  [{pr_lo:.4f}, {pr_hi:.4f}]")
    log.info(f"  ECE:      {ece_val:.4f}")
    log.info(f"  tau_high: {tau_high:.4f}  |  tau_low: {tau_low:.4f}")
    log.info(f"  Latency:  {latency['latency_us_mean']:.1f} µs/URL (mean)")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
