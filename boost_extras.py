"""Builds the missing report tables R1, R2, R6-R13 and figures R1-R7 for an already trained model (lgb | xgb | rf)
from artifacts/<m>_thesis (saved model, calibrator, frozen thresholds, predictions.npz). Nothing is re-tuned; the test set is only scored."""
import sys, json, time, platform, warnings
from pathlib import Path
from urllib.parse import urlsplit
import joblib, numpy as np, pandas as pd, optuna, sklearn, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, shap
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.tree import DecisionTreeClassifier
from train_rf_thesis import FEATURES, SEED, Curve, threshold_metrics, reliability, risk_levels
from train_boost_thesis import build
from url_ml.thesis_features import extract
warnings.filterwarnings("ignore"); optuna.logging.set_verbosity(optuna.logging.WARNING)
K = sys.argv[1]; TH = int(sys.argv[2]); SHAPN = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
NAME = {"lgb": "LightGBM", "xgb": "XGBoost", "rf": "Random Forest"}[K]
out = Path(f"artifacts/{K}_thesis"); rep = out / "report"; T0 = time.time()
log = lambda m: print(f"[{(time.time()-T0)/60:5.1f} min] {m}", flush=True)
plt.rcParams.update({"font.family": "serif", "figure.dpi": 150, "axes.grid": True, "grid.alpha": .3})
art = joblib.load(out / "url_phishing_model.joblib"); model, iso = art["model"], art["calibrator"]
S = json.load(open(rep / "summary.json")); P = np.load(out / "predictions.npz")
tau_high, tau_low = S["tau_high"], S["tau_low"]; bp = S["best_params"]
df = pd.read_csv("Dataset_Hybrid.csv"); tr = df[df.partition == "train"].reset_index(drop=True); te = df[df.partition == "test"].reset_index(drop=True)
Xtr, ytr = tr[FEATURES].astype("float32").values, tr.Label.values.astype(int); Xte, yte = te[FEATURES].astype("float32").values, te.Label.values.astype(int)
raw, cal, lvl = P["test_raw"], P["test_cal"], P["level"]; pred = (cal >= tau_high).astype(int)
assert np.array_equal(yte, P["y_test"])
hard = ((te.IsShortener == 1) | (te.HasEmbeddedURL == 1)).astype(int).values
point = Curve(raw, yte).stats(np.ones(len(yte))); point.update(threshold_metrics(yte, pred, cal, np.ones(len(yte))))
def save(n, d): d.to_csv(rep / f"{n}.csv", index=False); return d
# R1
save("R1_partitions", pd.DataFrame({"Partition": ["Training", "Test (locked)", "Total"], "URLs": [len(tr), len(te), len(df)],
    "Malicious URLs": [ytr.sum(), yte.sum(), ytr.sum() + yte.sum()], "Malicious share": [ytr.mean(), yte.mean(), (ytr.sum() + yte.sum()) / len(df)],
    "Registered domains": [tr.registered_domain.nunique(), te.registered_domain.nunique(), df.registered_domain.nunique()]}))
# R2 params
save("R2_params", pd.DataFrame({"Hyperparameter": list(bp), "Selected value": [str(v) for v in bp.values()]}))
# R6
tpr_h, fpr_h = point["Recall (TPR)"], point["FPR"]; pos_n, neg_n = yte.sum(), (yte == 0).sum(); cv = Curve(raw, yte); rows = []
for b in [.4, .1, .05, .01]:
    w = np.where(yte == 1, 1.0, (1 - b) / b * pos_n / neg_n)
    rows.append({"Phishing base rate": f"{b:.0%}", "Precision at tau_high": tpr_h * b / (tpr_h * b + fpr_h * (1 - b)), "PR-AUC": cv.stats(w)["PR-AUC"], "False alarms / 10,000 benign scans": fpr_h * 1e4})
r6 = save("R6_base_rates", pd.DataFrame(rows))
# R7/R8
auc = {f: roc_auc_score(ytr, Xtr[:, i]) for i, f in enumerate(FEATURES)}
r8 = pd.DataFrame({"Feature": list(auc), "AUC": list(auc.values())}); r8["max(AUC,1-AUC)"] = np.maximum(r8.AUC, 1 - r8.AUC); r8["Flag (>0.95)"] = np.where(r8["max(AUC,1-AUC)"] > .95, "Yes", "No")
save("R8_single_feature_auc", r8.sort_values("max(AUC,1-AUC)", ascending=False))
def lf(urls):
    p = [urlsplit(u) for u in urls]; return np.c_[[len(u) for u in urls], [len(x.path) for x in p], [u.count("/") for u in urls], [1 if x.query else 0 for x in p]]
dt = DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=SEED).fit(lf(tr.url.values), ytr); ref = dt.predict_proba(lf(te.url.values))[:, 1]
rs_ = Curve(ref, yte).stats(np.ones(len(yte))); rf1 = threshold_metrics(yte, (ref >= .5).astype(int), ref, np.ones(len(yte)))["F1-score"]
save("R7_reference_models", pd.DataFrame([{"Model": "Majority class (always benign)", "ROC-AUC": .5, "PR-AUC": yte.mean(), "TPR@1%FPR": 0., "F1-score": 0.},
    {"Model": "URL-length-only tree (depth 2)", "ROC-AUC": rs_["ROC-AUC"], "PR-AUC": rs_["PR-AUC"], "TPR@1%FPR": rs_["TPR@1%FPR"], "F1-score": rf1},
    {"Model": f"{NAME} (19 static features)", "ROC-AUC": point["ROC-AUC"], "PR-AUC": point["PR-AUC"], "TPR@1%FPR": point["TPR@1%FPR"], "F1-score": point["F1-score"]}]))
# R9 split sensitivity: same config on random URL-level split
rsd = np.random.default_rng(SEED).permutation(len(df)); nte = len(te); rte, rtr = rsd[:nte], rsd[nte:]
aX = df[FEATURES].astype("float32").values; ay = df.Label.values.astype(int); ad = df.registered_domain.values
seen = float(np.mean(np.isin(ad[rte], np.unique(ad[rtr]))))
pw = float((ytr == 0).sum() / (ytr == 1).sum()); m2 = build(K, bp, TH, pw).fit(aX[rtr], ay[rtr]); rst = Curve(m2.predict_proba(aX[rte])[:, 1], ay[rte]).stats(np.ones(nte)); del m2
save("R9_split_sensitivity", pd.DataFrame([{"Split": "Grouped by registered domain (thesis)", "Test URLs on training domains": 0.0, **{k: point[k] for k in ["ROC-AUC", "PR-AUC", "TPR@1%FPR", "TPR@0.1%FPR"]}},
    {"Split": "Random URL-level split", "Test URLs on training domains": seen, **rst}])); log("R9 done")
# R10
depth = np.array([len([s for s in urlsplit(u).path.split("/") if s]) for u in te.url.values])
strata = {"Hard URLs": hard == 1, "Other URLs": hard == 0, "Path depth 0-1": depth <= 1, "Path depth 2-3": (depth >= 2) & (depth <= 3), "Path depth >=4": depth >= 4}; rows = []
for nm, m in strata.items():
    ym = yte[m]; tm = threshold_metrics(ym, pred[m], cal[m], np.ones(m.sum()))
    rows.append({"Stratum": nm, "URLs": int(m.sum()), "Malicious share": ym.mean(), "Recall": tm["Recall (TPR)"], "FPR": tm["FPR"], "FNR": tm["FNR"],
                 "PR-AUC": Curve(raw[m], ym).stats(np.ones(m.sum()))["PR-AUC"] if 0 < ym.sum() < len(ym) else np.nan, "Safe share": float(np.mean(lvl[m] == 0))})
save("R10_strata", pd.DataFrame(rows))
# R11
rng = np.random.default_rng(SEED); base = average_precision_score(yte, raw); pi = []
for i in range(len(FEATURES)):
    d = []
    for _ in range(3): Xp = Xte.copy(); Xp[:, i] = rng.permutation(Xp[:, i]); d.append(base - average_precision_score(yte, model.predict_proba(Xp)[:, 1]))
    pi.append((np.mean(d), np.std(d)))
log("perm done"); sidx = np.random.default_rng(SEED).choice(len(yte), SHAPN, replace=False)
sv = shap.TreeExplainer(model).shap_values(Xte[sidx], check_additivity=False); sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv); log("shap done")
fi = np.asarray(model.feature_importances_, float); fi = fi / fi.sum()
r11 = save("R11_feature_importance", pd.DataFrame({"Feature": FEATURES, "Permutation dPR-AUC": [p[0] for p in pi], "Permutation SD": [p[1] for p in pi],
    "Importance share (native)": fi, "Mean |SHAP| (SHAPUNIT)": np.abs(sv).mean(0)}).sort_values("Permutation dPR-AUC", ascending=False))
# R12 latency (incl. feature extraction)
urls = te.url.values[:1050]; fe = []
for u in urls[50:]: t0 = time.perf_counter(); extract(u); fe.append((time.perf_counter() - t0) * 1e3)
try: model.set_params(n_jobs=1)
except Exception: pass
li, lc = [], []
for i in range(1050):
    x = Xte[i:i + 1]; t0 = time.perf_counter(); p = model.predict_proba(x)[:, 1]; t1 = time.perf_counter(); c = iso.predict(p); risk_levels(c, np.zeros(1), tau_high, tau_low); t2 = time.perf_counter()
    if i >= 50: li.append((t1 - t0) * 1e3); lc.append((t2 - t1) * 1e3)
q = lambda v: [np.median(v), np.percentile(v, 95), np.percentile(v, 99)]; tot = np.array(fe) + np.array(li) + np.array(lc)
save("R12_latency", pd.DataFrame([["Feature extraction (re-implemented extractor)", *q(fe)], [f"{NAME} inference", *q(li)], ["Calibration + risk level", *q(lc)], ["Total", *q(tot)]], columns=["Stage", "Median ms", "p95 ms", "p99 ms"]))
try: model.set_params(n_jobs=TH)
except Exception: pass
import lightgbm, xgboost
save("R13_environment", pd.DataFrame({"Component": ["Python", "scikit-learn", "LightGBM", "XGBoost", "Optuna", "SHAP", "pandas / NumPy", "Platform"],
    "Version": [platform.python_version(), sklearn.__version__, lightgbm.__version__, xgboost.__version__, optuna.__version__, shap.__version__, f"{pd.__version__} / {np.__version__}", platform.platform()]}))
ex = {"split_sensitivity_seen_domains": seen, "n_hard_test": int(hard.sum()), "shap_n": SHAPN, "extraction_median_ms": float(np.median(fe))}
json.dump(ex, open(rep / "extras.json", "w"))
# ---------- figures
col = "#1f77b4"; st = optuna.load_study(study_name=f"{K}_thesis", storage=f"sqlite:///{out.as_posix()}/optuna_study.db"); vals = [t.value for t in st.trials if t.value is not None]
fig, ax = plt.subplots(figsize=(6.5, 3.6)); ax.scatter(range(len(vals)), vals, s=14, label="Completed trial"); ax.plot(np.maximum.accumulate(vals), c="tab:red", label="Best so far")
ax.set(xlabel="Trial", ylabel="Mean CV PR-AUC", title=f"Optuna TPE optimisation history ({NAME})", ylim=(max(min(vals), .93), max(vals) + .002)); ax.legend(); fig.tight_layout(); fig.savefig(rep / "figR1_optuna_history.png"); plt.close()
fig, ax = plt.subplots(figsize=(5, 4.6))
for s, nm, e in [(raw, f"Raw {NAME}", S["ece_raw"]), (cal, "Isotonic-calibrated", S["ece_cal"])]:
    x, y_ = reliability(yte, s); ax.plot(x, y_, "o-", ms=3, label=f"{nm} (ECE {e:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set(xlabel="Mean predicted probability", ylabel="Observed malicious share", title="Reliability diagram, test set"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(rep / "figR2_reliability.png"); plt.close()
def rp(s):
    o = np.argsort(-s, kind="stable"); ss = s[o]; yy = yte[o]; l = np.r_[ss[1:] != ss[:-1], True]; tp_ = np.cumsum(yy)[l]; fp_ = np.cumsum(1 - yy)[l]
    return np.r_[0, fp_ / fp_[-1]], np.r_[0, tp_ / tp_[-1]], tp_ / (tp_ + fp_), tp_ / tp_[-1]
f1_, t1_, p1_, r1_ = rp(raw); f2_, t2_, p2_, r2_ = rp(ref); fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
axs[0].plot(f1_, t1_, lw=2, label=f"{NAME} (AUC {point['ROC-AUC']:.4f})"); axs[0].plot(f2_, t2_, "--", c="gray", label="URL-length-only tree"); axs[0].scatter([fpr_h], [tpr_h], c="k", zorder=5, label=r"$\tau_{high}$ (frozen)")
for x in (1e-3, 1e-2): axs[0].axvline(x, c="red", ls=":", lw=1)
axs[0].set(xscale="log", xlim=(1e-4, 1), ylim=(0, 1), xlabel="False positive rate (log scale)", ylabel="True positive rate", title="(a) ROC curve, locked test set"); axs[0].legend(loc="lower right", fontsize=8)
axs[1].plot(r1_, p1_, lw=2, label=f"{NAME} (AP {point['PR-AUC']:.4f})"); axs[1].plot(r2_, p2_, "--", c="gray", label="URL-length-only tree"); axs[1].scatter([tpr_h], [point["Precision"]], c="k", zorder=5)
axs[1].axhline(yte.mean(), c="k", ls=":", lw=1, label=f"Prevalence {yte.mean():.2f}"); axs[1].set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision", title="(b) Precision-recall curve"); axs[1].legend(loc="lower left", fontsize=8); fig.tight_layout(); fig.savefig(rep / "figR3_roc_pr.png"); plt.close()
names = ["Safe", "Suspicious", "Malicious"]; cols = ["#4c9a2a", "#e3a21a", "#c0392b"]; fig, ax = plt.subplots(figsize=(7, 3.6))
for yi, (c, nm) in enumerate([(1, "Malicious URL"), (0, "Benign URL")]):
    m = yte == c; left = 0
    for k in range(3):
        sh = np.mean(lvl[m] == k); ax.barh(yi, sh, left=left, color=cols[k], label=names[k] if yi == 0 else None)
        if sh > .05: ax.text(left + sh / 2, yi, f"{sh:.1%}", ha="center", va="center", color="w", fontsize=8)
        left += sh
ax.set(yticks=[0, 1], yticklabels=["Malicious URL", "Benign URL"], xlim=(0, 1), xlabel="Share of test URLs", title="Three-level risk output on the test set"); ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, -.2)); fig.tight_layout(); fig.savefig(rep / "figR4_risk_levels.png"); plt.close()
bb = np.logspace(-3, np.log10(.5), 200); pp = tpr_h * bb / (tpr_h * bb + fpr_h * (1 - bb)); fig, ax = plt.subplots(figsize=(5.6, 3.6)); ax.plot(bb, pp, lw=2)
for b in [.4, .1, .05, .01]: v = tpr_h * b / (tpr_h * b + fpr_h * (1 - b)); ax.scatter([b], [v], c="k", zorder=5); ax.annotate(f"{v:.3f}", (b, v), textcoords="offset points", xytext=(6, -12), fontsize=8)
ax.set(xscale="log", ylim=(0, 1.02), xlabel="Phishing base rate (log scale)", ylabel=r"Precision at $\tau_{high}$", title="Precision under different base rates"); fig.tight_layout(); fig.savefig(rep / "figR5_base_rate.png"); plt.close()
s = r11.sort_values("Permutation dPR-AUC"); fig, ax = plt.subplots(figsize=(7, 5.5)); ax.barh(s.Feature, s["Permutation dPR-AUC"], xerr=s["Permutation SD"]); ax.set(xlabel="Drop in PR-AUC when the feature is permuted", title="Permutation importance on held-out test URLs"); fig.tight_layout(); fig.savefig(rep / "figR6_permutation.png"); plt.close()
shap.summary_plot(sv, Xte[sidx], feature_names=FEATURES, show=False, max_display=19); plt.title(f"SHAP summary on {SHAPN:,} test URLs", fontsize=10); plt.tight_layout(); plt.savefig(rep / "figR7_shap.png"); plt.close()
log("DONE")
