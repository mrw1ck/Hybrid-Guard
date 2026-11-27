import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Pt, Inches
E7 = pd.read_csv("artifacts/e7_features25/E7_features25.csv"); ST = pd.read_csv("artifacts/e7_features25/E7_new_feature_stats.csv"); W1 = pd.read_csv("artifacts/webapp_update/W1_score_reproduction.csv")
# figure
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for a, m in zip(ax, ["PR-AUC", "TPR@1%FPR"]):
    d = E7[E7.Metric == m]; sets = ["19 features (stored)", "24 features (25 without HasHTTPS)", "25 features (Table 3.5)"]
    for i, s in enumerate(sets):
        for j, mod in enumerate(["LightGBM", "XGBoost", "Random Forest"]):
            r = d[(d.Model == mod) & (d["Feature set"] == s)].iloc[0]
            a.errorbar(i + (j - 1) * .2, r.Value, yerr=[[r.Value - r["CI lower"]], [r["CI upper"] - r.Value]], fmt="o", color=f"C{j}", capsize=3, label=mod if i == 0 else None)
    a.set_xticks(range(3)); a.set_xticklabels(["19", "24 (no HasHTTPS)", "25"]); a.set_title(m); a.set_xlabel("feature set"); a.grid(alpha=.3)
ax[0].legend(); plt.tight_layout(); plt.savefig("artifacts/e7_features25/figE7_featuresets.png", dpi=150); plt.close()
D = Document(); D.styles["Normal"].font.name = "Calibri"; D.styles["Normal"].font.size = Pt(10.5)
def tbl(df, cap):
    D.add_paragraph(cap).runs[0].bold = True
    t = D.add_table(rows=1, cols=len(df.columns)); t.style = "Light Grid Accent 1"
    for i, c in enumerate(df.columns): t.rows[0].cells[i].text = str(c)
    for _, r in df.iterrows():
        cs = t.add_row().cells
        for i, v in enumerate(r): cs[i].text = f"{v:.4f}" if isinstance(v, (float, np.floating)) and not np.isnan(v) else ("" if isinstance(v, float) else str(v))
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for r in p.runs: r.font.size = Pt(8.5)
    D.add_paragraph()
P = D.add_paragraph; H = D.add_heading
H("Hybrid-Guard: items 8–10 — REFORMS, webapp, Table 3.5", 0)
H("A. Table 3.5 versus the 19 implemented features (item 10)", 1)
P("Table 3.5 lists 25 features; the dataset and every trained model use 19. The two sets share 16 features. Nine Table-3.5 features are not model inputs and three implemented features are not in Table 3.5.")
tbl(pd.DataFrame([
 ["URLLength, HostnameLength, PathLength, QueryLength", "absolute lengths", "HostRatio, PathRatio", "lengths divided by URL length (scale-free)"],
 ["SubdomainDepth", "count of subdomain labels", "SubdomainRatio", "subdomain length / host length"],
 ["IsIPHost", "host is an IP", "—", "not a model input (IP hosts: handled by risk_policy `is_ip`)"],
 ["IsSuspiciousTLD", "TLD in risky list", "—", "not a model input (webapp rule, `RISKY_TLDS`)"],
 ["HasHTTPS", "scheme is https", "—", "excluded on purpose, see E7"],
 ["NoOfSuspiciousKeywords", "keyword hits", "—", "not a model input (webapp rule, `SUSPICIOUS_KEYWORDS`)"]],
 columns=["Table 3.5 feature(s)", "Meaning", "Implemented counterpart", "Note"]), "Table T1. Mapping from Table 3.5 to the implemented feature set")
P("Decision: Table 3.5 is corrected to the 19 implemented features (Section 2.2 already lists them) rather than retraining on 25. Reason (experiment E7): the 25-feature set was reconstructed from the report text and the three tuned models were refitted on it with the SAME hyper-parameters and evaluated once on the locked test set.")
tbl(ST.round(4), "Table E7-1. Statistics of the nine extra features (train split for AUC; all rows for means)")
P(f"HasHTTPS is a dataset-construction artefact: {ST.loc[ST.Feature=='HasHTTPS','Scheme added by dataset builder (share)'].iloc[0]:.1%} of the URLs had no scheme in the source file and the builder prefixed 'http://' (column `scheme_added`). Mean HasHTTPS is 3.1% for benign and 22.2% for malicious URLs, so the feature mostly encodes which source list a URL came from, not whether the site uses TLS. Using it would reward a collection artefact.")
t = E7[E7.Metric.isin(["PR-AUC", "TPR@1%FPR"])][["Model", "Feature set", "Metric", "Value", "CI lower", "CI upper", "Delta vs 19", "Delta CI lower", "Delta CI upper"]]
tbl(t.round(4), "Table E7-2. Test performance by feature set (95% CI: domain-cluster bootstrap, 1000 resamples; delta CI is paired)")
D.add_picture("artifacts/e7_features25/figE7_featuresets.png", width=Inches(6.4)); P("Figure E7. Test PR-AUC and TPR@1%FPR by feature set.")
g = lambda m, s, k: E7[(E7.Model == m) & (E7["Feature set"] == s) & (E7.Metric == k)].iloc[0]
l25 = g("LightGBM", "25 features (Table 3.5)", "PR-AUC"); l24 = g("LightGBM", "24 features (25 without HasHTTPS)", "PR-AUC"); t25 = g("LightGBM", "25 features (Table 3.5)", "TPR@1%FPR"); t24 = g("LightGBM", "24 features (25 without HasHTTPS)", "TPR@1%FPR")
P(f"Interpretation. With all 25 features LightGBM PR-AUC rises by {l25['Delta vs 19']:+.4f} (paired CI {l25['Delta CI lower']:+.4f} to {l25['Delta CI upper']:+.4f}) and TPR@1%FPR by {t25['Delta vs 19']*100:+.1f} pp; XGBoost and Random Forest behave the same. Removing only HasHTTPS cuts the PR-AUC gain to {l24['Delta vs 19']:+.4f} and the TPR gain to {t24['Delta vs 19']*100:+.1f} pp, so about two thirds of the PR-AUC gain and about half of the TPR gain come from the artefact feature. The remaining gain (24 features) is small but its paired CI excludes 0 for LightGBM and XGBoost (Random Forest PR-AUC: CI touches 0). Absolute URL/path lengths and IsIPHost add a little real signal.")
P("Caveats: (1) hyper-parameters were tuned on 19 features and not re-tuned, which can only understate the 25-feature models; (2) the nine extra features are reconstructed from the report text, not from the original code; (3) the 24-feature gain is evidence that the ratio features lose some information, which is a candidate future improvement, not a finding fixed by this work. The reported models stay on 19 features.")
H("B. Webapp now serves the calibrated thesis models (item 9)", 1)
P("Before: `webapp/app/config.py` pointed to `artifacts/lgb_final` (which no longer exists), a 22-feature, uncalibrated model trained on a random URL-level split with fixed thresholds 0.35/0.50. Now:")
for s in ["config.py: MODEL_PATHS = artifacts/{lgb,xgb,rf}_thesis/url_phishing_model.joblib; default mode lgb (best CV PR-AUC); legacy paths only used if no thesis artifact exists. Old webapp copied to webapp_backup_before_thesis_models/.",
          "New services/thesis_model_service.py: extracts the 19 features with url_ml/thesis_features.py, applies the isotonic calibrator, and uses the frozen tau_high / tau_low stored in the artifact (Suspicious band starts just above tau_low because the isotonic score is a step function).",
          "analyzer_service.py: thresholds come from the model; the 'uncertain' band for external checks is the Suspicious band; shortener / embedded-URL URLs can never be shown as Safe (Section 3.5.3 rule).",
          "url_ml/thesis_features.py: registered domain now uses tldextract (public-suffix list incl. private suffixes) and keeps IPs whole; the earlier heuristic mis-split IP hosts and *.workers.dev. SubdomainRatio mismatch against Dataset_Hybrid.csv went from 14.4% of rows (first 4,000) to 0.00% on 20,000 random rows; no feature differs on more than 0.34% of rows."]:
    D.add_paragraph(s, style="List Bullet")
tbl(W1.round(4), "Table W1. Webapp score reproduction on 1,500 random test URLs (webapp extractor + calibrator vs. stored test_cal)")
P("Interpretation: >99% of scores are identical to the stored calibrated scores and 99.9% of three-level decisions agree. Remaining differences come from URLs where the re-implemented extractor differs from the dataset on HasBrandInSubdomainOrPath / HasEmbeddedURL / SpecialCharRatio (≤0.34% of rows, see validate_extractor). Because isotonic scores are steps, one flipped feature can move a score a lot (max |Δ| ≈ 0.3–0.4) even though it is rare. A live FastAPI TestClient call to /api/analyze with mode=lgb returned the thesis thresholds (0.8466 / 0.0600) and sensible verdicts (bit.ly and an IP/port Mozi URL: malicious; www.google.com: base 0.61 lowered to benign by the existing trust guardrail).")
P("Not verified: the browser UI was not opened; the E4 hybrid-vs-ML-only experiment showed the external-check fusion cannot be shown to improve separation, so the guardrails remain heuristic.")
H("C. REFORMS checklist (item 8)", 1)
P("REFORMS (Kapoor et al., Science Advances 2024) has eight modules. The wording below is a condensed paraphrase written from memory of the checklist, not the official form; the status reflects what this project actually did. Y = done, P = partly, N = not done / not applicable.")
R = [
("1 Study goals","Claim stated and tied to evidence (H1: gradient boosting > RF)","P","H1 supported weakly in CV (corrected t-test p=0.007), not on the test set (bootstrap CIs contain 0). Thesis definition of H1 not in the files provided."),
("1 Study goals","Scope: prediction of phishing from static URL only","Y","Section 2.1; webapp adds external checks, evaluated separately in E4."),
("2 Computational reproducibility","Code and data available; environment listed","P","Scripts and Dataset_Hybrid.csv in project; fixed seed 42; library versions listed in report; no lockfile or container."),
("2 Computational reproducibility","Compute requirements stated","Y","Training time per model, 20 cores, 7 GB RAM."),
("3 Data quality","Population and sampling described; labels' source","P","Dataset provenance described in thesis; label noise not audited."),
("3 Data quality","Sample size adequate; class balance reported","Y","650,770 URLs, train 520,616 / test 130,154."),
("3 Data quality","Distribution shift vs deployment considered","P","E4 fresh set (OpenPhish + Tranco) showed ML-only FPR 92% on bare-domain URLs: strong shift, reported not solved."),
("4 Data preprocessing","Missing data, outliers, transformations","Y","Extractor validated (validate_extractor.py); URLs without scheme get 'http://' prefix (flagged as artefact, E7)."),
("4 Data preprocessing","Features documented","Y","Table 3.5 corrected, mapping T1."),
("5 Modelling","Multiple model families compared with same protocol","Y","LGB, XGB, RF; same folds, seed, 19 features, 50 Optuna trials."),
("5 Modelling","Hyper-parameter tuning only on training data","Y","Grouped 5-fold CV on train; test used once."),
("5 Modelling","XGBoost search space justified","N","Space is an assumption (docx has none); stated."),
("6 Data leakage","Train/test separated; duplicates removed","Y","0 shared domains between partitions (grouped by registered domain)."),
("6 Data leakage","No feature uses the label or future information","P","HasHTTPS (not used) is a collection artefact; used features validated; brand list was derived from positive-class tokens (mild leakage risk for HasBrandInSubdomainOrPath, only in the re-implemented extractor used by E6/E4/webapp)."),
("6 Data leakage","Pre-processing / calibration fit on training only","Y","Calibrator and thresholds derived from out-of-fold scores (oof_cal, oof_raw stored)."),
("6 Data leakage","Random-split sensitivity","Y","Random URL split inflates PR-AUC (XGB 0.979 vs 0.958); 62.8% test URLs on seen domains."),
("7 Metrics","Metrics appropriate for imbalance and use case","Y","ROC-AUC, PR-AUC, TPR@1%FPR, precision/recall at tau, ECE."),
("7 Metrics","Uncertainty reported","Y","Domain-cluster bootstrap CIs; corrected t-test; Holm correction."),
("7 Metrics","Baselines and ablations","P","E2 feature-group ablation done; no simple baseline (e.g. logistic regression/blacklist)."),
("7 Metrics","Robustness/perturbation tests","Y","E6: URL perturbations and QR noise; padding attack drops TPR@1%FPR by 0.6."),
("7 Metrics","Explanation of model behaviour","Y","SHAP (RF on 1,000 URLs, boosting on 5,000), feature importance."),
("8 Generalisation","External validation on independent data","P","E4 fresh set only (n=1,050), confounded by dead URLs; not a full external validation."),
("8 Generalisation","Subgroup/failure analysis","P","RF failure on path depth 2–3 (FNR ~54–57%); no TLD or brand subgroup analysis."),
("8 Generalisation","Deployment/base-rate effects","Y","Precision falls to 0.616 at 1% base rate."),
("8 Generalisation","Limitations stated","Y","Section 13 and this document."),
]
tbl(pd.DataFrame(R, columns=["Module", "Item", "Status", "Evidence / gap"]), "Table P1. REFORMS checklist")
c = pd.Series([r[2] for r in R]).value_counts(); P(f"Summary: {len(R)} items: Y={c.get('Y',0)}, P={c.get('P',0)}, N={c.get('N',0)}. The open gaps are provenance and label-noise audit, a trivial baseline, genuine external validation, reproducibility packaging (lockfile), and the H1 definition.")
H("D. Updated 'not run' statement", 1)
P("Previously listed as not run: E2, E4, E6 and the REFORMS checklist. All four are now done (E2/E4/E6 in Experiments_E2_E4_E6_Report.docx; REFORMS in Table P1). Still not done: a 25-feature retrain with re-tuning, original XGBoost search space, SHAP-RF on 5,000 URLs, browser UI test.")
D.save("Items_8_9_10_Report.docx"); print("saved")
