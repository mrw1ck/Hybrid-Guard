import json, pandas as pd
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches
R = Path("artifacts/rf_thesis/report"); S = json.load(open(R/"summary.json")); T = S["test"]
rd = lambda n: pd.read_csv(R/f"{n}.csv")
d = Document(); d.styles["Normal"].font.name = "Times New Roman"; d.styles["Normal"].font.size = Pt(11)
def P(t): d.add_paragraph(t)
def H(t, l=1): d.add_heading(t, l)
def fig(f, cap, w=6.3): d.add_picture(str(R/f), width=Inches(w)); d.add_paragraph(cap).italic = True
def tab(df, cap, fmt=None):
    d.add_paragraph(cap).runs[0].bold = True
    t = d.add_table(rows=1, cols=len(df.columns)); t.style = "Light Grid Accent 1"
    for i, c in enumerate(df.columns): t.rows[0].cells[i].text = str(c)
    for _, r in df.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(r):
            cells[i].text = "" if pd.isna(v) else ((f"{int(v):,}" if v.is_integer() and v != 0 else f"{v:,.4f}") if isinstance(v, float) else (f"{v:,}" if isinstance(v, int) else str(v)))
    P("")
pc = lambda x: f"{x*100:.2f}%"
r4 = rd("R4_test_metrics").set_index("Metric")
ci = lambda k, p=False: (f"{r4.loc[k,'Value']*100:.2f}% [{r4.loc[k,'CI lower']*100:.2f}%, {r4.loc[k,'CI upper']*100:.2f}%]" if p
                         else f"{r4.loc[k,'Value']:.4f} [{r4.loc[k,'CI lower']:.4f}, {r4.loc[k,'CI upper']:.4f}]")
cf = S["confusion_at_tau_high"]; r9 = rd("R9_split_sensitivity"); r10 = rd("R10_strata"); r11 = rd("R11_feature_importance")

d.add_heading("EXPERIMENTAL REPORT", 0)
P(f"Model 2: Random Forest on static URL features\nHybrid-Guard, trained under the protocol of Chapter 3 · model version {S['version']}")

H("1. Summary of findings")
for t in [
 f"A Random Forest was tuned and trained on the 520,616 training URLs of the URL Decoded Dataset and evaluated once on the locked test partition of 130,154 URLs. The test partition shares no registered domain with the training partition (verified programmatically). All confidence intervals are 95% domain-cluster bootstrap intervals ({S['bootstrap']:,} replicates).",
 f"Ranking quality: ROC-AUC {ci('ROC-AUC')} and PR-AUC {ci('PR-AUC')}.",
 f"Primary operating metric: TPR at 1% FPR is {ci('TPR@1%FPR',True)}; TPR at 0.1% FPR is {ci('TPR@0.1%FPR',True)} (indicative only).",
 f"At the frozen threshold τ_high = {S['tau_high']:.4f}: precision {T['Precision']:.4f}, recall {T['Recall (TPR)']:.4f}, F1-score {T['F1-score']:.4f}, MCC {T['MCC']:.4f}, FPR {pc(T['FPR'])} ({T['FPR']*1e4:.1f} false alarms per 10,000 benign scans).",
 f"Three-level output: {S['coverage']*100:.1f}% of test URLs receive a definite Safe or Malicious verdict; on those URLs the F1-score is {S['f1_on_definite']:.4f}. The remaining {100-S['coverage']*100:.1f}% are labelled Suspicious.",
 f"Split sensitivity: a random URL-level split places {S['split_sensitivity_seen_domains']*100:.1f}% of test URLs on domains seen in training and reports PR-AUC {r9.iloc[1]['PR-AUC']:.4f} and TPR at 1% FPR {pc(r9.iloc[1]['TPR@1%FPR'])}, against {r9.iloc[0]['PR-AUC']:.4f} and {pc(r9.iloc[0]['TPR@1%FPR'])} under the domain-grouped split.",
 "Shortcut check: no single feature exceeds an AUC of 0.95 (strongest: PathRatio, 0.788); the URL-length-only reference tree reaches PR-AUC 0.6516 and TPR at 1% FPR 26.45%.",
 f"Calibration: isotonic calibration lowers the test ECE from {S['ece_raw']:.3f} to {S['ece_cal']:.3f} and the Brier score from {S['brier_raw']:.4f} to {S['brier_cal']:.4f}.",
 f"Latency: static scoring (feature extraction + inference + calibration) takes a median of {rd('R12_latency').iloc[3]['Median ms']:.2f} ms per URL at batch size 1."]:
    d.add_paragraph(t, style="List Bullet")

H("2. Experimental setup")
H("2.1 Data and split", 2)
P("The experiment uses Dataset_Hybrid.csv (650,770 URLs, 60% benign / 40% malicious; 43 columns including 14 QR columns, identical to the earlier Dataset.csv in all shared columns). The partition column assigns every registered domain to either training or test; no domain is shared. Within the training partition, five StratifiedGroupKFold folds grouped by registered domain were used for every tuning trial and for the out-of-fold (OOF) predictions.")
tab(rd("R1_partitions"), "Table R1. Data partitions used in the experiment")
H("2.2 Features", 2)
P("The model uses the same 19 static features as the LightGBM model: HostRatio, PathRatio, NoOfDots, NoOfHyphens, NoOfDigits, NoOfSlashes, NoOfQueryParams, DigitRatio, SpecialCharRatio, NoOfObfuscatedChar, HostEntropy, PathEntropy, HasAtSymbol, SubdomainRatio, IsPunycode, HasNonStandardPort, HasBrandInSubdomainOrPath, IsShortener, HasEmbeddedURL. Using an identical feature set keeps the comparison with Model 1 fair.")
P("The 14 QR columns in Dataset_Hybrid.csv are not inputs of this model (19 URL features only). Their effect is evaluated in experiment E8 (QR_Features_E8_Report.docx) with this same Random Forest and tuned hyper-parameters: QR only gives PR-AUC 0.600 and TPR@1%FPR 0.076; URL+QR gives PR-AUC 0.9696 and TPR@1%FPR 0.817 (URL only: 0.9599 and 0.7824), almost entirely from QRPayloadLength.")
H("2.3 Training protocol", 2)
for t in ["Tuning: Optuna TPE sampler, 50 trials, objective = mean PR-AUC over the five grouped folds, class_weight=\"balanced\". No pruner was used (all 50 trials ran to completion), which removes the pruning deviation noted for Model 1.",
          "Refit: the best configuration was refitted on the whole training partition.",
          "Calibration: isotonic regression fitted on the OOF predictions.",
          "Thresholds: τ_high is the lowest calibrated score with OOF FPR ≤ 1%; τ_low is the highest calibrated score with OOF FNR ≤ 2%. Shortener / embedded-URL URLs are never labelled Safe.",
          "Test set: used exactly once, after tuning, calibration and thresholds were frozen."]:
    d.add_paragraph(t, style="List Bullet")

H("3. Hyperparameter tuning")
bp = S["best_params"]
P(f"50 trials completed in {S['tuning_minutes']:.0f} minutes. The best trial (#{S['best_trial']}) reached a mean cross-validated PR-AUC of {S['best_cv_pr_auc']:.4f}.")
tab(pd.DataFrame({"Hyperparameter": ["n_estimators","max_depth","max_features","min_samples_leaf","min_samples_split"],
    "Search range": ["100–300","unlimited or 8–40","sqrt, log2, 0.3, 0.5, 0.8","3–30","2–20"],
    "Selected value": [bp["n_estimators"],bp["max_depth"],bp["max_features"],bp["min_samples_leaf"],bp["min_samples_split"]]}),
    "Table R2. Search space and selected Random Forest hyperparameters")
fig("figR1_optuna_history.png", "Figure R1. Optuna optimisation history")
P("Interpretation: 49 of 50 trials lie between 0.957 and 0.964 mean CV PR-AUC, so the model is insensitive to hyperparameters. The only clearly weak trial (0.930) used max_depth = 8; shallow trees limit the forest. Best results come from deep trees (max_depth 36 or unlimited), max_features = 0.3 and a small min_samples_leaf.")
tab(rd("R3_oof_folds"), "Table R3. Out-of-fold performance of the selected configuration (grouped 5-fold CV)")
P("Interpretation: fold-to-fold SD of PR-AUC is about 0.003, so the estimate is stable. Fold 5 is consistently the hardest.")

H("4. Calibration and frozen thresholds")
P(f"On OOF predictions, τ_high = {S['tau_high']:.4f} gives FPR {S['oof_fpr_at_tau_high']*100:.2f}% and τ_low = {S['tau_low']:.4f} gives FNR {S['oof_fnr_at_tau_low']*100:.2f}%. On the test set, calibrated ECE is {S['ece_cal']:.3f} (raw {S['ece_raw']:.3f}); Brier {S['brier_cal']:.4f} (raw {S['brier_raw']:.4f}). Raw RF scores are biased upward by class_weight=\"balanced\", so calibration is required before thresholding.")
fig("figR2_reliability.png", "Figure R2. Reliability diagram on the locked test set", 4.5)

H("5. Results on the locked test set")
t4 = r4.reset_index().copy(); t4["95% CI"] = [f"[{a:.4f}, {b:.4f}]" for a, b in zip(t4["CI lower"], t4["CI upper"])]
tab(t4[["Metric","Value","95% CI"]], "Table R4. Test-set performance of Random Forest (n = 130,154; thresholded metrics at τ_high)")
u = S["uncalibrated_at_0.5"]
P(f"At τ_high the model produces {cf['TP']:,} true positives, {cf['FP']:,} false positives, {cf['FN']:,} false negatives and {cf['TN']:,} true negatives. τ_high is a conservative operating point (1% FPR target), which explains the high precision and lower recall. For comparison, uncalibrated scores thresholded at 0.5 give precision {u['Precision']:.4f}, recall {u['Recall (TPR)']:.4f}, F1 {u['F1-score']:.4f} and FPR {pc(u['FPR'])}.")
fig("figR3_roc_pr.png", "Figure R3. ROC curve (log FPR axis) and precision-recall curve; the dot marks τ_high", 6.5)

H("6. Three-level risk output")
P(f"A calibrated score ≥ τ_high is Malicious, ≤ τ_low is Safe (except shortener / embedded-URL cases), all others Suspicious (candidates for Deep Scan). Coverage is {S['coverage']*100:.1f}%. About 44.5% of benign URLs fall into Suspicious: the 2% FNR target for τ_low is strict and costs benign coverage.")
tab(rd("R5_risk_levels"), "Table R5. Distribution of risk levels on the test set")
fig("figR4_risk_levels.png", "Figure R4. Share of each risk level per true class", 5.5)

H("7. Precision under realistic base rates")
r6 = rd("R6_base_rates"); P(f"Precision at the dataset's 40% phishing rate overstates deployed performance. By Bayes' rule, precision at τ_high falls to {r6.iloc[3,1]:.3f} at a 1% base rate. These rates are scenarios, not estimates.")
tab(r6, "Table R6. Precision, PR-AUC and false alarms at different phishing base rates")
fig("figR5_base_rate.png", "Figure R5. Precision at τ_high as a function of the phishing base rate", 4.8)

H("8. Shortcut audit and reference models")
P("No feature exceeds the 0.95 flag level, and the length-only tree is far below the Random Forest, so the model does not rely on a trivial shortcut.")
tab(rd("R7_reference_models"), "Table R7. Reference models against Random Forest on the test set")
tab(rd("R8_single_feature_auc").head(10), "Table R8. Single-feature ROC-AUC on the training partition (top 10)")

H("9. Split sensitivity (experiment E3)")
P(f"The selected configuration was also trained on a random URL-level split of the same size. {S['split_sensitivity_seen_domains']*100:.1f}% of its test URLs belong to domains also seen in training (data-snooping risk P3). The random split overstates PR-AUC by {r9.iloc[1]['PR-AUC']-r9.iloc[0]['PR-AUC']:.4f} and TPR at 1% FPR by {(r9.iloc[1]['TPR@1%FPR']-r9.iloc[0]['TPR@1%FPR'])*100:.2f} percentage points.")
tab(r9, "Table R9. Effect of the split strategy")

H("10. Results by stratum")
w = r10[~r10.Stratum.isin(["Hard URLs","Other URLs"])].sort_values("FNR", ascending=False).iloc[0]
P(f"The hard-URL stratum (shorteners / embedded URLs) has 913 test URLs, 95.5% malicious; none are labelled Safe. Among path-depth strata the highest FNR at τ_high occurs for \"{w.Stratum}\" ({w.FNR*100:.1f}%), so malicious URLs with this path shape are the main source of missed detections at the strict threshold. The hard-URL FPR rests on only 41 benign URLs and is imprecise.")
tab(r10, "Table R10. Performance at τ_high by stratum")

H("11. Feature importance and explanations")
P(f"Permutation importance (drop in PR-AUC) ranks {', '.join(r11.Feature[:3])} as the three most important features. Gini share and mean |SHAP| (probability units, 1,000 test URLs) are shown for comparison. NoOfObfuscatedChar, HasAtSymbol and IsPunycode contribute almost nothing because they are very rare in this dataset.")
tab(r11.round(4), "Table R11. Feature importance of the final Random Forest")
fig("figR6_permutation.png", "Figure R6. Permutation importance on held-out test URLs", 5.5)
fig("figR7_shap.png", "Figure R7. SHAP summary for 1,000 test URLs; positive values push towards Malicious", 5.2)

H("12. Latency of the static scoring path")
P("Latency was measured for single URLs (batch size 1), 1,000 calls after 50 warm-up calls, with n_jobs=1. Feature extraction uses the re-implemented extractor (url_ml/thesis_features.py; the original extractor is not in the repository) and is negligible. Random Forest inference dominates (273 trees, depth up to 36) and varies between runs on this machine (a first measurement gave a median of 4.5 ms, repeat measurements 10-11 ms), so it should be read as roughly 5-10 ms, far slower than the boosting models (about 0.4 ms).")
tab(rd("R12_latency").round(3), "Table R12. Per-stage latency (ms)")

H("13. Comparison with the other models and deviations from the Chapter 3 protocol")
def ld(k):
    z = json.load(open(f"artifacts/{k}_thesis/report/summary.json")); return z
C0 = pd.read_csv("artifacts/comparison/C0_model_summary.csv").set_index("Model")
c4 = pd.read_csv("artifacts/comparison/C4_paired_bootstrap.csv"); c1 = pd.read_csv("artifacts/comparison/C1_corrected_t_test.csv")
rows = [("CV PR-AUC (5 folds)", "CV PR-AUC", "{:.4f}"), ("ROC-AUC", "ROC-AUC", "{:.4f}"), ("PR-AUC", "PR-AUC", "{:.4f}"), ("TPR@1% FPR", "TPR@1%FPR", "pct"),
        ("TPR@0.1% FPR", "TPR@0.1%FPR", "pct"), ("Precision at τ_high", "Precision", "{:.4f}"), ("Recall at τ_high", "Recall", "{:.4f}"), ("F1 at τ_high", "F1", "{:.4f}"),
        ("FPR at τ_high", "FPR", "pct"), ("ECE (calibrated)", "ECE", "{:.3f}"), ("Coverage", "Coverage", "pct1")]
f = lambda v, k: f"{v*100:.2f}%" if k == "pct" else (f"{v*100:.1f}%" if k == "pct1" else k.format(v))
lg = pd.DataFrame({"Metric": [r[0] for r in rows], **{m: [f(C0.loc[m, r[1]], r[2]) for r in rows] for m in ["LightGBM", "XGBoost", "Random Forest"]}})
tab(lg, "Table R14. Random Forest versus the rebuilt LightGBM and XGBoost on the locked test set (point estimates, same folds and features)")
P("LightGBM here is the rebuild without pruner (the original report's values were ROC-AUC 0.9637, PR-AUC 0.9598, TPR at 1% FPR 79.33%; they differ slightly because the original artifact was lost and could not be reproduced exactly).")
t1 = c1.copy(); t1["Mean dPR-AUC"] = t1["Mean dPR-AUC"].map("{:+.4f}".format); t1["p"] = t1["p"].map("{:.4f}".format); t1["p (Holm)"] = t1["p (Holm)"].map("{:.4f}".format)
tab(t1[["Pair", "Mean dPR-AUC", "t (df=4)", "p (Holm)", "Folds won by first"]], "Table R15. Corrected resampled t-test on cross-validated PR-AUC (Holm-adjusted)")
t4 = c4[c4.Pair.str.contains("Random Forest")].copy()
t4["Difference [95% CI]"] = [f"{a:+.4f} [{b:+.4f}, {c:+.4f}]" for a, b, c in zip(t4["Diff (first - second)"], t4["CI lower"], t4["CI upper"])]; t4["p (Holm)"] = t4["p (Holm)"].map("{:.2f}".format)
tab(t4[["Pair", "Metric", "Difference [95% CI]", "p (Holm)"]], "Table R16. Paired domain-cluster bootstrap on the test set, pairs involving Random Forest")
P("Reading: in cross-validation both boosting models beat Random Forest on all five folds by about 0.002 PR-AUC (Holm p = 0.007 for LightGBM, 0.010 for XGBoost), so H1 gets weak support there. On the locked test set every paired-bootstrap interval contains zero (Holm p = 1.0): the three models cannot be separated, and Random Forest even has the highest point estimate of ROC-AUC and PR-AUC. The DeLong and McNemar results (see Model_Comparison_Report.docx) give very small p-values but treat the 130,154 URLs as independent although they come from about 57,000 domains, so they overstate significance. Random Forest operates at the most conservative point (precision 0.991, FPR 0.47%) and the boosting models at higher recall; this reflects each model's own τ_high, not ranking quality.")
for t in ["Feature set: 19 implemented features instead of the 25 of Table 3.5 (all three models).",
          "Models: XGBoost and LightGBM were trained under the same protocol (same folds, seed, 50 trials, no pruner); see XGBoost_Model3_Report.docx and LightGBM_Rebuilt_Model1_Report.docx. The XGBoost search space is an assumption and LightGBM is a rebuild.",
          "Pruning: none was used for any of the three models.",
          "τ_low: FNR target 2%, as in the other models.",
          "SHAP: 1,000 URLs for Random Forest (exact TreeSHAP on the deep forest needed 85 minutes), 5,000 for the boosting models; probability units for Random Forest, log-odds for the boosting models.",
          "Latency: R12 now includes the re-implemented feature extractor (about 0.013 ms median).",
          "E2 (feature-group ablation), E6 (URL perturbation and QR round trip) and E4 (hybrid experiments) were run for all three models; results are in Experiments_E2_E4_E6_Report.docx. Main findings: the structure-count group matters most; the models are sensitive to benign padding of the URL path; and the hybrid gain on historical URLs is confounded by dead links.",
          "REFORMS checklist: done, see Table P1 in Items_8_9_10_Report.docx.",
          "Deployment: the web application now loads the calibrated thesis models (artifacts/{lgb,xgb,rf}_thesis) with the frozen thresholds; see Items_8_9_10_Report.docx, part B."]:
    d.add_paragraph(t, style="List Bullet")

H("14. Reproducibility")
P(f"Command: python train_rf_thesis.py --data Dataset_Hybrid.csv --output-dir artifacts/rf_thesis (seed 42). Optuna study: artifacts/rf_thesis/optuna_study.db. Model: url_phishing_model.joblib (RF + isotonic calibrator + thresholds); feature_schema.json; version {S['version']}. Total run time: {S['total_minutes']:.0f} minutes (tuning {S['tuning_minutes']:.0f}).")
tab(rd("R13_environment"), "Table R13. Software environment")
d.save("RandomForest_Model2_Report.docx"); print("saved")
