import sys, json, pandas as pd, numpy as np
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches
K = sys.argv[1]; NAME = {"lgb": "LightGBM", "xgb": "XGBoost"}[K]; NUM = {"lgb": 1, "xgb": 3}[K]
R = Path(f"artifacts/{K}_thesis/report"); S = json.load(open(R / "summary.json")); T = S["test"]; X = json.load(open(R / "extras.json"))
rd = lambda n: pd.read_csv(R / f"{n}.csv")
d = Document(); d.styles["Normal"].font.name = "Times New Roman"; d.styles["Normal"].font.size = Pt(11)
def P(t): d.add_paragraph(t)
def B(t): d.add_paragraph(t, style="List Bullet")
def H(t, l=1): d.add_heading(t, l)
def fig(f, cap, w=6.3): d.add_picture(str(R / f), width=Inches(w)); d.add_paragraph(cap)
def tab(df, cap):
    d.add_paragraph(cap).runs[0].bold = True
    t = d.add_table(rows=1, cols=len(df.columns)); t.style = "Light Grid Accent 1"
    for i, c in enumerate(df.columns): t.rows[0].cells[i].text = str(c)
    for _, r in df.iterrows():
        cs = t.add_row().cells
        for i, v in enumerate(r):
            if isinstance(v, str):
                try: v = float(v) if v.replace('.', '', 1).isdigit() and '.' in v else v
                except ValueError: pass
            cs[i].text = "" if (not isinstance(v, str) and pd.isna(v)) else ((f"{int(v):,}" if float(v).is_integer() and v != 0 else f"{v:,.4f}") if isinstance(v, (float, np.floating)) else (f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)))
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for run in p.runs: run.font.size = Pt(9)
    P("")
pc = lambda x: f"{x*100:.2f}%"
r4 = rd("R4_test_metrics").set_index("Metric")
ci = lambda k, p=False: (f"{r4.loc[k,'Value']*100:.2f}% [{r4.loc[k,'CI lower']*100:.2f}%, {r4.loc[k,'CI upper']*100:.2f}%]" if p else f"{r4.loc[k,'Value']:.4f} [{r4.loc[k,'CI lower']:.4f}, {r4.loc[k,'CI upper']:.4f}]")
cf = S["confusion_at_tau_high"]; r9 = rd("R9_split_sensitivity"); r10 = rd("R10_strata"); r11 = rd("R11_feature_importance"); r12 = rd("R12_latency"); r6 = rd("R6_base_rates"); r8 = rd("R8_single_feature_auc"); r7 = rd("R7_reference_models")
r3 = rd("R3_oof_folds"); bp = S["best_params"]
d.add_heading("EXPERIMENTAL REPORT", 0)
P(f"Model {NUM}{' (rebuilt)' if K=='lgb' else ''}: {NAME} on static URL features\nHybrid-Guard, trained under the protocol of Chapter 3 · model version {S['version']}")
if K == "lgb":
    P("Note: the original LightGBM artifact and training script (LightGBM_Model1_Report.docx) are no longer available. This model was rebuilt with the same search space (Table 3.7), seed, folds and 19 features, but with the median pruner removed (50 of 50 trials completed) so that the three models are compared under an identical budget. Numbers therefore differ slightly from the original report.")
else:
    P("Note: Chapter 3 does not specify the XGBoost search space. The one below is an assumption chosen to mirror the LightGBM space; it should be stated as such in the thesis.")
H("1. Summary of findings")
for t in [
 f"{NAME} was tuned and trained on 520,616 training URLs and evaluated once on the locked test partition of 130,154 URLs, which shares no registered domain with training (verified programmatically). Confidence intervals are 95% domain-cluster bootstrap intervals ({S['bootstrap']:,} replicates).",
 f"Ranking quality: ROC-AUC {ci('ROC-AUC')}, PR-AUC {ci('PR-AUC')}.",
 f"Primary operating metric: TPR at 1% FPR is {ci('TPR@1%FPR',True)}; TPR at 0.1% FPR {ci('TPR@0.1%FPR',True)} (indicative only).",
 f"At the frozen τ_high = {S['tau_high']:.4f}: precision {T['Precision']:.4f}, recall {T['Recall (TPR)']:.4f}, F1 {T['F1-score']:.4f}, MCC {T['MCC']:.4f}, FPR {pc(T['FPR'])} ({T['FPR']*1e4:.1f} false alarms per 10,000 benign scans).",
 f"Three-level output: {S['coverage']*100:.1f}% of test URLs receive a definite Safe or Malicious verdict (F1 on those: {S['f1_on_definite']:.4f}); {100-S['coverage']*100:.1f}% are Suspicious.",
 f"Split sensitivity: a random URL-level split puts {X['split_sensitivity_seen_domains']*100:.1f}% of test URLs on seen domains and reports PR-AUC {r9.iloc[1]['PR-AUC']:.4f} / TPR@1%FPR {pc(r9.iloc[1]['TPR@1%FPR'])}, against {r9.iloc[0]['PR-AUC']:.4f} / {pc(r9.iloc[0]['TPR@1%FPR'])} under the grouped split.",
 f"Shortcut check: strongest single feature {r8.iloc[0]['Feature']} (AUC-distance {r8.iloc[0]['max(AUC,1-AUC)']:.3f}); the length-only tree reaches PR-AUC {r7.iloc[1]['PR-AUC']:.4f}.",
 f"Calibration: ECE falls from {S['ece_raw']:.3f} (raw) to {S['ece_cal']:.3f}; Brier from {S['brier_raw']:.4f} to {S['brier_cal']:.4f}.",
 f"Latency: median {r12.iloc[3]['Median ms']:.2f} ms per URL at batch size 1, including feature extraction ({r12.iloc[0]['Median ms']:.3f} ms)."]: B(t)
H("2. Experimental setup")
H("2.1 Data and split", 2)
P("Dataset_Hybrid.csv: 650,770 URLs and 43 columns (the 19 URL features, identifiers/partition/label columns, and 14 QR columns added in this version; the earlier Dataset.csv has the same rows and identical values in all shared columns). The partition column assigns every registered domain wholly to training or test. Five StratifiedGroupKFold folds grouped by registered domain are used for every trial and for OOF predictions.")
tab(rd("R1_partitions"), "Table R1. Data partitions used in the experiment")
H("2.2 Features", 2)
P("The same 19 static features as the other two models (HostRatio, PathRatio, NoOfDots, NoOfHyphens, NoOfDigits, NoOfSlashes, NoOfQueryParams, DigitRatio, SpecialCharRatio, NoOfObfuscatedChar, HostEntropy, PathEntropy, HasAtSymbol, SubdomainRatio, IsPunycode, HasNonStandardPort, HasBrandInSubdomainOrPath, IsShortener, HasEmbeddedURL). Table 3.5 lists 25 features; only these 19 exist in the data and extractor, so Table 3.5 is corrected to the 19 (mapping in Table T1 of Items_8_9_10_Report.docx). Experiment E7 refitted the tuned model on a reconstructed 25-feature set: PR-AUC rose by about 0.013 and TPR@1%FPR by about 4.5 points, but most of it comes from HasHTTPS, which is a dataset-construction artefact (70% of URLs had the scheme added by the dataset builder); without it the gain is about 0.004 and 2.2 points.")
P("Dataset_Hybrid.csv also contains 14 QR-image columns (12 of them non-constant). They are NOT inputs of this model: it uses only the 19 URL features above, so its results are unchanged by the new columns (verified: the Random Forest refit on this file reproduces the stored test metrics to 4 decimals). QR features were evaluated for Random Forest only in experiment E8 (QR_Features_E8_Report.docx): QR alone is weak (PR-AUC 0.600), and URL+QR gains only about 0.01 PR-AUC, almost entirely from QRPayloadLength (absolute URL length).")
H("2.3 Training protocol", 2)
for t in ["Tuning: Optuna TPE, 50 trials, objective = mean PR-AUC over five grouped folds, NopPruner, seed 42.",
          "Refit on the whole training partition; isotonic calibration fitted on OOF predictions.",
          "τ_high: lowest calibrated score with OOF FPR ≤ 1%; τ_low: highest calibrated score with OOF FNR ≤ 2%. Shortener / embedded-URL URLs are never Safe.",
          "Test set used once, after everything was frozen."]: B(t)
H("3. Hyperparameter tuning")
P(f"50 trials finished in {S['tuning_minutes']:.0f} minutes. Best trial #{S['best_trial']} reached mean CV PR-AUC {S['best_cv_pr_auc']:.4f}.")
RANGES = {"lgb": {"n_estimators": "100–1,000", "num_leaves": "15–255", "max_depth": "−1 or 3–12", "learning_rate": "0.01–0.3 (log)", "feature_fraction": "0.6–1.0", "bagging_fraction": "0.6–1.0 (freq = 1)", "min_child_samples": "10–100"},
          "xgb": {"n_estimators": "100–1,000", "max_depth": "3–12", "learning_rate": "0.01–0.3 (log)", "subsample": "0.6–1.0", "colsample_bytree": "0.6–1.0", "min_child_weight": "1–10", "reg_lambda": "0.01–10 (log)"}}[K]
r2 = rd("R2_params"); r2.insert(1, "Search range", [RANGES.get(h, "") for h in r2.Hyperparameter])
tab(r2, f"Table R2. Search space and selected {NAME} hyperparameters" + (" (Table 3.7)" if K == "lgb" else " (assumed, not in Chapter 3)"))
fig("figR1_optuna_history.png", "Figure R1. Optuna optimisation history (no trial pruned)")
m = r3.iloc[:5]; import optuna; optuna.logging.set_verbosity(optuna.logging.ERROR)
_st = optuna.load_study(study_name=f"{K}_thesis", storage=f"sqlite:///artifacts/{K}_thesis/optuna_study.db"); _v = np.array([t.value for t in _st.trials if t.value is not None])
_w = _st.trials[int(np.argmin(_v))].params
P(f"Interpretation: {int((_v > _v.max() - 0.002).sum())} of {len(_v)} trials lie within 0.002 of the best value (median {np.median(_v):.4f}, best {_v.max():.4f}), so the model is insensitive to hyperparameters near the optimum. The weakest trial ({_v.min():.4f}) used max_depth = {_w.get('max_depth')}, a shallow-tree setting that limits capacity. Fold PR-AUC ranges from {m['PR-AUC'].astype(float).min():.4f} to {m['PR-AUC'].astype(float).max():.4f}; the hardest fold is fold {int(m.loc[m['PR-AUC'].astype(float).idxmin(),'Fold'])}.")
tab(r3, "Table R3. Out-of-fold performance of the selected configuration (grouped 5-fold CV)")
H("4. Calibration and frozen thresholds")
P(f"On OOF predictions τ_high = {S['tau_high']:.4f} gives FPR {S['oof_fpr_at_tau_high']*100:.2f}% and τ_low = {S['tau_low']:.4f} gives FNR {S['oof_fnr_at_tau_low']*100:.2f}%. On the test set calibrated ECE is {S['ece_cal']:.3f} (raw {S['ece_raw']:.3f}). The class-balancing weights bias raw scores upward, so calibration is needed before thresholding.")
fig("figR2_reliability.png", "Figure R2. Reliability diagram on the locked test set", 4.5)
H("5. Results on the locked test set")
t4 = r4.reset_index().copy(); t4["95% CI"] = [f"[{a:.4f}, {b:.4f}]" for a, b in zip(t4["CI lower"], t4["CI upper"])]
tab(t4[["Metric", "Value", "95% CI"]], f"Table R4. Test-set performance of {NAME} (n = 130,154; thresholded metrics at τ_high)")
P(f"Reading: the CI of PR-AUC is wide ([{r4.loc['PR-AUC','CI lower']:.3f}, {r4.loc['PR-AUC','CI upper']:.3f}]) because the bootstrap resamples whole domains; it reflects the roughly 57,000 independent domains, not 130,154 independent URLs. ")
P(f"At τ_high: {cf['TP']:,} true positives, {cf['FP']:,} false positives, {cf['FN']:,} false negatives, {cf['TN']:,} true negatives. τ_high is conservative (1% FPR target on OOF), hence high precision and moderate recall. The test FPR of {pc(T['FPR'])} differs from the 1% target because the OOF-to-test shift in score distribution is not zero.")
fig("figR3_roc_pr.png", "Figure R3. ROC curve (log FPR axis) and precision-recall curve; the dot marks τ_high", 6.5)
H("6. Three-level risk output")
r5 = rd("R5_risk_levels"); _b = r5.iloc[0]; _m = r5.iloc[1]
P(f"Coverage (definite verdicts) is {S['coverage']*100:.1f}%. Benign URLs: Safe {_b['Safe']}, Suspicious {_b['Suspicious']}, Malicious {_b['Malicious']}. Malicious URLs: Safe {_m['Safe']}, Suspicious {_m['Suspicious']}, Malicious {_m['Malicious']}. The Suspicious share is concentrated among benign URLs because the 2% FNR target for τ_low is strict: it costs benign coverage to keep missed malicious URLs (labelled Safe) low. Suspicious URLs are the candidates for a Deep Scan.")
tab(r5, "Table R5. Distribution of risk levels on the test set"); fig("figR4_risk_levels.png", "Figure R4. Share of each risk level per true class", 5.5)
H("7. Precision under realistic base rates")
P(f"Precision at the dataset's 40% phishing rate overstates deployed performance. By Bayes' rule precision at τ_high falls to {r6.iloc[3,1]:.3f} at a 1% base rate, with the same {r6.iloc[0,3]:.1f} false alarms per 10,000 benign scans. These rates are scenarios, not estimates.")
tab(r6, "Table R6. Precision, PR-AUC and false alarms at different phishing base rates"); fig("figR5_base_rate.png", "Figure R5. Precision at τ_high versus phishing base rate", 4.8)
H("8. Shortcut audit and reference models")
P(f"No feature exceeds the 0.95 flag level (strongest {r8.iloc[0]['Feature']}, {r8.iloc[0]['max(AUC,1-AUC)']:.3f}); the length-only tree is far below {NAME}, so the model does not rest on a trivial shortcut.")
tab(r7, "Table R7. Reference models against " + NAME + " on the test set"); tab(r8.head(10), "Table R8. Single-feature ROC-AUC on the training partition (top 10)")
H("9. Split sensitivity (experiment E3)")
P(f"The selected configuration was also trained on a random URL-level split of the same size: {X['split_sensitivity_seen_domains']*100:.1f}% of its test URLs belong to domains seen in training. The random split overstates PR-AUC by {r9.iloc[1]['PR-AUC']-r9.iloc[0]['PR-AUC']:.4f} and TPR@1%FPR by {(r9.iloc[1]['TPR@1%FPR']-r9.iloc[0]['TPR@1%FPR'])*100:.2f} points, which is the data-snooping effect of domain leakage.")
tab(r9, "Table R9. Effect of the split strategy")
H("10. Results by stratum")
w = r10[r10.Stratum.str.startswith("Path")].sort_values("FNR", ascending=False).iloc[0]; hh = r10.iloc[0]
P(f"Hard URLs (shorteners / embedded URLs): {int(hh.URLs)} test URLs, {hh['Malicious share']*100:.1f}% malicious; none can be Safe. Their FPR rests on few benign URLs and is imprecise. Among path-depth strata the highest FNR at τ_high is \"{w.Stratum}\" ({w.FNR*100:.1f}%), the main source of misses at the strict threshold.")
tab(r10, "Table R10. Performance at τ_high by stratum")
H("11. Feature importance and explanations")
r11 = r11.rename(columns={"Mean |SHAP| (SHAPUNIT)": "Mean |SHAP| (log-odds)"})
_top = r11.iloc[0]; _shap = r11.sort_values("Mean |SHAP| (log-odds)", ascending=False).Feature.iloc[0]
P(f"Permutation importance ranks {', '.join(r11.Feature[:3])} highest; the top feature {_top.Feature} alone removes {_top['Permutation dPR-AUC']:.3f} PR-AUC when shuffled, and the SHAP ranking leads with {_shap}. The features at the bottom (permutation drop below 0.001) are {', '.join(r11[r11['Permutation dPR-AUC'] < 0.001].Feature)}; these rare binary flags matter little individually, although group G5 as a whole still gives a small significant loss in the ablation E2. Native importance is split-count/gain based and can disagree with permutation importance. SHAP values ({X['shap_n']:,} test URLs) are in log-odds units for this tree booster, not probability units as for Random Forest. Rare binary flags contribute little because they are rare in the data.")
tab(r11.round(4), f"Table R11. Feature importance of the final {NAME}")
fig("figR6_permutation.png", "Figure R6. Permutation importance on held-out test URLs", 5.5); fig("figR7_shap.png", f"Figure R7. SHAP summary for {X['shap_n']:,} test URLs", 5.2)
H("12. Latency of the static scoring path")
P("Single URL, batch size 1, 1,000 calls after 50 warm-up calls, n_jobs=1. Feature extraction uses the re-implemented extractor (url_ml/thesis_features.py); the original extractor is not in the repository. Network checks are excluded.")
tab(r12.round(3), "Table R12. Per-stage latency (ms)")
H("13. Deviations from the Chapter 3 protocol")
for t in ["Feature set: 19 implemented features instead of the 25 of Table 3.5 (all three models).",
          "Pruning: none; this removes the pruning deviation of the original LightGBM report." ,
          "τ_low: FNR target 2%, as in the other models.",
          ("LightGBM is a rebuild; XGBoost search space is assumed." if K == "xgb" else "The original artifact is lost; this is a rebuild and not the original model."),
          "As requested, E8 (QR vs URL features) was carried out for Random Forest only.",
          "Pairwise tests and H1 are reported in Model_Comparison_Report.docx; robustness (E6), ablation (E2) and hybrid (E4) results in Experiments_E2_E4_E6_Report.docx.",
          "Done after this report: REFORMS checklist (Table P1) and webapp update to the calibrated thesis models (see Part F / Items_8_9_10_Report.docx). Still open: gaps listed in Table P1."]: B(t)
H("14. Reproducibility")
P(f"Command: python train_boost_thesis.py --model {K} --output-dir artifacts/{K}_thesis (seed 42), then python boost_extras.py {K} 20 5000 for tables R1, R2, R6–R13 and figures R1–R7. Model: url_phishing_model.joblib (model + isotonic calibrator + thresholds); version {S['version']}. Run time: {S['total_minutes']:.0f} min (tuning {S['tuning_minutes']:.0f}).")
tab(rd("R13_environment"), "Table R13. Software environment")
out = {"lgb": "LightGBM_Rebuilt_Model1_Report.docx", "xgb": "XGBoost_Model3_Report.docx"}[K]; 
try: d.save(out)
except PermissionError: out = out.replace(".docx", "_new.docx"); d.save(out)
print("saved", out)
