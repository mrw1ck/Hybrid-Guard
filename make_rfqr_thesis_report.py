import json, pandas as pd, numpy as np
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches
Q = Path("artifacts/rf_qr_thesis/report"); U = Path("artifacts/rf_thesis/report"); E8 = Path("artifacts/e8_qr")
SQ = json.load(open(Q/"summary.json")); SU = json.load(open(U/"summary.json")); TQ, TU = SQ["test"], SU["test"]
rd = lambda n, p=Q: pd.read_csv(p/f"{n}.csv")
d = Document(); d.styles["Normal"].font.name = "Times New Roman"; d.styles["Normal"].font.size = Pt(11)
P = lambda t: d.add_paragraph(t); H = lambda t, l=1: d.add_heading(t, l); Bu = lambda t: d.add_paragraph(t, style="List Bullet")
def fig(p, cap, w=6.3):
    d.add_picture(str(p), width=Inches(w)); d.add_paragraph(cap)
def tab(df, cap):
    d.add_paragraph(cap).runs[0].bold = True
    t = d.add_table(rows=1, cols=len(df.columns)); t.style = "Light Grid Accent 1"
    for i, c in enumerate(df.columns): t.rows[0].cells[i].text = str(c)
    for _, r in df.iterrows():
        cs = t.add_row().cells
        for i, v in enumerate(r):
            cs[i].text = "" if (not isinstance(v, str) and pd.isna(v)) else (f"{v:,.4f}" if isinstance(v, (float, np.floating)) else (f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)))
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for r in p.runs: r.font.size = Pt(8.5)
    P("")
r4 = rd("R4_test_metrics").set_index("Metric"); r4u = rd("R4_test_metrics", U).set_index("Metric")
ci = lambda k, pct=False: (f"{r4.loc[k,'Value']*100:.2f}% [{r4.loc[k,'CI lower']*100:.2f}%, {r4.loc[k,'CI upper']*100:.2f}%]" if pct else f"{r4.loc[k,'Value']:.4f} [{r4.loc[k,'CI lower']:.4f}, {r4.loc[k,'CI upper']:.4f}]")
cf = SQ["confusion_at_tau_high"]; r9 = rd("R9_split_sensitivity"); r11 = rd("R11_feature_importance")
d.add_heading("EXPERIMENTAL REPORT", 0)
P(f"Model 2b: Random Forest on URL + QR features (31 features)\nHybrid-Guard, same protocol as Model 2 (Chapter 3) · model version {SQ['version']}")
H("1. Summary of findings")
for t in [
 "A Random Forest was re-tuned and trained from scratch on Dataset_Hybrid.csv using 19 URL features and 12 non-constant QR features (31 inputs), then evaluated once on the locked test partition of 130,154 URLs (no registered domain shared with the 520,616 training URLs). Every step follows RandomForest_Model2_Report: grouped 5-fold CV, 50 Optuna trials without pruner, isotonic calibration on OOF, frozen thresholds, domain-cluster bootstrap.",
 f"Ranking quality: ROC-AUC {ci('ROC-AUC')}, PR-AUC {ci('PR-AUC')}. URL-only Random Forest: ROC-AUC {TU['ROC-AUC']:.4f}, PR-AUC {TU['PR-AUC']:.4f}.",
 f"Primary operating metric: TPR at 1% FPR = {ci('TPR@1%FPR',True)} (URL only: {TU['TPR@1%FPR']*100:.2f}%); TPR at 0.1% FPR = {ci('TPR@0.1%FPR',True)} (URL only: {TU['TPR@0.1%FPR']*100:.2f}%).",
 f"At the frozen τ_high = {SQ['tau_high']:.4f}: precision {TQ['Precision']:.4f}, recall {TQ['Recall (TPR)']:.4f}, F1 {TQ['F1-score']:.4f}, MCC {TQ['MCC']:.4f}, FPR {TQ['FPR']*100:.2f}%. URL only: precision {TU['Precision']:.4f}, recall {TU['Recall (TPR)']:.4f}, F1 {TU['F1-score']:.4f}.",
 f"Three-level output: coverage {SQ['coverage']*100:.1f}% (URL only {SU['coverage']*100:.1f}%).",
 f"Calibration: test ECE {SQ['ece_raw']:.3f} raw → {SQ['ece_cal']:.3f} calibrated.",
 "Key finding on QR vs URL: QR features alone are weak, and the gain from adding them to URL features comes almost entirely from one column, QRPayloadLength, which equals the absolute length of the raw URL (Section 9)."]:
    Bu(t)
H("2. Experimental setup")
H("2.1 Data and split", 2)
P("Dataset_Hybrid.csv: 650,770 URLs, 60% benign / 40% malicious, 43 columns. The partition column assigns every registered domain wholly to train or test. Five StratifiedGroupKFold folds grouped by registered domain are used for every tuning trial and for OOF predictions.")
tab(rd("R1_partitions"), "Table R1. Data partitions")
H("2.2 Features", 2)
P("URL features (19): HostRatio, PathRatio, NoOfDots, NoOfHyphens, NoOfDigits, NoOfSlashes, NoOfQueryParams, DigitRatio, SpecialCharRatio, NoOfObfuscatedChar, HostEntropy, PathEntropy, HasAtSymbol, SubdomainRatio, IsPunycode, HasNonStandardPort, HasBrandInSubdomainOrPath, IsShortener, HasEmbeddedURL.")
P("QR features (12): QRVersion, QRDimension, QRTotalModules, QRBlackModules, QRWhiteModules, QRDotDensity, QRECCLevel, QRDataMask, QRUnusedECC, QRPayloadLength, QRCapacityRatio, QRModulePixelSize. HasValidQR and QRImageAspectRatio are constant (always 1) and are excluded.")
H("2.3 Training protocol", 2)
for t in ["Tuning: Optuna TPE, 50 trials, objective = mean PR-AUC over the five grouped folds, class_weight=\"balanced\", NopPruner, seed 42. Unlike experiment E8, hyper-parameters are re-tuned for the 31-feature set.",
          "Refit on the whole training partition; isotonic calibration on OOF predictions.",
          "τ_high: lowest calibrated OOF score with FPR ≤ 1%; τ_low: highest calibrated OOF score with FNR ≤ 2%. Shortener / embedded-URL URLs are never Safe.",
          "Test set used once, after everything was frozen."]: Bu(t)
H("3. Hyperparameter tuning")
bp = SQ["best_params"]; bu = SU["best_params"]
P(f"50 trials completed in {SQ['tuning_minutes']:.0f} minutes. Best trial #{SQ['best_trial']} reached mean CV PR-AUC {SQ['best_cv_pr_auc']:.4f} (URL only: {SU['best_cv_pr_auc']:.4f}).")
tab(pd.DataFrame({"Hyperparameter": ["n_estimators","max_depth","max_features","min_samples_leaf","min_samples_split"],
    "Search range": ["100–300","unlimited or 8–40","sqrt, log2, 0.3, 0.5, 0.8","3–30","2–20"],
    "URL+QR (selected)": [bp[k] for k in ["n_estimators","max_depth","max_features","min_samples_leaf","min_samples_split"]],
    "URL only (selected)": [bu[k] for k in ["n_estimators","max_depth","max_features","min_samples_leaf","min_samples_split"]]}), "Table R2. Search space and selected hyperparameters")
fig(Q/"figR1_optuna_history.png", "Figure R1. Optuna optimisation history (URL+QR)")
tab(rd("R3_oof_folds"), "Table R3. Out-of-fold performance (grouped 5-fold CV), URL+QR")
tab(rd("R3_oof_folds", U), "Table R3b. Out-of-fold performance, URL only (for comparison)")
H("4. Calibration and frozen thresholds")
P(f"On OOF predictions, τ_high = {SQ['tau_high']:.4f} gives FPR {SQ['oof_fpr_at_tau_high']*100:.2f}%, τ_low = {SQ['tau_low']:.4f} gives FNR {SQ['oof_fnr_at_tau_low']*100:.2f}%. Test ECE: raw {SQ['ece_raw']:.3f}, calibrated {SQ['ece_cal']:.3f}; Brier {SQ['brier_raw']:.4f} → {SQ['brier_cal']:.4f}.")
fig(Q/"figR2_reliability.png", "Figure R2. Reliability diagram on the locked test set", 4.5)
H("5. Results on the locked test set")
t4 = r4.reset_index().copy(); t4["95% CI"] = [f"[{a:.4f}, {b:.4f}]" for a, b in zip(t4["CI lower"], t4["CI upper"])]
t4["URL only"] = [r4u.loc[m, "Value"] for m in t4.Metric]
tab(t4[["Metric", "Value", "95% CI", "URL only"]], "Table R4. Test-set performance of Random Forest URL+QR (n = 130,154), with URL-only value for reference")
P(f"At τ_high: {cf['TP']:,} true positives, {cf['FP']:,} false positives, {cf['FN']:,} false negatives, {cf['TN']:,} true negatives (URL only: {SU['confusion_at_tau_high']['TP']:,} / {SU['confusion_at_tau_high']['FP']:,} / {SU['confusion_at_tau_high']['FN']:,} / {SU['confusion_at_tau_high']['TN']:,}).")
fig(Q/"figR3_roc_pr.png", "Figure R3. ROC (log FPR) and precision-recall; dot = τ_high", 6.5)
H("6. Three-level risk output")
tab(rd("R5_risk_levels"), "Table R5. Risk levels on the test set, URL+QR")
tab(rd("R5_risk_levels", U), "Table R5b. Risk levels, URL only")
fig(Q/"figR4_risk_levels.png", "Figure R4. Share of each risk level per class", 5.5)
H("7. Precision under realistic base rates")
tab(rd("R6_base_rates"), "Table R6. Precision and false alarms at different base rates (URL+QR)")
fig(Q/"figR5_base_rate.png", "Figure R5. Precision vs base rate", 4.8)
H("8. Shortcut audit, split sensitivity and strata")
tab(rd("R7_reference_models"), "Table R7. Reference models")
tab(rd("R8_single_feature_auc").head(12), "Table R8. Single-feature ROC-AUC on training partition (top 12; QR features included)")
tab(r9, "Table R9. Effect of the split strategy (E3)")
tab(rd("R10_strata"), "Table R10. Performance by stratum")
H("9. Làm rõ số liệu QR và URL")
m = pd.read_csv(E8/"E8_metrics.csv"); dl = pd.read_csv(E8/"E8_paired_delta.csv"); dg = pd.read_csv(E8/"E8_qr_feature_diagnostics.csv")
H("9.1 Ba cấu hình cạnh nhau", 2)
def g(c, k): return m[(m.Configuration == c) & (m.Metric == k)].Value.iloc[0]
rows = [{"Cấu hình": "URL only (19), E8", **{k: g("URL only (19)", k) for k in ["ROC-AUC","PR-AUC","TPR@1%FPR","TPR@0.1%FPR"]}},
        {"Cấu hình": "QR only (12), E8", **{k: g("QR only (12)", k) for k in ["ROC-AUC","PR-AUC","TPR@1%FPR","TPR@0.1%FPR"]}},
        {"Cấu hình": "QRPayloadLength only (1), E8", **{k: g("QRPayloadLength only (1)", k) for k in ["ROC-AUC","PR-AUC","TPR@1%FPR","TPR@0.1%FPR"]}},
        {"Cấu hình": "URL + QR (31), E8 (tham số của URL only)", **{k: g("URL + QR (31)", k) for k in ["ROC-AUC","PR-AUC","TPR@1%FPR","TPR@0.1%FPR"]}},
        {"Cấu hình": "URL + QR (31), báo cáo này (tune lại)", **{k: TQ[k] for k in ["ROC-AUC","PR-AUC","TPR@1%FPR","TPR@0.1%FPR"]}}]
tab(pd.DataFrame(rows), "Bảng Q1. So sánh URL, QR và URL+QR trên test")
P("Chỉ dùng QR phát hiện kém: tại FPR 1% chỉ bắt được khoảng 7,6% URL độc hại, thấp hơn cả cây độ sâu 2 chỉ dùng độ dài URL (26,5%). URL + QR cao hơn URL only một mức nhỏ. Dòng cuối (đã tune lại) cho biết việc tune riêng cho 31 đặc trưng thay đổi kết quả bao nhiêu so với dùng lại tham số của mô hình URL.")
H("9.2 Chênh lệch có ý nghĩa thống kê không", 2)
x = dl.copy(); x["Δ [95% CI]"] = x.apply(lambda r: f"{r['Delta vs URL only']:+.4f} [{r['CI lower']:+.4f}, {r['CI upper']:+.4f}]", axis=1)
t = x.pivot(index="Configuration", columns="Metric", values="Δ [95% CI]")[["ROC-AUC","PR-AUC","TPR@1%FPR","TPR@0.1%FPR"]].reset_index()
tab(t, "Bảng Q2. Chênh lệch so với URL only (bootstrap ghép cặp theo domain, từ E8)")
P("Mọi CI đều không chứa 0: URL+QR tốt hơn URL only một cách nhất quán nhưng nhỏ (PR-AUC khoảng +0,01); QR only kém hơn rất nhiều.")
H("9.3 Từng cột QR mang thông tin gì", 2)
tab(dg[["Feature","Mean benign","Mean malicious","Distinct values","max(AUC,1-AUC)","Spearman vs URL length"]], "Bảng Q3. Chẩn đoán 14 cột QR (tập train)")
fig(E8/"figE8_3_qr_features.png", "Hình Q1. AUC đơn lẻ (trái) và tương quan hạng với độ dài URL (phải)", 6.3)
P("Bảy cột (QRVersion, QRDimension, QRTotalModules, QRBlack/WhiteModules, QRPayloadLength, QRCapacityRatio) có tương quan 0,88–0,98 với độ dài URL: chúng là cùng một thông tin (độ dài) viết theo nhiều cách. Các cột không phụ thuộc độ dài (QRDataMask, QRDotDensity, QRECCLevel, QRModulePixelSize, QRUnusedECC) có AUC đơn lẻ gần 0,5, tức gần như không mang tín hiệu.")
H("9.4 Mô hình URL+QR dựa vào đặc trưng nào", 2)
r11q = r11.copy(); r11q["Nhóm"] = np.where(r11q.Feature.str.startswith("QR"), "QR", "URL")
tab(r11q.round(4), "Bảng Q4. Tầm quan trọng đặc trưng của RF URL+QR (permutation trên test)")
fig(Q/"figR6_permutation.png", "Hình Q2. Permutation importance (URL+QR)", 5.5)
qr_imp = r11q[r11q["Nhóm"] == "QR"].head(3); url_top = r11q[r11q["Nhóm"] == "URL"].head(3)
P("Trong nhóm URL, ba đặc trưng đầu: " + ", ".join(f"{a} ({b:.4f})" for a, b in zip(url_top.Feature, url_top["Permutation dPR-AUC"])) + ". Trong nhóm QR, ba đặc trưng đầu: " + ", ".join(f"{a} ({b:.4f})" for a, b in zip(qr_imp.Feature, qr_imp["Permutation dPR-AUC"])) + ".")
if (Q/"figR7_shap.png").exists(): fig(Q/"figR7_shap.png", "Hình Q3. SHAP summary (URL+QR)", 5.2)
H("9.5 Vì sao URL + QR tăng", 2)
b = json.load(open(E8/"E8b_url_plus_payload.json"))
P(f"Kiểm tra E8b: chỉ thêm một cột QRPayloadLength vào 19 đặc trưng URL cho PR-AUC {b['PR-AUC']['value']:.4f} và TPR@1%FPR {b['TPR@1%FPR']['value']:.4f}, gần bằng cả 12 cột QR. Cột này bằng len(url_raw) ở 99,99% số dòng, tức độ dài tuyệt đối của URL. Mô hình URL only chỉ dùng tỷ lệ (HostRatio, PathRatio, …) nên không có độ dài tuyệt đối. Ảnh QR trong dataset được sinh từ chính URL, nên QR không thêm thông tin nào ngoài URL.")
H("10. Latency")
tab(rd("R12_latency").round(3), "Table R12. Latency, URL+QR (QR features assumed precomputed; QR decoding time not included)")
tab(rd("R12_latency", U).round(3), "Table R12b. Latency, URL only")
H("11. Deviations and limitations")
for t in ["Feature set: 19 URL + 12 QR features; HasValidQR and QRImageAspectRatio excluded as constants.",
          "QR images in the dataset are generated from the URL itself, so QR columns are functions of the URL; conclusions do not transfer to real-world scanned QR images with noise or backgrounds.",
          "A model that uses absolute URL length may be more sensitive to perturbations that change length (tracking parameters, path padding, see E6); E6 was not rerun for this model.",
          "SHAP uses 300 test URLs only; permutation importance uses 3 repeats.",
          "In deployment the QR features require decoding and re-analysing the QR image; since they add nothing beyond URL length, adding absolute length features to the URL set is the cheaper alternative.",
          "Rebuilt feature extractor covers the 19 URL features only; QR features come from the dataset columns, not from a re-implemented extractor.",
          "Only Random Forest was run with QR features; LightGBM / XGBoost not."]: Bu(t)
H("12. Reproducibility")
P(f"python train_rf_qr_thesis.py --data Dataset_Hybrid.csv --output-dir artifacts/rf_qr_thesis --shap-n 300 (seed 42; total {SQ['total_minutes']:.0f} min). Report: python make_rfqr_thesis_report.py. Model: artifacts/rf_qr_thesis/url_phishing_model.joblib, version {SQ['version']}.")
tab(rd("R13_environment"), "Table R13. Software environment")
d.save("RF_URL_QR_Model2b_Report.docx"); print("saved")
