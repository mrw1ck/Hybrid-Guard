import json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, optuna
from sklearn.metrics import roc_curve, precision_recall_curve
from docx import Document; from docx.shared import Cm, Pt
optuna.logging.set_verbosity(optuna.logging.ERROR)
A = {"LightGBM": "lgb", "XGBoost": "xgb", "Random Forest": "rf"}; col = {"LightGBM": "#1f77b4", "XGBoost": "#ff7f0e", "Random Forest": "#2ca02c"}
out = "artifacts/comparison/"
# fig1 ROC/PR
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5)); P = {}
for m, k in A.items():
    z = np.load(f"artifacts/{k}_thesis/predictions.npz"); print(m, list(z.keys())) if m == "LightGBM" else None; P[m] = z
    yk = [x for x in z.keys() if "y" in x and "test" in x][0]; sk = [x for x in z.keys() if "test" in x and x != yk and "y" not in x[:2]][0]
    y, s = z[yk], z[sk]; f, t, _ = roc_curve(y, s); pr, rc, _ = precision_recall_curve(y, s)
    ax[0].plot(f, t, c=col[m], label=m); ax[1].plot(rc, pr, c=col[m], label=m)
ax[0].set_xlim(0, .05); ax[0].set_ylim(.4, 1); ax[0].axvline(.01, c="gray", ls=":"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].set_title("ROC (zoom FPR<=5%) - test"); ax[0].legend()
ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_ylim(.7, 1); ax[1].set_title("Precision-Recall - test"); plt.tight_layout(); plt.savefig(out + "figC1_roc_pr.png", dpi=150); plt.close()
# fig2 CV folds + optuna
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for m, k in A.items():
    o = pd.read_csv(f"artifacts/{k}_thesis/report/R3_oof_folds.csv"); c = [x for x in o.columns if "PR" in x][0]; v = o[c].values[:5].astype(float) if o[c].dtype != object else None
    if v is not None: ax[0].plot(range(1, len(v) + 1), v, "o-", c=col[m], label=m)
    st = optuna.load_study(study_name=None, storage=f"sqlite:///artifacts/{k}_thesis/optuna_study.db")
    vs = [t.value for t in st.trials if t.value is not None]; ax[1].plot(np.maximum.accumulate(vs), c=col[m], label=m)
ax[0].set_title("CV PR-AUC per fold (same folds)"); ax[0].set_xlabel("fold"); ax[0].legend(); ax[1].set_title("Optuna best-so-far CV PR-AUC (50 trials)"); ax[1].set_xlabel("trial"); ax[1].set_ylim(.955, .967)
plt.tight_layout(); plt.savefig(out + "figC2_cv_optuna.png", dpi=150); plt.close()
# fig3 pairwise bootstrap
c4 = pd.read_csv(out + "C4_paired_bootstrap.csv"); fig, ax = plt.subplots(figsize=(8, 4.5)); lab = []
for i, r in c4.iterrows():
    ax.errorbar(r["Diff (first - second)"], i, xerr=[[r["Diff (first - second)"] - r["CI lower"]], [r["CI upper"] - r["Diff (first - second)"]]], fmt="o", c="k"); lab.append(f"{r.Pair} | {r.Metric}")
ax.axvline(0, c="r"); ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab, fontsize=7); ax.set_title("Paired domain-cluster bootstrap: difference and 95% CI"); plt.tight_layout(); plt.savefig(out + "figC3_paired_bootstrap.png", dpi=150); plt.close()
# fig4 risk levels
fig, ax = plt.subplots(figsize=(8, 4)); w = .27
for j, m in enumerate(A):
    r = pd.read_csv(f"artifacts/{A[m]}_thesis/report/R5_risk_levels.csv" if m != "Random Forest" else "artifacts/rf_thesis/report/R5_risk_levels.csv")
    num = lambda s: float(str(s).split("(")[1].rstrip("%)"))
    ax.bar(np.arange(3) + j * w, [num(r.iloc[1]["Safe"]), num(r.iloc[1]["Suspicious"]), num(r.iloc[1]["Malicious"])], w, color=col[m], label=m)
ax.set_xticks(np.arange(3) + w); ax.set_xticklabels(["Safe", "Suspicious", "Malicious"]); ax.set_ylabel("% of malicious URLs"); ax.set_title("Three-tier routing of malicious test URLs"); ax.legend(); plt.tight_layout(); plt.savefig(out + "figC4_risk_levels.png", dpi=150); plt.close()

D = Document(); D.styles["Normal"].font.name = "Times New Roman"; D.styles["Normal"].font.size = Pt(11)
def P_(t): return D.add_paragraph(t)
def Bl(t): D.add_paragraph(t, style="List Bullet")
def T(df, cap, fmt="{:.4f}"):
    P_(cap).runs[0].bold = True; t = D.add_table(rows=1, cols=len(df.columns)); t.style = "Light Grid Accent 1"
    for i, c in enumerate(df.columns): t.rows[0].cells[i].text = str(c)
    for _, r in df.iterrows():
        cs = t.add_row().cells
        for i, v in enumerate(r): cs[i].text = (f"{v:.2e}" if isinstance(v, float) and 0 < abs(v) < 1e-3 else fmt.format(v)) if isinstance(v, (float, np.floating)) else str(v)
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for run in p.runs: run.font.size = Pt(8)
def F(p, cap, w=16): D.add_picture(p, width=Cm(w)); P_(cap)
D.add_heading("So sánh ba mô hình: LightGBM, XGBoost, Random Forest (mục 2-4, H1)", 0)
P_("Cùng giao thức Chương 3: 5-fold CV phân tầng theo nhóm domain, seed 42, 19 đặc trưng, 50 trial Optuna không pruner, hiệu chỉnh xác suất, ngưỡng τ_high/τ_low đóng băng từ OOF, đánh giá một lần trên 130.154 URL test. LightGBM là bản dựng lại (bản gốc không còn). Không gian tìm kiếm XGBoost là suy đoán vì báo cáo gốc không có.")
D.add_heading("1. Kết quả trên tập test", 1)
c0 = pd.read_csv(out + "C0_model_summary.csv"); T(c0, "Bảng C0. Tóm tắt ba mô hình (ngưỡng τ_high cho Precision/Recall/F1/FPR)")
F(out + "figC1_roc_pr.png", "Hình C1. ROC (phóng đại FPR ≤ 5%) và Precision-Recall trên test.")
for m, k in A.items():
    r = pd.read_csv(f"artifacts/{k}_thesis/report/R4_test_metrics.csv"); T(r, f"Bảng. {m}: chỉ số test và CI 95% (bootstrap theo domain)")
F(out + "figC4_risk_levels.png", "Hình C4. Phân luồng ba mức của URL độc hại.", 12)
D.add_heading("Giải thích", 2)
Bl("Ba mô hình gần như ngang nhau: ROC-AUC 0,9626–0,9634, PR-AUC 0,9578–0,9599, TPR@1%FPR 77,9–78,5%. Random Forest còn nhỉnh nhất về ROC/PR-AUC trên test dù thấp nhất ở CV.")
Bl("Khác biệt nằm ở điểm vận hành: Random Forest thận trọng nhất (Precision 0,991, FPR 0,47%, Recall 74,3%), XGBoost mạnh tay nhất (FPR 0,93%). Do ngưỡng τ_high được chọn từ OOF của từng mô hình nên không so Precision/Recall trực tiếp giữa các mô hình như so năng lực phân tách.")
Bl("ECE sau hiệu chỉnh ≈ 0,021–0,023 ở cả ba, tức xác suất khá đáng tin. Khoảng 35% URL rơi vào vùng Suspicious; đây là nhược điểm chung của giao thức ba mức với 19 đặc trưng tĩnh.")
D.add_heading("2. Kiểm định giả thuyết H1 (mô hình boosting tốt hơn RF) bằng kiểm định cặp", 1)
F(out + "figC2_cv_optuna.png", "Hình C2. PR-AUC theo fold và lịch sử Optuna.")
T(pd.read_csv(out + "C1_corrected_t_test.csv"), "Bảng C1. Corrected resampled t-test (Nadeau-Bengio) trên 5 fold, hiệu chỉnh Holm")
T(pd.read_csv(out + "C2_delong.csv"), "Bảng C2. DeLong cho chênh lệch ROC-AUC trên test")
T(pd.read_csv(out + "C3_mcnemar.csv"), "Bảng C3. McNemar chính xác trên quyết định tại ngưỡng riêng của từng mô hình")
T(c4, "Bảng C4. Bootstrap ghép cặp theo cụm domain (kiểm định đáng tin nhất)")
F(out + "figC3_paired_bootstrap.png", "Hình C3. Chênh lệch giữa các cặp và CI 95%.", 14)
D.add_heading("Giải thích", 2)
Bl("CV: cả LightGBM và XGBoost đều hơn RF 5/5 fold (p Holm = 0,007 và 0,010), nhưng mức chênh chỉ khoảng 0,002 PR-AUC. LightGBM và XGBoost không khác nhau (p = 0,45). Chỉ 5 fold nên kiểm định có độ tin cậy hạn chế.")
Bl("DeLong và McNemar cho p cực nhỏ, nhưng chúng coi 130k URL là độc lập trong khi chỉ có khoảng 57k domain. Vì vậy p bị đánh giá quá thấp. McNemar còn so quyết định ở ngưỡng khác nhau nên phản ánh chênh lệch ngưỡng hơn là chất lượng. Hai bảng này chỉ nên coi là tham khảo.")
Bl("Bootstrap theo domain (tính cả tương quan trong cụm): mọi CI của chênh lệch ROC-AUC, PR-AUC, TPR@1%FPR đều chứa 0 và p Holm = 1,0. Trên test không phân biệt được ba mô hình.")
Bl("Kết luận H1: chỉ được hỗ trợ yếu ở CV (chênh nhỏ, nhất quán) và không được hỗ trợ trên test. Nên trình bày là 'ba mô hình tương đương', thay vì khẳng định boosting vượt trội.")
D.save("Model_Comparison_Report.docx"); print("saved")
