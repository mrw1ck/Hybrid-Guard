import json, numpy as np, pandas as pd, joblib, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Cm, Pt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.inspection import permutation_importance
from train_rf_thesis import FEATURES as F19
QR = ["QRVersion","QRDimension","QRTotalModules","QRBlackModules","QRWhiteModules","QRDotDensity","QRECCLevel","QRDataMask","QRUnusedECC","QRPayloadLength","QRCapacityRatio","QRModulePixelSize"]
F = F19 + QR
d = pd.read_csv("Dataset_Hybrid_RF_QR.csv"); u = pd.read_csv("Dataset_Hybrid_RF.csv", usecols=["rf_prob_raw","rf_prob_cal","rf_risk_level","partition"])
art = joblib.load("artifacts/rf_thesis/url_phishing_model_qr.joblib"); th, tl = art["thresholds"]["tau_high"], art["thresholds"]["tau_low"]
te = d.partition == "test"; t = d[te].reset_index(drop=True); ut = u[te].reset_index(drop=True)
y = t.Label.values; s = t.rfqr_prob_raw.values; c = t.rfqr_prob_cal.values; s0 = ut.rf_prob_raw.values
def tpr_at(y, s, f):
    fpr, tpr, _ = roc_curve(y, s); return float(np.interp(f, fpr, tpr))
def ece(y, p, n=10):
    b = np.minimum((p * n).astype(int), n - 1); return sum((b == i).mean() * abs(y[b == i].mean() - p[b == i].mean()) for i in range(n) if (b == i).any())
dom = t.registered_domain.values; ud, inv = np.unique(dom, return_inverse=True); idx = [np.where(inv == i)[0] for i in range(len(ud))]
rng = np.random.default_rng(42); B = 300
def met(ix): return (roc_auc_score(y[ix], s[ix]), average_precision_score(y[ix], s[ix]), tpr_at(y[ix], s[ix], .01), roc_auc_score(y[ix], s0[ix]), average_precision_score(y[ix], s0[ix]), tpr_at(y[ix], s0[ix], .01))
bs = []
for _ in range(B):
    ix = np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))]); bs.append(met(ix))
bs = np.array(bs); full = met(np.arange(len(y))); ci = lambda k: np.percentile(bs[:, k], [2.5, 97.5])
dl = bs[:, :3] - bs[:, 3:]
pred = c >= th; tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum(); fn = (~pred & (y == 1)).sum(); tn = (~pred & (y == 0)).sum()
prec, rec = tp/(tp+fp), tp/(tp+fn); f1 = 2*prec*rec/(prec+rec); fpr = fp/(fp+tn)
mcc = (tp*tn-fp*fn)/np.sqrt(float(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
lvl = t.rfqr_risk_level.values
# importance
from train_rf_thesis import make_rf
import os; os.makedirs("artifacts/rfqr_report", exist_ok=True)
sub = t.sample(30000, random_state=42)
pi = permutation_importance(art["model"], sub[F].astype("float32").values, sub.Label.values, scoring="average_precision", n_repeats=3, random_state=42, n_jobs=1)
imp = pd.DataFrame({"Feature": F, "Group": ["URL"]*19+["QR"]*12, "dPR-AUC": pi.importances_mean, "SD": pi.importances_std}).sort_values("dPR-AUC", ascending=False)
# figures
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for sc, lab in ((s, "RF URL+QR (31)"), (s0, "RF URL only (19)")):
    a, b_, _ = roc_curve(y, sc); ax[0].plot(a, b_, label=lab); p, r, _ = precision_recall_curve(y, sc); ax[1].plot(r, p, label=lab)
ax[0].set_xlim(0, .05); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].set_title("ROC (FPR<=5%)"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title("Precision-Recall"); ax[0].legend(); ax[1].legend()
plt.tight_layout(); plt.savefig("artifacts/rfqr_report/fig1.png", dpi=150); plt.close()
top = imp.head(15)[::-1]; plt.figure(figsize=(7, 5)); plt.barh(top.Feature, top["dPR-AUC"], color=["tab:red" if g == "QR" else "tab:blue" for g in top.Group]); plt.xlabel("dPR-AUC (permutation)"); plt.title("Top 15 (blue=URL, red=QR)"); plt.tight_layout(); plt.savefig("artifacts/rfqr_report/fig2.png", dpi=150); plt.close()
# docx
D = Document(); D.styles["Normal"].font.name = "Times New Roman"; D.styles["Normal"].font.size = Pt(11)
P = lambda x: D.add_paragraph(x); Bu = lambda x: D.add_paragraph(x, style="List Bullet")
def T(df, cap):
    P(cap).runs[0].bold = True; tb = D.add_table(rows=1, cols=len(df.columns)); tb.style = "Light Grid Accent 1"
    for i, h in enumerate(df.columns): tb.rows[0].cells[i].text = str(h)
    for _, r in df.iterrows():
        cs = tb.add_row().cells
        for i, v in enumerate(r): cs[i].text = f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)
    for row in tb.rows:
        for cl in row.cells:
            for p in cl.paragraphs:
                for r in p.runs: r.font.size = Pt(9)
D.add_heading("Báo cáo Random Forest trên đặc trưng URL + QR (31 đặc trưng)", 0)
D.add_heading("1. Tóm tắt", 1)
Bu(f"Dữ liệu: Dataset_Hybrid.csv, 650.770 URL; train 520.616 / test 130.154, không trùng registered domain. Mô hình dùng 19 đặc trưng URL + 12 đặc trưng QR không hằng số (bỏ HasValidQR, QRImageAspectRatio).")
Bu("Giao thức: siêu tham số RF đã tối ưu cho 19 đặc trưng (273 cây, max_depth 36, max_features 0,3, min_samples_leaf 3, min_samples_split 3), không tune lại; 5-fold CV nhóm theo domain (seed 42); hiệu chỉnh isotonic trên OOF; ngưỡng đóng băng; test dùng một lần.")
Bu(f"Kết quả test: ROC-AUC {full[0]:.4f}, PR-AUC {full[1]:.4f}, TPR@1%FPR {full[2]:.4f}. Tại τ_high={th:.4f}: precision {prec:.4f}, recall {rec:.4f}, F1 {f1:.4f}, FPR {fpr:.4f}.")
Bu(f"So với RF chỉ URL: ΔPR-AUC {dl[:,1].mean():+.4f} [{np.percentile(dl[:,1],2.5):+.4f}, {np.percentile(dl[:,1],97.5):+.4f}], ΔTPR@1%FPR {dl[:,2].mean():+.4f} [{np.percentile(dl[:,2],2.5):+.4f}, {np.percentile(dl[:,2],97.5):+.4f}] (bootstrap ghép cặp theo domain, {B} lần).")
D.add_heading("2. Kết quả trên tập test", 1)
T(pd.DataFrame([{"Metric": n, "URL+QR (31)": full[k], "CI 95% thấp": ci(k)[0], "CI 95% cao": ci(k)[1], "URL only (19)": full[k+3]} for n, k in (("ROC-AUC", 0), ("PR-AUC", 1), ("TPR@1%FPR", 2))]), "Bảng 1. Chỉ số xếp hạng (CI bootstrap theo cụm domain)")
T(pd.DataFrame([{"Metric": "Precision", "Value": prec}, {"Metric": "Recall", "Value": rec}, {"Metric": "F1", "Value": f1}, {"Metric": "MCC", "Value": mcc}, {"Metric": "FPR", "Value": fpr}, {"Metric": "ECE (hiệu chỉnh)", "Value": ece(y, c)}, {"Metric": "TP / FP / FN / TN", "Value": f"{tp} / {fp} / {fn} / {tn}"}, {"Metric": "τ_high / τ_low", "Value": f"{th:.4f} / {tl:.4f}"}]), "Bảng 2. Chỉ số tại ngưỡng τ_high đóng băng")
D.add_picture("artifacts/rfqr_report/fig1.png", width=Cm(16)); P("Hình 1. ROC và Precision-Recall: URL+QR so với URL only.")
D.add_heading("3. Phân luồng ba mức rủi ro", 1)
names = {0: "Safe", 1: "Suspicious", 2: "Malicious"}
ct = pd.crosstab(pd.Series(y).map({0: "Lành tính", 1: "Độc hại"}), pd.Series(lvl).map(names)).reindex(columns=["Safe", "Suspicious", "Malicious"]).reset_index().rename(columns={"row_0": "Nhãn thật", "index": "Nhãn thật"})
T(ct, "Bảng 3. Phân bố mức rủi ro trên test")
cov = ((lvl != 1).mean()); P(f"Độ phủ (Safe hoặc Malicious): {cov:.4f}. Mô hình URL only: {(ut.rf_risk_level.values != 1).mean():.4f}.")
D.add_heading("4. Tầm quan trọng đặc trưng", 1)
T(imp.head(15).reset_index(drop=True), "Bảng 4. Permutation importance (30.000 URL test, đo bằng giảm PR-AUC)")
D.add_picture("artifacts/rfqr_report/fig2.png", width=Cm(12)); P("Hình 2. Top 15 đặc trưng.")
qrs = imp[imp.Group == "QR"].head(3)
P("Trong nhóm QR, các cột quan trọng nhất: " + ", ".join(f"{r.Feature} ({r['dPR-AUC']:.4f})" for _, r in qrs.iterrows()) + ". Các cột QRVersion, QRDimension, QRTotalModules, QRBlack/WhiteModules, QRPayloadLength, QRCapacityRatio tương quan 0,88–0,98 với độ dài URL (E8), nên chúng chủ yếu mang cùng một thông tin: độ dài tuyệt đối của URL.")
D.add_heading("5. Nhận xét và hạn chế", 1)
Bu("URL+QR cao hơn URL only một mức nhỏ nhưng có ý nghĩa; mức tăng gần như hoàn toàn đến từ QRPayloadLength (= độ dài URL gốc), không phải từ hình ảnh mã QR.")
Bu("Ảnh QR trong dataset được sinh từ chính URL, nên QR không thêm thông tin ngoài URL; kết luận không áp dụng cho ảnh QR thật có nhiễu hoặc nền khác.")
Bu("Mô hình dựa vào độ dài tuyệt đối nên có thể nhạy với biến đổi làm đổi độ dài URL (thêm tham số tracking, đệm đường dẫn; xem E6).")
Bu("Siêu tham số không tune lại cho 31 đặc trưng; CI dùng 300 lần bootstrap (thấp hơn 2.000 của các báo cáo trước); permutation importance chỉ trên mẫu 30.000 URL.")
Bu("Chỉ chạy Random Forest; LightGBM/XGBoost chưa chạy với đặc trưng QR.")
D.add_heading("6. Tái lập", 1)
P("python export_rf_qr.py 18  → Dataset_Hybrid_RF_QR.csv và artifacts/rf_thesis/url_phishing_model_qr.joblib; python make_rfqr_report.py → RF_URL_QR_Report.docx. Dataset_Hybrid_RF.csv (RF chỉ URL) dùng để so sánh.")
D.save("RF_URL_QR_Report.docx"); print("saved", full, dl.mean(0))
