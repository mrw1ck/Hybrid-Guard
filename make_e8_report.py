import json, pandas as pd, numpy as np
from docx import Document
from docx.shared import Cm, Pt
D = Document(); D.styles["Normal"].font.name = "Times New Roman"; D.styles["Normal"].font.size = Pt(11)
P = lambda t: D.add_paragraph(t); B = lambda t: D.add_paragraph(t, style="List Bullet")
def T(df, cap, fmt="{:.4f}"):
    P(cap).runs[0].bold = True
    t = D.add_table(rows=1, cols=len(df.columns)); t.style = "Light Grid Accent 1"
    for i, c in enumerate(df.columns): t.rows[0].cells[i].text = str(c)
    for _, r in df.iterrows():
        cs = t.add_row().cells
        for i, v in enumerate(r): cs[i].text = "" if (isinstance(v, float) and v != v) else (fmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for run in p.runs: run.font.size = Pt(8)
def F(path, cap, w=16): D.add_picture(path, width=Cm(w)); P(cap)
A = "artifacts/e8_qr/"
m = pd.read_csv(A + "E8_metrics.csv"); dl = pd.read_csv(A + "E8_paired_delta.csv"); dg = pd.read_csv(A + "E8_qr_feature_diagnostics.csv")
im = pd.read_csv(A + "E8_importance_url_qr.csv"); rk = pd.read_csv(A + "E8_risk_levels.csv"); b = json.load(open(A + "E8b_url_plus_payload.json"))
order = ["URL only (19)", "QR only (12)", "URL + QR (31)", "QRPayloadLength only (1)", "URL-length-only tree (depth 2)", "Logistic regression (19 URL)"]
def cell(c, k):
    r = m[(m.Configuration == c) & (m.Metric == k)].iloc[0]
    return f"{r.Value:.4f}" + ("" if r["CI lower"] != r["CI lower"] else f" [{r['CI lower']:.4f}, {r['CI upper']:.4f}]")

D.add_heading("Thực nghiệm E8: đặc trưng QR so với đặc trưng URL (Random Forest)", 0)
P("Mục tiêu: trả lời hai câu hỏi của yêu cầu bổ sung. (1) Chỉ dùng đặc trưng QR thì phát hiện phishing được đến đâu? (2) Thêm đặc trưng QR vào 19 đặc trưng URL có cải thiện không? Chỉ huấn luyện Random Forest (mô hình 2). Số liệu lấy từ artifacts/e8_qr; script e8_qr_vs_url.py và e8b_len_check.py.")
D.add_heading("1. Thiết kế", 1)
B("Dữ liệu: Dataset_Hybrid.csv, 650.770 URL, cùng partition train/test như các báo cáo trước (520.616 / 130.154; không trùng registered domain). So với Dataset.csv, các cột cũ giống hệt từng dòng; chỉ thêm 14 cột QR.")
B("Đặc trưng QR: 14 cột trong file, nhưng HasValidQR và QRImageAspectRatio là hằng số (luôn bằng 1) nên bị loại; còn 12 cột QR dùng được.")
B("Mô hình: Random Forest với siêu tham số đã tối ưu cho 19 đặc trưng URL (273 cây, max_depth 36, max_features 0,3, min_samples_leaf 3), không tune lại cho từng tập đặc trưng (cùng quy ước E2). Do đó cấu hình chỉ-QR có thể bị đánh giá hơi thấp.")
B("Giao thức giữ nguyên Chương 3: 5-fold nhóm theo domain, seed 42, hiệu chỉnh isotonic trên OOF, ngưỡng τ_high/τ_low đóng băng, test dùng một lần; CI 95% bằng bootstrap theo cụm domain (2.000 lần); Δ so với 'URL only' dùng bootstrap ghép cặp cùng trọng số.")
B("Cấu hình: URL only (19), QR only (12), URL + QR (31). Thêm ba mốc tham chiếu: chỉ QRPayloadLength (1 cột), cây độ sâu 2 chỉ dùng độ dài URL (đã có ở báo cáo trước) và hồi quy logistic trên 19 đặc trưng URL (baseline đơn giản mà checklist REFORMS còn thiếu).")
D.add_heading("2. Kết quả", 1)
rows = []
for c in order:
    rows.append({"Configuration": c, "ROC-AUC": cell(c, "ROC-AUC"), "PR-AUC": cell(c, "PR-AUC"), "TPR@1%FPR": cell(c, "TPR@1%FPR"), "TPR@0.1%FPR": cell(c, "TPR@0.1%FPR")})
T(pd.DataFrame(rows), "Bảng E8-1. Chỉ số trên tập test (giá trị [CI 95% bootstrap theo domain])")
rows = []
for c in order:
    g = lambda k: m[(m.Configuration == c) & (m.Metric == k)].Value.iloc[0]
    rows.append({"Configuration": c, "Precision": g("Precision"), "Recall": g("Recall (TPR)"), "F1": g("F1-score"), "MCC": g("MCC"), "FPR": g("FPR"), "ECE": g("ECE (calibrated)"), "Coverage": g("Coverage (Safe+Malicious share)")})
T(pd.DataFrame(rows), "Bảng E8-2. Chỉ số tại ngưỡng τ_high đóng băng, hiệu chuẩn và độ phủ")
x = dl.copy(); x["Δ [95% CI]"] = x.apply(lambda r: f"{r['Delta vs URL only']:+.4f} [{r['CI lower']:+.4f}, {r['CI upper']:+.4f}]", axis=1)
t = x.pivot(index="Configuration", columns="Metric", values="Δ [95% CI]")[["ROC-AUC", "PR-AUC", "TPR@1%FPR", "TPR@0.1%FPR"]].loc[order[1:]].reset_index()
T(t, "Bảng E8-3. Chênh lệch so với 'URL only (19)' (bootstrap ghép cặp theo domain; mọi CI đều không chứa 0)")
F(A + "figE8_2_configs.png", "Hình E8-1. PR-AUC (trái) và TPR@1%FPR (phải) trên test của từng cấu hình, kèm CI 95%.", 17)
P("Cách đọc Hình E8-1: mỗi thanh là một cấu hình, thanh càng dài càng tốt, vạch đen là CI 95%. Thanh xanh dương (chỉ URL) và xanh lá (URL + QR) dài gần bằng nhau và vượt xa các thanh còn lại. Thanh đỏ (chỉ QR) ngắn: PR-AUC 0,600 và TPR@1%FPR chỉ 0,076, tức tại FPR 1% chỉ bắt được khoảng 7,6% URL độc hại; CI của PR-AUC rộng ([0,49; 0,69]) vì chỉ có khoảng 57.000 domain độc lập. Chỉ QR còn kém cây chỉ-độ-dài-URL (PR-AUC 0,652; TPR@1%FPR 0,265).")
F(A + "figE8_1_roc_pr.png", "Hình E8-2. Đường ROC (trái) và Precision-Recall (phải) trên test.", 17)
P("Cách đọc Hình E8-2: đường càng sát góc trên-trái (ROC) hoặc góc trên-phải (PR) càng tốt. Hai đường xanh (URL, URL + QR) gần như chồng nhau, URL + QR nằm cao hơn một chút ở vùng FPR thấp. Đường đỏ (QR) và tím (một cột QRPayloadLength) chồng nhau và nằm sát đường chéo ngẫu nhiên ở phần đầu đường cong, nghĩa là bắt được rất ít URL độc hại trước khi bắt đầu báo nhầm. Hồi quy logistic (nâu) nằm giữa: tốt hơn QR nhưng kém Random Forest rõ rệt, cho thấy quan hệ giữa các đặc trưng URL là phi tuyến.")
D.add_heading("3. Vì sao QR yếu và vì sao URL + QR lại tăng", 1)
d2 = dg[["Feature", "Mean benign", "Mean malicious", "Distinct values", "max(AUC,1-AUC)", "Spearman vs URL length"]]
T(d2, "Bảng E8-4. Chẩn đoán 14 cột QR trên tập train (AUC đơn lẻ và tương quan hạng với độ dài URL)")
F(A + "figE8_3_qr_features.png", "Hình E8-3. Trái: AUC đơn lẻ của từng đặc trưng QR. Phải: tương quan hạng (Spearman) của chúng với độ dài URL.", 17)
P("Cách đọc Hình E8-3: ở bảng trái, AUC đơn lẻ cao nhất chỉ khoảng 0,62 (0,5 là ngẫu nhiên), nên không cột QR nào tự phân biệt tốt. Ở bảng phải, bảy cột (QRVersion, QRDimension, QRTotalModules, QRBlack/WhiteModules, QRPayloadLength, QRCapacityRatio) có tương quan 0,88–0,98 với độ dài URL, tức chúng chủ yếu là cùng một thông tin (độ dài) viết theo cách khác. Các cột không phụ thuộc độ dài (QRDataMask, QRDotDensity, QRECCLevel, QRModulePixelSize, QRUnusedECC) có AUC gần 0,5, tức gần như không mang tín hiệu.")
F(A + "figE8_4_importance.png", "Hình E8-4. Mô hình URL + QR: mức giảm PR-AUC trên test khi hoán vị từng đặc trưng (top 15; xanh = URL, đỏ = QR).", 12)
P("Cách đọc Hình E8-4: đặc trưng nào làm PR-AUC giảm nhiều nhất khi bị xáo trộn thì mô hình dựa vào nó nhiều nhất. Phần lớn top đầu là đặc trưng URL (PathRatio, PathEntropy, NoOfSlashes, NoOfDots). Trong nhóm QR chỉ QRPayloadLength (giảm khoảng 0,009) và QRCapacityRatio (khoảng 0,003) đáng kể; phần còn lại gần như không đóng góp.")
P(f"Kiểm tra bổ sung E8b: chỉ thêm đúng một cột QRPayloadLength vào 19 đặc trưng URL cho PR-AUC {b['PR-AUC']['value']:.4f} [{b['PR-AUC']['lo']:.4f}, {b['PR-AUC']['hi']:.4f}] và TPR@1%FPR {b['TPR@1%FPR']['value']:.4f} [{b['TPR@1%FPR']['lo']:.4f}, {b['TPR@1%FPR']['hi']:.4f}], gần như bằng cả 12 cột QR (PR-AUC 0,9696; TPR@1%FPR 0,8174). Vậy gần như toàn bộ mức tăng đến từ một cột duy nhất. Cột này bằng đúng len(url_raw) ở 99,99% số dòng, tức là độ dài tuyệt đối của URL gốc. Mô hình URL only chỉ dùng tỷ lệ (HostRatio, PathRatio, ...) nên không có độ dài tuyệt đối. Điều này khớp với E7 (Bảng 3.5): URLLength, PathLength cũng mang thêm tín hiệu thật. Chưa tính CI ghép cặp riêng cho cấu hình E8b.")
D.add_heading("4. Kết luận E8", 1)
B("Đặc trưng QR đứng riêng không đủ phát hiện phishing: PR-AUC 0,600 và TPR@1%FPR 0,076, kém cả cây chỉ-độ-dài-URL (0,652; 0,265). Chênh lệch so với URL only là −0,36 PR-AUC và −0,71 TPR@1%FPR, CI không chứa 0.")
B("URL + QR cao hơn URL only một mức nhỏ nhưng có ý nghĩa thống kê: PR-AUC +0,0098 [+0,0060; +0,0153], TPR@1%FPR +0,035 [+0,026; +0,045], TPR@0,1%FPR +0,068 [+0,044; +0,104].")
B("Mức tăng đó không đến từ đặc điểm ảnh QR mà từ độ dài URL tuyệt đối (QRPayloadLength). Nếu cần cải thiện, thêm đặc trưng độ dài tuyệt đối vào bộ URL là đủ, không cần đường QR.")
B("Hồi quy logistic (PR-AUC 0,873; TPR@1%FPR 0,406) kém Random Forest rõ rệt (0,960; 0,782), nên Random Forest không phải là phức tạp thừa. Baseline này lấp một mục còn thiếu của REFORMS (Bảng P1, module 7).")
D.add_heading("5. Hạn chế", 1)
B("Ảnh QR trong dataset được sinh từ chính URL (QRPayloadLength khớp url_raw), chưa phải ảnh QR thu thập ngoài đời thực; vì vậy các cột QR chỉ là hàm của URL và không thể thêm thông tin ngoài URL. Kết luận không áp dụng cho ảnh QR thật có nhiễu hoặc nền khác.")
B("Siêu tham số không tune lại cho cấu hình QR hoặc URL + QR; kết quả của hai cấu hình này có thể được cải thiện nhẹ nếu tune riêng.")
B("Chỉ Random Forest được chạy; LightGBM và XGBoost chưa chạy cho E8. Các báo cáo trước cho thấy ba mô hình tương đương, nhưng chưa kiểm chứng riêng cho E8.")
B("Chưa kiểm tra độ bền của mô hình chứa QRPayloadLength trước các biến đổi ở E6 (ví dụ thêm tham số tracking làm tăng độ dài URL).")
D.save("QR_Features_E8_Report.docx"); print("saved")
