import pandas as pd, numpy as np
from docx import Document
from docx.shared import Cm, Pt
D = Document(); D.styles["Normal"].font.name = "Times New Roman"; D.styles["Normal"].font.size = Pt(11)
def H(t, l=1): D.add_heading(t, l)
def P(t): return D.add_paragraph(t)
def B(t): D.add_paragraph(t, style="List Bullet")
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
def F(path, cap, w=16):
    D.add_picture(path, width=Cm(w)); P(cap)
M = ["LightGBM", "XGBoost", "Random Forest"]
D.add_heading("Hybrid-Guard – Báo cáo thực nghiệm E2, E6, E4", 0)
P("Ba mô hình (LightGBM, XGBoost, Random Forest) dùng cùng 19 đặc trưng, cùng fold, seed 42, 50 trial Optuna không pruner, hiệu chỉnh xác suất và đánh giá một lần trên tập test khóa (130.154 URL). Các số liệu dưới đây lấy từ artifacts/e2_ablation, e6_robustness, e4_hybrid.")

# ---------------- E2
H("1. Ablation nhóm đặc trưng (E2)")
P("Thiết kế: 19 đặc trưng chia thành 5 nhóm. Với mỗi mô hình, giữ nguyên siêu tham số đã tối ưu trên bộ đầy đủ (không tune lại cho từng tập con), huấn luyện lại trên tập train rồi đo trên test. Có hai chế độ: bỏ một nhóm (leave-one-group-out) và chỉ giữ một nhóm. Khoảng tin cậy 95% của chênh lệch lấy từ bootstrap ghép cặp theo cụm domain (1.000 lần, cùng trọng số cho mọi cấu hình).")
for g in ["G1 Length ratios: HostRatio, PathRatio, SubdomainRatio", "G2 Structure counts: NoOfDots, NoOfHyphens, NoOfSlashes, NoOfQueryParams",
          "G3 Character composition: NoOfDigits, DigitRatio, SpecialCharRatio, NoOfObfuscatedChar", "G4 Entropy: HostEntropy, PathEntropy",
          "G5 Phishing indicators: HasAtSymbol, IsPunycode, HasNonStandardPort, HasBrandInSubdomainOrPath, IsShortener, HasEmbeddedURL"]: B(g)
d = pd.read_csv("artifacts/e2_ablation/E2_ablation.csv")
pr = d[d.Metric == "PR-AUC"]
t = pr.pivot(index="Configuration", columns="Model", values="Value")[M].reset_index()
order = ["Full model (19 features)"] + [c for c in pr.Configuration.unique() if c.startswith("Drop")] + [c for c in pr.Configuration.unique() if c.startswith("Only")]
t = t.set_index("Configuration").loc[order].reset_index(); T(t, "Bảng E2-1. PR-AUC trên test theo cấu hình đặc trưng")
x = pr[pr.Configuration.str.startswith("Drop")].copy(); x["ΔPR-AUC [95% CI]"] = x.apply(lambda r: f"{r['Delta vs full']:+.4f} [{r['Delta CI lower']:+.4f}, {r['Delta CI upper']:+.4f}]", axis=1)
t2 = x.pivot(index="Configuration", columns="Model", values="ΔPR-AUC [95% CI]")[M].reset_index(); T(t2, "Bảng E2-2. Thay đổi PR-AUC khi bỏ một nhóm (CI bootstrap theo domain)")
tp = d[d.Metric == "TPR@1%FPR"].pivot(index="Configuration", columns="Model", values="Value")[M].loc[order].reset_index(); T(tp, "Bảng E2-3. TPR@1%FPR trên test theo cấu hình")
F("artifacts/e2_ablation/figE2_ablation.png", "Hình E2-1. Trái: ΔPR-AUC khi bỏ từng nhóm (thanh sai số = CI 95%). Phải: PR-AUC của mô hình chỉ dùng một nhóm.", 17)
H("Giải thích E2", 2)
B("Cả ba mô hình cho cùng thứ tự quan trọng: G2 (đếm cấu trúc) mất nhiều nhất khi bị bỏ (ΔPR-AUC khoảng −0,022 đến −0,025, CI không chứa 0 ở mọi mô hình). Tiếp theo là G1 và G3 (khoảng −0,009 đến −0,014), rồi G5 (−0,004) và G4 (−0,001 đến −0,006).")
B("Mức giảm khi bỏ một nhóm đều nhỏ (dưới 0,03). Điều này cho thấy các nhóm thừa thông tin cho nhau: ví dụ bỏ G1 hoặc G3 thì với LightGBM và XGBoost CI vẫn chạm 0, nghĩa là chưa đủ bằng chứng là chúng không thể thay thế nhau.")
B("Ngược lại, mô hình chỉ có một nhóm kém hơn hẳn (PR-AUC giảm từ 0,07 ở G1 đến 0,39 ở G5). Không nhóm nào tự đủ; sức mạnh đến từ việc kết hợp. G5 (cờ nhị phân) đứng riêng thì yếu nhất vì phần lớn URL có toàn giá trị 0 ở các cờ hiếm.")
B("Thứ tự nhóm nhất quán giữa ba họ mô hình nên kết luận không phụ thuộc thuật toán. Hạn chế: siêu tham số không được tune lại, nên mức giảm của tập con có thể bị đánh giá hơi cao.")

# ---------------- E6
H("2. Kiểm tra robust: biến đổi URL và vòng QR (E6)")
P("Thiết kế: lấy mẫu phân tầng 20.000 URL từ tập test (10.000 mỗi lớp). Mỗi URL được biến đổi theo các phép không đổi ý nghĩa, trích lại 19 đặc trưng bằng bộ trích xuất viết lại (khớp Dataset_Hybrid.csv: 15/19 đặc trưng khớp 100%, SpecialCharRatio 99,95%, IsShortener 99,99%, HasBrandInSubdomainOrPath 99,71%, HasEmbeddedURL 99,69%), rồi chấm lại bằng mô hình đóng băng. Dòng 'identity' kiểm tra tính nhất quán của pipeline (kỳ vọng thay đổi bằng 0).")
e = pd.read_csv("artifacts/e6_robustness/E6_perturbation.csv")
v = e[e.Model == "LightGBM"][["Perturbation", "% URLs changed", "ROC-AUC", "PR-AUC", "TPR@1%FPR", "dTPR@1%FPR", "mean |dp|", "flip rate @0.5 (%)"]]
T(v, "Bảng E6-1. LightGBM: tác động của biến đổi URL")
for m in ["XGBoost", "Random Forest"]:
    v = e[e.Model == m][["Perturbation", "ROC-AUC", "PR-AUC", "TPR@1%FPR", "dTPR@1%FPR", "mean |dp|", "flip rate @0.5 (%)"]]; T(v, f"Bảng E6-2 ({m}). Tác động của biến đổi URL")
for m, tag in [("LightGBM", "lgb"), ("XGBoost", "xgb"), ("Random Forest", "rf")]:
    F(f"artifacts/e6_robustness/figE6_perturbation_{tag}.png", f"Hình E6-1{tag[0]}. {m}: thay đổi TPR@1%FPR, ROC-AUC và tỷ lệ đổi nhãn theo từng biến đổi.", 17)
F("artifacts/e6_robustness/figE6_perturbation.png", "Hình E6-1. Thay đổi TPR@1%FPR (trái) và tỷ lệ đổi nhãn ở p=0,5 (phải) theo từng biến đổi.", 17)
q = pd.read_csv("artifacts/e6_robustness/E6_qr.csv"); q = q[q.Model == "LightGBM"][["Degradation", "n", "decode success (%)", "mean |dp| (decoded)", "flip rate @0.5 (decoded, %)"]]
T(q, "Bảng E6-3. Vòng QR (sinh mã, làm suy giảm ảnh, giải mã bằng OpenCV, chấm điểm URL giải mã)", "{:.3f}")
F("artifacts/e6_robustness/figE6_qr.png", "Hình E6-2. Tỷ lệ giải mã đúng theo kiểu suy giảm ảnh.", 11)
H("Giải thích E6", 2)
B("So sánh ba mô hình (Hình E6-1a/b/c): thứ tự biến đổi gây hại gần như giống nhau ở cả ba. Khác biệt đáng chú ý duy nhất là 'thêm tham số tracking': Random Forest giảm TPR@1%FPR 0,50 và 22,0% URL đổi nhãn, so với 0,24–0,30 và 15,5–15,7% ở LightGBM và XGBoost, tức RF nhạy hơn rõ rệt với số tham số truy vấn. Phép chèn đường dẫn lành tính làm cả ba mất 0,60–0,62 TPR@1%FPR, nên đây là điểm yếu của bộ đặc trưng, không phải của một thuật toán. Chưa có kiểm định thống kê cho các chênh lệch giữa mô hình ở E6 (mỗi mô hình chỉ chạy một lần trên một mẫu 20.000 URL).")
B("Kiểm tra tính nhất quán đạt: biến đổi 'identity' và viết hoa host cho thay đổi bằng 0 (đặc trưng tính trên host chữ thường), nên chênh lệch ở các dòng còn lại do biến đổi thật.")
B("Mô hình nhạy với các thay đổi nhỏ không đổi ý nghĩa. Thêm tham số tracking làm TPR@1%FPR của LightGBM giảm 0,24 và 15,5% URL đổi nhãn; thêm /index.html giảm 0,29. Nguyên nhân: nhiều đặc trưng là tỷ lệ hoặc số đếm trên chiều dài URL (HostRatio, PathRatio, DigitRatio, entropy), nên thêm ký tự lành tính làm đổi phân phối.")
B("Phép tấn công 'đệm đường dẫn lành tính' làm ROC-AUC rơi từ 0,963 xuống khoảng 0,66 và TPR@1%FPR giảm 0,62. Một kẻ tấn công biết đặc trưng có thể né mô hình chỉ bằng cách nối thêm đường dẫn. Đây là hạn chế chính cần nêu trong luận văn và là lý do cần lớp kiểm tra ngoài (E4).")
B("Các phép chỉ đổi scheme hoặc www (http→https, thêm www) ảnh hưởng vừa (giảm TPR@1%FPR 0,08–0,10). Điều này cho thấy mô hình dùng scheme/www như dấu hiệu của dữ liệu huấn luyện, không phải bản chất phishing.")
B("QR: khi giải mã được thì URL giống hệt bản gốc (|Δp| = 0, 0% đổi nhãn), nên QR không làm hỏng điểm số mà chỉ gây thất bại ở bước đọc. Tỷ lệ giải mã sạch 98,9%; nhiễu Gaussian 93,5%; xoay 15° chỉ 79,9% (bộ giải mã mặc định của OpenCV kém với góc xoay). Phần 1,1% thất bại ở ảnh sạch do URL quá dài hoặc có ký tự không phải ASCII.")
B("Giới hạn: biến đổi chọn thủ công và bộ trích xuất là bản viết lại (không phải bản gốc), nên số liệu là xấp xỉ.")

# ---------------- E4
H("3. Thí nghiệm hybrid (E4)")
P("Thiết kế: chạy toàn bộ pipeline của webapp (UrlAnalyzerService: DNS, SSL, tuổi tên miền RDAP, HTTP, chính sách rủi ro) với xác suất nền do từng mô hình thesis cung cấp. Hai tập dữ liệu: (a) 'stale': 1.500 URL độc hại và 1.500 URL lành tính lấy phân tầng từ tập test (dữ liệu lịch sử); (b) 'fresh': 300 URL phishing mới từ OpenPhish (toàn bộ feed lúc chạy) và 750 trang chủ của Tranco top-100k. Nội dung trang chỉ đọc tối đa 64 KB, không thực thi hay lưu. Kết quả phụ thuộc thời điểm chạy nên không tái lập chính xác.")
C = pd.read_csv("artifacts/e4_hybrid/E4_checks_by_label.csv"); T(C, "Bảng E4-1. Tỷ lệ kiểm tra ngoài có kết quả, theo nhãn", "{:.1f}")
O = pd.read_csv("artifacts/e4_hybrid/E4_hybrid.csv")
for s in ["stale", "fresh"]:
    x = O[O.Set == s][["Model", "System", "n", "ROC-AUC", "PR-AUC", "TPR@1%FPR", "Precision", "Recall", "F1", "FPR", "dPR-AUC CI", "dTPR@1%FPR CI"]]
    T(x, f"Bảng E4-2 ({s}). ML thuần so với hybrid")
F("artifacts/e4_hybrid/figE4_prauc.png", "Hình E4-1. PR-AUC của ML thuần và hybrid theo từng tập.", 15)
F("artifacts/e4_hybrid/figE4_checks.png", "Hình E4-2. Mức sẵn có của kiểm tra ngoài theo nhãn: đây là yếu tố gây nhiễu.", 11)
H("Giải thích E4", 2)
s = O[(O.Set == "stale") & (O.Model == "LightGBM")].set_index("System"); f = O[(O.Set == "fresh") & (O.Model == "LightGBM")].set_index("System")
B(f"Tập stale (LightGBM): hybrid nâng PR-AUC từ {s.loc['ML only','PR-AUC']:.3f} lên {s.loc['Hybrid (ML + policy + external checks)','PR-AUC']:.3f} (CI ΔPR-AUC {s.loc['Hybrid (ML + policy + external checks)','dPR-AUC CI']}, không chứa 0) và giảm FPR từ {s.loc['ML only','FPR']:.3f} xuống {s.loc['Hybrid (ML + policy + external checks)','FPR']:.3f}. Tuy nhiên TPR@1%FPR gần như không đổi (CI {s.loc['Hybrid (ML + policy + external checks)','dTPR@1%FPR CI']} chứa 0).")
B(f"Phần cải thiện này bị nhiễu chắc chắn: ở tập stale, chỉ {C.iloc[0,3]:.0f}% URL độc hại còn phân giải được DNS so với {C.iloc[1,3]:.0f}% URL lành tính; HTTP lần lượt {C.iloc[0,6]:.0f}% so với {C.iloc[1,6]:.0f}%. URL phishing cũ phần lớn đã chết, nên 'không có DNS/SSL' tự nó dự đoán nhãn. Hybrid đang khai thác tuổi dữ liệu chứ không chỉ chứng cứ phishing. Không nên coi đây là bằng chứng hybrid tốt hơn trong triển khai.")
B(f"Tập fresh (không có thiên lệch tuổi): kết quả đảo chiều. ML thuần gắn cờ {100*f.loc['ML only','FPR']:.0f}% trang chủ Tranco là độc hại (precision {f.loc['ML only','Precision']:.2f}), vì dữ liệu huấn luyện hầu như không có URL dạng 'https://domain/' lành tính, nên đây là lệch phân phối. Hybrid giảm FPR xuống {100*f.loc['Hybrid (ML + policy + external checks)','FPR']:.0f}% nhờ SSL và tuổi tên miền, nhưng PR-AUC và TPR@1%FPR giảm mạnh (TPR@1%FPR từ {f.loc['ML only','TPR@1%FPR']:.2f} xuống {f.loc['Hybrid (ML + policy + external checks)','TPR@1%FPR']:.2f}). Lớp điều chỉnh làm hỏng thứ hạng của điểm số, dù lọc được nhiều báo động giả ở ngưỡng cố định.")
B("Kết luận trung thực: lớp hybrid giúp giảm báo động giả ở ngưỡng quyết định, nhưng không chứng minh được cải thiện khả năng phân tách. Cần tập phishing còn sống và tập lành tính cùng dạng URL (có đường dẫn) để kết luận. Chỉ 300 URL phishing mới nên CI rộng.")
B("Rủi ro vận hành: truy cập trực tiếp URL độc hại từ IP của người dùng, kết quả đổi theo thời gian; nên chạy trong sandbox và lưu cache.")
H("4. Hạn chế chung", 1)
B("Bộ trích xuất 19 đặc trưng được dựng lại; HasBrandInSubdomainOrPath và HasEmbeddedURL khớp khoảng 99,7%, không tuyệt đối.")
B("Không gian tìm kiếm XGBoost và LightGBM dựng lại là suy đoán/không đồng nhất hoàn toàn với báo cáo gốc.")
B("E6 và E4 chỉ là xấp xỉ: biến đổi tự chọn, mẫu mạng phụ thuộc thời điểm.")
D.save("Experiments_E2_E4_E6_Report.docx"); print("saved")
