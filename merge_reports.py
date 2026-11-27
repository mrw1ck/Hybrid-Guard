from docx import Document
from docx.shared import Pt
from docxcompose.composer import Composer
cover = Document(); cover.styles["Normal"].font.name = "Times New Roman"; cover.styles["Normal"].font.size = Pt(11)
cover.add_heading("Hybrid-Guard: BÁO CÁO THỰC NGHIỆM TỔNG HỢP", 0)
cover.add_paragraph("Phát hiện URL phishing bằng đặc trưng tĩnh (19 đặc trưng URL, 650.770 URL; Dataset_Hybrid.csv có thêm 14 cột QR). Ba mô hình LightGBM, XGBoost và Random Forest được huấn luyện cùng giao thức Chương 3: 5-fold CV nhóm theo domain, seed 42, 50 trial Optuna không pruner, hiệu chỉnh isotonic, ngưỡng đóng băng, đánh giá một lần trên 130.154 URL test.")
cover.add_heading("Mục lục", 1)
parts = [("Phần A", "So sánh ba mô hình và kiểm định giả thuyết H1", "Model_Comparison_Report.docx"),
         ("Phần B", "Báo cáo chi tiết Mô hình 1: LightGBM (dựng lại)", "LightGBM_Rebuilt_Model1_Report.docx"),
         ("Phần C", "Báo cáo chi tiết Mô hình 2: Random Forest", "RandomForest_Model2_Report.docx"),
         ("Phần D", "Báo cáo chi tiết Mô hình 3: XGBoost", "XGBoost_Model3_Report.docx"),
         ("Phần E", "Thực nghiệm E2 (ablation), E6 (robust, QR), E4 (hybrid)", "Experiments_E2_E4_E6_Report.docx"),
         ("Phần F", "Bảng 3.5 (E7), cập nhật webapp, checklist REFORMS", "Items_8_9_10_Report.docx"),
         ("Phần G", "Thực nghiệm E8: đặc trưng QR so với đặc trưng URL (Random Forest)", "QR_Features_E8_Report.docx")]
for a, b, c in parts: cover.add_paragraph(f"{a}. {b}  ({c})", style="List Bullet")
cover.add_heading("Kết luận chính", 1)
for t in ["Ba mô hình tương đương trên tập test: ROC-AUC 0,9626–0,9634, PR-AUC 0,9578–0,9599, TPR@1%FPR 77,9–78,5%. Mọi khoảng tin cậy bootstrap ghép cặp theo domain của chênh lệch đều chứa 0.",
          "H1 (boosting tốt hơn RF) chỉ được hỗ trợ yếu ở cross-validation (hơn 5/5 fold, khoảng 0,002 PR-AUC, p Holm 0,007–0,010) và không được hỗ trợ trên test.",
          "Nhóm đặc trưng đếm cấu trúc (G2) quan trọng nhất ở cả ba mô hình (E2); bỏ một nhóm chỉ làm PR-AUC giảm dưới 0,03.",
          "Mô hình nhạy với biến đổi URL không đổi ý nghĩa (E6): thêm tham số tracking làm TPR@1%FPR giảm 0,24–0,50; chèn đường dẫn lành tính làm giảm khoảng 0,60 ở cả ba mô hình. Đọc mã QR thất bại ở khâu giải mã (79,9% khi xoay 15°) chứ không ở mô hình.",
          "Lớp hybrid (E4) giảm báo động giả ở ngưỡng cố định nhưng không chứng minh được cải thiện khả năng phân tách: trên dữ liệu cũ kết quả bị nhiễu bởi URL đã chết, trên dữ liệu mới TPR@1%FPR giảm.",
          "Hạn chế: 19 thay vì 25 đặc trưng của Bảng 3.5; LightGBM là bản dựng lại; không gian tìm kiếm XGBoost là giả định; bộ trích xuất đặc trưng được viết lại (khớp dataset 99,7–100%). Webapp đã chuyển sang mô hình thesis đã hiệu chỉnh với ngưỡng đóng băng; checklist REFORMS còn nhiều mục chỉ đạt một phần (Phần F). Với 25 đặc trưng PR-AUC cao hơn 0,009–0,013, phần lớn do HasHTTPS là artefact của dataset (E7)."]:
    cover.add_paragraph(t, style="List Bullet")
for t in ["E8 (Phần G, chỉ Random Forest): đặc trưng QR đứng riêng yếu (PR-AUC 0,600; TPR@1%FPR 0,076), kém cả cây chỉ-độ-dài-URL (0,652; 0,265). URL + QR (31 đặc trưng) cao hơn URL only một mức nhỏ nhưng có ý nghĩa (PR-AUC 0,9696 so với 0,9599; TPR@1%FPR 0,817 so với 0,782). Gần như toàn bộ mức tăng đến từ một cột là QRPayloadLength, bằng độ dài URL gốc; ảnh QR trong dataset được sinh từ URL nên không thêm thông tin ngoài URL. Hồi quy logistic trên 19 đặc trưng URL đạt PR-AUC 0,873, kém Random Forest rõ rệt.",
          "Dataset_Hybrid.csv giống Dataset.csv ở mọi cột cũ (so từng dòng trên 650.770 dòng) và chỉ thêm 14 cột QR, nên các mô hình và thí nghiệm E2–E7 không cần huấn luyện lại."]:
    cover.add_paragraph(t, style="List Bullet")
cover.add_page_break()
c = Composer(cover)
import os
for _, _, f in parts:
    f = f.replace('.docx','_new.docx') if os.path.exists(f.replace('.docx','_new.docx')) else f
    sep = Document(); sep.add_page_break(); c.append(sep); c.append(Document(f))
c.save("Hybrid_Guard_Full_Report.docx"); print("saved")
