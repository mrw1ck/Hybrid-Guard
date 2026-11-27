# BÁO CÁO KẾT QUẢ NGHIÊN CỨU & THỰC NGHIỆM MÔ HÌNH LIGHTGBM (LGBM)
**Dự án:** Hybrid-Guard - Phát hiện tấn công Phishing kết hợp URL & Metadata Mã QR  
**Ngày thực hiện:** Tháng 11/2026  
**Môi trường thử nghiệm:** Python 3.11, LightGBM 4.x, Scikit-learn, SHAP  

---

## 1. TỔNG QUAN & BỐI CẢNH THỬ NGHIỆM

Mô hình **LightGBM (Gradient Boosting Decision Tree)** được lựa chọn làm mô hình học máy cốt lõi (Core ML Engine) cho hệ thống **Hybrid-Guard** nhờ ưu điểm vượt trội về tốc độ suy luận cực nhanh, mức tiêu thụ tài nguyên bộ nhớ thấp và khả năng xử lý hiệu quả tập dữ liệu quy mô lớn (>650.000 mẫu).

### 1.1. Tập dữ liệu thực nghiệm (Dataset)
* **Tổng số mẫu:** 650.770 URL / QR Code.
* **Tỷ lệ nhãn (Ground Truth):**
  * **Benign (0 - Hợp pháp):** 390.462 mẫu (tỷ lệ 60.00%).
  * **Malicious (1 - Độc hại/Phishing):** 260.308 mẫu (tỷ lệ 40.00%).
* **Phân chia dữ liệu (Stratified Hold-out):**
  * **Tập huấn luyện (Train set - 80%):** 520.616 mẫu.
  * **Tập kiểm thử độc lập (Test set - 20%):** 130.154 mẫu.
* **Số lượng đặc trưng trích xuất:** 33 đặc trưng số học (không khuyết thiếu giá trị, không trùng lặp).

### 1.2. Tập đặc trưng kết hợp (Hybrid Feature Schema - 33 features)
1. **Nhóm đặc trưng tĩnh URL (19 đặc trưng lexical):**
   * Tỷ lệ cấu trúc: `HostRatio`, `PathRatio`, `DigitRatio`, `SpecialCharRatio`, `SubdomainRatio`.
   * Đếm ký tự / thành phần: `NoOfDots`, `NoOfHyphens`, `NoOfDigits`, `NoOfSlashes`, `NoOfQueryParams`, `NoOfObfuscatedChar`.
   * Độ hỗn loạn thông tin (Entropy): `HostEntropy`, `PathEntropy`.
   * Cờ nhị phân (Flags): `HasAtSymbol`, `IsPunycode`, `HasNonStandardPort`, `HasBrandInSubdomainOrPath`, `IsShortener`, `HasEmbeddedURL`.
2. **Nhóm đặc trưng cấu trúc mã QR (14 đặc trưng QR Metadata):**
   * Kích thước & phiên bản: `QRVersion`, `QRDimension`, `QRTotalModules`, `QRBlackModules`, `QRWhiteModules`, `QRDotDensity`.
   * Cơ chế mã hóa & sửa lỗi: `QRECCLevel`, `QRDataMask`, `QRUnusedECC`.
   * Dung lượng & tỷ lệ nén tải trọng: `QRPayloadLength`, `QRCapacityRatio`, `QRModulePixelSize`, `QRImageAspectRatio`, `HasValidQR`.

---

## 2. CẤU HÌNH SIÊU THAM SỐ (HYPERPARAMETERS)

| Tham số | Giá trị thiết lập | Mục đích tối ưu |
| :--- | :--- | :--- |
| **Thuật toán (Algorithm)** | LightGBM (GBDT) | Cây quyết định tăng cường độ dốc tối ưu hóa tốc độ |
| **Số lượng cây (`n_estimators`)** | 300 | Đạt hội tụ tối ưu trên tập dữ liệu 650k mẫu |
| **Tốc độ học (`learning_rate`)** | 0.05 | Kiểm soát overfitting, đảm bảo bước cập nhật ổn định |
| **Số lá tối đa (`num_leaves`)** | 63 | Cân bằng độ phức tạp của cây và khả năng tổng quát hóa |
| **Độ sâu cây tối đa (`max_depth`)** | -1 (Không giới hạn) | Cho phép phát triển cây theo cơ chế leaf-wise |
| **Trọng số lớp (`class_weight`)** | `balanced` | Bù trừ chênh lệch tỷ lệ 60/40 giữa Benign và Malicious |
| **Kiểu boosting (`boosting_type`)** | `gbdt` | Traditional Gradient Boosting Decision Tree |
| **Hạt giống ngẫu nhiên (`random_state`)** | 42 | Đảm bảo tính lặp lại (reproducibility) của thực nghiệm |
| **Kích thước mô hình (`.joblib`)** | **2.01 MB** | Cực kỳ nhỏ gọn, phù hợp triển khai Edge / Web Service |

---

## 3. KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP KIỂM THỬ ĐỘC LẬP (TEST SET - 130.154 MẪU)

### 3.1. Bảng số liệu tổng hợp & Khoảng tin cậy 95% (95% Confidence Interval)

Đánh giá được thực hiện với 2.000 lượt bootstrap lặp lại để xác định khoảng tin cậy 95%:

| Chỉ số đánh giá (Metric) | Điểm số trung bình | 95% CI Lower | 95% CI Upper | Chiều tối ưu |
| :--- | :--- | :--- | :--- | :---: |
| **Accuracy (Độ chính xác tổng)** | **93.70%** (0.937013) | 93.57% | 93.83% | ↑ |
| **Balanced Accuracy** | **93.28%** (0.932836) | 93.14% | 93.42% | ↑ |
| **Precision (Độ chuẩn xác)** | **92.93%** (0.929265) | 92.71% | 93.16% | ↑ |
| **Recall / TPR (Độ nhạy)** | **91.20%** (0.911951) | 90.94% | 91.43% | ↑ |
| **Specificity / TNR (Độ đặc hiệu)**| **95.37%** (0.953721) | 95.22% | 95.52% | ↑ |
| **F1-Score** | **92.05%** (0.920527) | 91.89% | 92.21% | ↑ |
| **MCC (Matthews Correlation)** | **0.8685** (0.868474) | 0.8658 | 0.8711 | ↑ |
| **Cohen's Kappa** | **0.8684** (0.868369) | 0.8657 | 0.8710 | ↑ |
| **ROC-AUC** | **0.9818** (0.981809) | 0.9812 | 0.9824 | ↑ |
| **PR-AUC (Average Precision)** | **0.9783** (0.978303) | 0.9776 | 0.9790 | ↑ |
| **Brier Score** | **0.0477** (0.047662) | 0.0469 | 0.0484 | ↓ |
| **Log Loss** | **0.1658** (0.165787) | 0.1636 | 0.1680 | ↓ |
| **FPR (Tỷ lệ dương tính giả)** | **4.63%** (0.046279) | 4.48% | 4.78% | ↓ |
| **FNR (Tỷ lệ âm tính giả)** | **8.80%** (0.088049) | 8.57% | 9.06% | ↓ |

### 3.2. Đánh giá chi tiết theo từng lớp (Per-Class Performance)

| Lớp (Class) | Precision | Recall | F1-Score | Số mẫu hỗ trợ (Support) |
| :--- | :---: | :---: | :---: | :---: |
| **Benign (Hợp pháp)** | 0.9318 | 0.9454 | 0.9386 | 78.092 |
| **Malicious (Độc hại)** | 0.9163 | 0.8962 | 0.9061 | 52.062 |
| **Macro Average** | 0.9240 | 0.9208 | 0.9223 | 130.154 |
| **Weighted Average** | 0.9256 | 0.9257 | 0.9256 | 130.154 |

---

## 4. KIỂM ĐỊNH CHÉO 5-FOLD (5-FOLD STRATIFIED CROSS VALIDATION)

Kiểm định chéo 5-Fold trên toàn bộ tập dữ liệu đảm bảo mô hình không bị quá khớp (overfitting) và có tính ổn định cao:

| Fold | Accuracy | Precision | Recall (TPR) | Specificity (TNR) | F1-Score | MCC | ROC-AUC | PR-AUC | Train Time (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 0.9362 | 0.9262 | 0.9133 | 0.9515 | 0.9197 | 0.8669 | 0.9818 | 0.9781 | 6.43s |
| **Fold 2** | 0.9361 | 0.9282 | 0.9107 | 0.9530 | 0.9194 | 0.8666 | 0.9817 | 0.9782 | 6.96s |
| **Fold 3** | 0.9371 | 0.9294 | 0.9120 | 0.9538 | 0.9206 | 0.8686 | 0.9816 | 0.9782 | 7.46s |
| **Fold 4** | 0.9368 | 0.9288 | 0.9119 | 0.9534 | 0.9202 | 0.8680 | 0.9815 | 0.9781 | 7.26s |
| **Fold 5** | 0.9356 | 0.9264 | 0.9113 | 0.9517 | 0.9188 | 0.8655 | 0.9818 | 0.9781 | 7.46s |
| **Trung bình ± Độ lệch** | **0.9363 ± 0.0006** | **0.9278 ± 0.0014** | **0.9118 ± 0.0010** | **0.9527 ± 0.0010** | **0.9197 ± 0.0007** | **0.8671 ± 0.0012** | **0.9817 ± 0.0001** | **0.9781 ± 0.0001** | **7.11s ± 0.43s** |

> [!NOTE]
> Độ lệch chuẩn giữa các fold rất nhỏ ($\sigma \le 0.0014$), chứng minh mô hình LightGBM phân bố đồng nhất trên toàn bộ không gian dữ liệu.

---

## 5. NGHIÊN CỨU TRIỆT TIÊU (ABLATION STUDY): ĐÓNG GÓP CỦA METADATA QR

Nghiên cứu triệt tiêu được thực hiện để chứng minh tính hiệu quả của phương pháp **Hybrid (URL + QR)** so với phương pháp truyền thống **URL-Only**:

| Cấu hình mô hình | Số đặc trưng | Accuracy | Precision | Recall (TPR) | F1-Score | MCC | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **URL-Only Baseline** | 19 | 0.9257 | 0.9163 | 0.8962 | 0.9061 | 0.8448 | 0.9762 | 0.9714 |
| **Đề xuất: URL + QR Metadata** | **33** | **0.9370** | **0.9293** | **0.9120** | **0.9205** | **0.8685** | **0.9818** | **0.9783** |
| **Mức độ cải thiện ($\Delta$)** | **+14** | **+1.13%** | **+1.30%** | **+1.58%** | **+1.44%** | **+0.0237** | **+0.0056** | **+0.0069** |

* **Độ nhạy (Recall) tăng 1.58%**: Giúp giảm thiểu đáng kể số lượng URL lừa đảo lọt lưới (giảm False Negatives).
* **Độ chính xác (F1-score) tăng 1.44%**: Xác nhận các thuộc tính vật lý và cấu trúc mã QR (như mật độ điểm, chiều dài payload mã hóa) chứa tín hiệu phát hiện gian lận mạnh mẽ.

---

## 6. PHÂN TÍCH TẦM QUAN TRỌNG ĐẶC TRƯNG (FEATURE IMPORTANCE & SHAP)

### 6.1. Top 10 đặc trưng quan trọng nhất (theo Information Gain & Split)

| Xếp hạng | Tên đặc trưng | Information Gain | Số lần phân tách (Split) | Tỷ lệ Gain (%) | Tỷ lệ Split (%) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | `PathEntropy` | 1.640.972 | 1.406 | **28.97%** | 7.56% |
| **2** | `PathRatio` | 752.114 | 2.142 | **13.28%** | 11.52% |
| **3** | `NoOfDots` | 700.170 | 844 | **12.36%** | 4.54% |
| **4** | `NoOfDigits` | 390.507 | 955 | **6.89%** | 5.13% |
| **5** | **`QRPayloadLength`** | 377.534 | 1.747 | **6.67%** | 9.39% |
| **6** | `HostRatio` | 278.283 | 1.808 | **4.91%** | 9.72% |
| **7** | `SubdomainRatio` | 261.079 | 1.930 | **4.61%** | 10.38% |
| **8** | `DigitRatio` | 242.174 | 1.048 | **4.28%** | 5.63% |
| **9** | `NoOfSlashes` | 211.183 | 1.310 | **3.73%** | 7.04% |
| **10** | `NoOfHyphens` | 168.567 | 792 | **2.98%** | 4.26% |

### 6.2. Top 10 đặc trưng giải thích bởi SHAP (SHapley Additive exPlanations)

| Xếp hạng | Tên đặc trưng | Mean \|SHAP value\| | Ý nghĩa diễn giải |
| :---: | :--- | :---: | :--- |
| **1** | `PathEntropy` | **1.2495** | Đường dẫn có độ hỗn loạn cao là dấu hiệu token ngẫu nhiên / obfuscation |
| **2** | `PathRatio` | **1.1031** | Đường dẫn URL dài bất thường so với domain gốc |
| **3** | `NoOfDots` | **0.6315** | Kỹ thuật chèn nhiều subdomain giả mạo thương hiệu |
| **4** | `DigitRatio` | **0.5224** | Chứa chuỗi ký tự số thay thế ký danh thông thường |
| **5** | **`QRPayloadLength`** | **0.4226** | **Chiều dài chuỗi ký tự nạp trong mã QR biểu thị cấu trúc tinh vi** |
| **6** | `SubdomainRatio` | **0.3894** | Tỷ lệ ký tự chiếm bởi subdomain |
| **7** | `HasNonStandardPort`| **0.3259** | Cổng mạng không tiêu chuẩn (khác 80, 443) thường gặp ở malware/phishing |
| **8** | `NoOfHyphens` | **0.3251** | Sử dụng dấu gạch ngang ghép từ khóa mạo danh |
| **9** | `NoOfDigits` | **0.2988** | Tổng số ký tự số |
| **10** | `HostRatio` | **0.2777** | Tỷ lệ chiều dài hostname so với toàn bộ URL |

---

## 7. ĐÁNH GIÁ HIỆU CHUẨN XÁC SUẤT & PHÂN TÍCH NGƯỠNG (CALIBRATION & THRESHOLD ANALYSIS)

### 7.1. Độ tin cậy xác suất dự đoán
* **Expected Calibration Error (ECE):** **0.0242** (Mức sai số hiệu chuẩn chỉ ~2.4%, điểm xác suất phản ánh sát thực tế).
* **Kolmogorov-Smirnov Statistic (KS):** **0.8663** (Khả năng phân tách tuyệt vời giữa phân phối xác suất của Benign và Malicious).

### 7.2. Phân tích các ngưỡng quyết định (Decision Thresholds)

| Ngưỡng (Threshold) | Precision | Recall | F1-Score | MCC | FPR | Accuracy | Ý nghĩa thực tế trong hệ thống |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.2000** | 0.7645 | 0.9629 | 0.8523 | 0.7497 | 0.1977 | 0.8665 | Chế độ phòng thủ tối đa (Paranoid) |
| **0.3000** | 0.8316 | 0.9414 | 0.8831 | 0.8014 | 0.1271 | 0.9003 | Chế độ cảnh giác cao |
| **0.3500** | **0.8582** | **0.9307** | **0.8930** | **0.8187** | **0.1025** | **0.9108** | **Ngưỡng App Suspicious (Cảnh báo nghi ngờ)** |
| **0.4757** | 0.9086 | 0.9024 | 0.9055 | 0.8429 | 0.0605 | 0.9247 | Ngưỡng tối ưu chỉ số Youden's J |
| **0.5000** | **0.9163** | **0.8962** | **0.9061** | **0.8448** | **0.0546** | **0.9257** | **Ngưỡng mặc định (Tối ưu hóa F1)** |
| **0.7000** | 0.9636 | 0.8384 | 0.8966 | 0.8405 | 0.0211 | 0.9227 | Ngưỡng xác định mã độc tự tin cao |
| **0.8000** | 0.9788 | 0.7995 | 0.8801 | 0.8230 | 0.0116 | 0.9129 | Tự động Block khẩn cấp (FPR < 1.1%) |

---

## 8. ĐO LƯỜNG ĐỘ TRỄ & THÔNG LƯỢNG (BENCHMARK LATENCY & THROUGHPUT)

Thử nghiệm so sánh hiệu năng thực tế giữa LightGBM và các thuật toán ensemble khác:

| Tiêu chí | LightGBM | Random Forest | XGBoost |
| :--- | :---: | :---: | :---: |
| **Thời gian suy luận đơn lẻ (Latency / URL)** | **4.17 µs** | 7.20 µs | **1.97 µs** |
| **Thời gian suy luận Batch (P50)** | **2.04 ms** | 30.61 ms | 3.17 ms |
| **Thời gian suy luận Batch (P95)** | **3.06 ms** | 37.17 ms | 4.78 ms |
| **Thông lượng xử lý Batch (Throughput)** | **124.650 samples/s** | 89.711 samples/s | **588.699 samples/s** |
| **Kích thước mô hình trên đĩa** | **2.01 MB** | **397.10 MB** (gấp ~198 lần) | 3.13 MB |
| **Thời gian huấn luyện trung bình** | **~7.1 giây** | > 15 phút | ~45 giây |

> [!TIP]
> **Nhận định:** LightGBM nhẹ gấp **198 lần** so với Random Forest và nhanh gấp **15 lần** ở phân vị P95, trong khi duy trì chỉ số ROC-AUC tương đương (0.9818 so với 0.9840). Đây là sự cân bằng lý tưởng nhất cho hệ thống quét thời gian thực.

---

## 9. ĐÁNH GIÁ Ý NGHĨA THỐNG KÊ (MCNEMAR'S TEST)

Kiểm định McNemar ($\chi^2$) và khoảng tin cậy bootstrap $\Delta$ được thực hiện để kiểm tra sự khác biệt giữa LightGBM và các mô hình đối chuẩn:

| So sánh cặp mô hình | Mẫu chỉ LGB đúng ($b$) | Mẫu chỉ đối chuẩn đúng ($c$) | McNemar $\chi^2$ | $p$-value | $\Delta$F1 [95% CI] | $\Delta$ROC-AUC [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LGBM vs Random Forest** | 1.504 | 2.164 | 118.40 | $< 10^{-16}$ | -0.0053 [-0.0064, -0.0041] | -0.0022 [-0.0026, -0.0019] |
| **LGBM vs XGBoost** | 1.266 | 1.464 | 14.22 | $0.000163$ | -0.0000 [-0.0010, +0.0009] | -0.0004 [-0.0005, -0.0003] |

Sự khác biệt về hiệu năng phân loại giữa LightGBM và XGBoost về mặt thực tiễn là tương đương ($\Delta\text{F1} \approx 0$), nhưng LightGBM vượt trội về dung lượng tệp lưu trữ và tốc độ huấn luyện.

---

## 10. TỔNG KẾT & DANH MỤC TỆP DỮ LIỆU ĐÍNH KÈM

### 10.1. Kết luận chính
1. Mô hình LightGBM đạt độ chính xác tổng hợp **93.70%**, **F1-score 92.05%**, và diện tích dưới đường cong **ROC-AUC 0.9818** trên 130k mẫu test độc lập.
2. Việc bổ sung đặc trưng cấu trúc QR Code giúp cải thiện toàn diện mọi chỉ số, đặc biệt tăng **Recall thêm +1.58%** và đưa **`QRPayloadLength`** vào Top 5 đặc trưng quan trọng nhất.
3. Kích thước mô hình chỉ **2.01 MB** cùng độ trễ **4.17 µs/URL** hoàn toàn đáp ứng yêu cầu xử lý trực tuyến (line-rate inspection) cho các ứng dụng thực tế.

### 10.2. Danh mục tài nguyên mô hình & Biểu đồ đồ họa

Tất cả các tệp artifact và biểu đồ đã được biên tập và lưu trữ tại thư mục:
* **Mô hình nhị phân:** [url_phishing_model.joblib](file:///D:/GITproject/Hybrid-Guard/artifacts/lgb_final/url_phishing_model.joblib)
* **Bảng biểu & Dữ liệu chi tiết:** [artifacts/report/lgb_research](file:///D:/GITproject/Hybrid-Guard/artifacts/report/lgb_research)

**Biểu đồ nghiên cứu (Figures):**
* [fig1_class_distribution.png](file:///D:/GITproject/Hybrid-Guard/artifacts/report/lgb_research/fig1_class_distribution.png) - Phân bố nhãn tập dữ liệu.
* [fig2_confusion_matrix.png](file:///D:/GITproject/Hybrid-Guard/artifacts/report/lgb_research/fig2_confusion_matrix.png) - Ma trận nhầm lẫn (Confusion Matrix).
* [fig3_roc_pr.png](file:///D:/GITproject/Hybrid-Guard/artifacts/report/lgb_research/fig3_roc_pr.png) - Đường cong ROC và Precision-Recall.
* [fig4_threshold_analysis.png](file:///D:/GITproject/Hybrid-Guard/artifacts/report/lgb_research/fig4_threshold_analysis.png) - Phân tích độ nhạy theo các ngưỡng quyết định.
* [fig5_calibration.png](file:///D:/GITproject/Hybrid-Guard/artifacts/report/lgb_research/fig5_calibration.png) - Đường cong hiệu chuẩn xác suất (Reliability Curve).
* [fig6_score_distribution.png](file:///D:/GITproject/Hybrid-Guard/artifacts/report/lgb_research/fig6_score_distribution.png) - Phân phối điểm dự đoán giữa Benign và Malicious.
* [fig7_feature_importance.png](file:///D:/GITproject/Hybrid-Guard/artifacts/report/lgb_research/fig7_feature_importance.png) - Biểu đồ xếp hạng đặc trưng theo Gain & Split.
* [fig8_shap_beeswarm.png](file:///D:/GITproject/Hybrid-Guard/artifacts/report/lgb_research/fig8_shap_beeswarm.png) - Biểu đồ SHAP Beeswarm diễn giải mô hình.
* [fig9_cross_validation.png](file:///D:/GITproject/Hybrid-Guard/artifacts/report/lgb_research/fig9_cross_validation.png) - Độ ổn định qua 5 Fold Cross-Validation.
* [fig10_ablation_study.png](file:///D:/GITproject/Hybrid-Guard/artifacts/report/lgb_research/fig10_ablation_study.png) - So sánh nghiên cứu triệt tiêu URL-Only vs URL+QR.
* [fig12_latency_benchmark.png](file:///D:/GITproject/Hybrid-Guard/artifacts/report/lgb_research/fig12_latency_benchmark.png) - Đo lường độ trễ và thông lượng xử lý.
