# Bảng Kiểm Định McNemar (LightGBM vs Decision Tree Baseline)

**Dữ liệu kiểm thử**: 130,154 mẫu test URL độc lập (domain-separated).  
**Ngưỡng quyết định**: $\tau_{high} = 0.8267$ (FPR $\le 1\%$).

| Phân loại | Số mẫu | Tỷ lệ (%) |
|---|---:|---:|
| Cả 2 mô hình cùng đoán ĐÚNG ($a$) | 86,692 | 66.61% |
| **LightGBM ĐÚNG, Baseline SAI ($b$)** | **31,355** | **24.09%** |
| Baseline ĐÚNG, LightGBM SAI ($c$) | 3,781 | 2.91% |
| Cả 2 mô hình cùng đoán SAI ($d$) | 8,326 | 6.40% |

### Kết quả kiểm định thống kê:
- **Chi-square ($\chi^2$)**: **21637.93** (với hiệu chỉnh liên tục continuity correction)
- **Degrees of freedom ($df$)**: 1
- **$p$-value**: **0.0000e+00** ($p < 0.0001$)
- **Kết luận**: Tỷ lệ dự đoán đúng vượt trội của LightGBM so với mô hình baseline có ý nghĩa thống kê ở mức độ tin cậy $> 99.99\%$.
