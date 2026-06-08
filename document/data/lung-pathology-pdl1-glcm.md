# Lung Pathology PD-L1 GLCM – Phân tích dữ liệu và ý nghĩa y tế

**File (discovery):** `../datasets/new_lung_glcm_autocorrelation_v2_20x_stain1_pdl1.parquet`  
**File (validation):** `../datasets/new_lung_glcm_autocorrelation_v2_20x_stain1_pdl1_validation.parquet`  
**Phiên bản:** v2 | **Độ phóng đại:** 20x | **Nhuộm:** stain1 (PD-L1 IHC)

---

## 1. Tổng quan cấu trúc

| Thuộc tính | Discovery | Validation |
|---|---|---|
| Số bệnh nhân | 105 | 52 |
| Số cột (features) | 18 | 18 |
| Null values | 0 | 0 |
| Cột giống nhau | Có (100%) | |
| Cohort | Discovery only | Validation only |
| Patient overlap | Không trùng nhau | |

---

## 2. Bối cảnh y tế: Giải phẫu bệnh kỹ thuật số (Digital Pathology)

### 2.1 Đây là loại dữ liệu gì?

File này KHÔNG phải radiomics từ CT scan. Đây là **đặc trưng texture từ tiêu bản giải phẫu bệnh kỹ thuật số** (Whole Slide Image – WSI) của mô phổi được nhuộm hóa mô miễn dịch (IHC) với kháng thể **PD-L1**.

```
Sinh thiết mô phổi
       │
       ▼
Cắt mỏng + nhuộm PD-L1 IHC (kháng thể gắn enzyme)
       │
       ▼
Vùng PD-L1 dương tính → nhuộm nâu (DAB)
Vùng PD-L1 âm tính   → nhuộm xanh (hematoxylin)
       │
       ▼
Quét toàn bộ tiêu bản (WSI) ở 20x → hàng triệu pixel
       │
       ▼
Trích xuất đặc trưng texture GLCM trên từng pixel → file này
```

### 2.2 Tại sao dùng 20x?

**20x** (20 lần phóng đại) là mức độ phóng đại tiêu chuẩn trong giải phẫu bệnh lâm sàng – đủ để nhìn thấy rõ tế bào riêng lẻ và kiến trúc mô. Ở 20x, mỗi pixel WSI tương đương ~0.5 µm thực tế → một tiêu bản tiêu chuẩn có thể có hàng trăm triệu đến hàng tỷ pixel.

**nobs (số pixel):** trung bình ~55 triệu pixel/bệnh nhân, max ~1.25 tỷ pixel → xác nhận đây là dữ liệu WSI toàn tiêu bản.

### 2.3 Channel_1 là gì?

Sau khi quét WSI (ảnh RGB), ảnh được **phân tách màu (color deconvolution)** để tách riêng tín hiệu từng chất nhuộm:
- **Channel 1 (DAB channel):** cường độ nhuộm PD-L1 – pixel càng sáng → nồng độ PD-L1 protein càng cao
- Channel 2 (Hematoxylin): nhân tế bào

File này chỉ phân tích **channel 1** (DAB) → phản ánh trực tiếp mức độ biểu hiện PD-L1 ở cấp độ pixel.

---

## 3. Cấu trúc tên cột

```
pixel_{transform}_{texture}_scale_{scale}_channel_{ch}_{stat}
```

| Phần | Giá trị | Ý nghĩa |
|---|---|---|
| `pixel` | pixel | Phân tích cấp độ pixel (không phải region) |
| `transform` | `original` | Không biến đổi ảnh, dùng giá trị pixel thô |
| `texture` | `glcm_Autocorrelation` hoặc `glcm_None` | Loại đặc trưng texture |
| `scale` | `None` | Không dùng multi-scale |
| `channel` | `1` | DAB channel (PD-L1 staining) |
| `stat` | 9 loại | Thống kê mô tả phân phối |

---

## 4. Hai nhóm đặc trưng chính

### Nhóm A: `glcm_Autocorrelation` (9 features)

Tính **GLCM Autocorrelation** tại mỗi pixel → rồi lấy thống kê phân phối của tất cả giá trị Autocorrelation trên toàn tiêu bản.

**GLCM Autocorrelation** đo mức độ tương quan của cường độ DAB (PD-L1) giữa một pixel và các pixel lân cận. Cao → các tế bào PD-L1 dương tính có xu hướng **tụ thành cụm** (spatial clustering); thấp → tế bào dương tính phân tán rải rác.

**Ý nghĩa lâm sàng:** Pattern phân bố không gian của PD-L1 quan trọng hơn chỉ mỗi tỷ lệ phần trăm:
- PD-L1 tụ cụm tại giao diện mô bướu–lymphocyte → tumor microenvironment "nóng" (inflamed) → đáp ứng ICI tốt hơn
- PD-L1 rải rác không theo pattern → immune-excluded hoặc immune-desert phenotype → ít đáp ứng ICI

### Nhóm B: `glcm_None` (9 features)

Lấy thống kê trực tiếp từ **cường độ pixel DAB thô** (không qua texture transform). Đây là phân phối mật độ PD-L1 trên toàn bộ diện tích tiêu bản.

**Ý nghĩa lâm sàng:** Bổ sung cho Sauter PD-L1 Score (TPS thủ công) bằng cách:
- Không giới hạn ở vùng bác sĩ chọn → phân tích toàn tiêu bản
- Cho biết không chỉ "bao nhiêu % tế bào dương tính" mà còn "cường độ nhuộm phân phối như thế nào"

---

## 5. Ý nghĩa từng thống kê (9 stats, áp dụng cho cả 2 nhóm)

| Stat | Ý nghĩa y tế |
|---|---|
| `nobs` | Tổng số pixel hợp lệ được phân tích. Phụ thuộc kích thước mảnh sinh thiết. Trung bình ~55M, max ~1.25B pixel. |
| `min` | Giá trị nhỏ nhất. Luôn = 1 (do chuẩn hóa) → không mang thông tin. |
| `max` | Giá trị lớn nhất (DAB channel: 0–255). Gần 256 → có điểm nhuộm PD-L1 đậm cực mạnh. |
| `mean` | **Quan trọng nhất.** Cường độ PD-L1 trung bình toàn tiêu bản. Tương quan với Sauter TPS nhưng khách quan hơn và không bị ceiling effect. Range: 1.6–86.8 (Autocorrelation), 9.2–129.0 (Raw). |
| `variance` | Độ phân tán cường độ nhuộm. Cao → PD-L1 không đồng đều giữa các vùng mô (spatial heterogeneity). Liên quan đến tumor microenvironment phức tạp. |
| `skewness` | Độ lệch phân phối pixel. Dương (>0) → phần lớn pixel có cường độ thấp, chỉ một số điểm rất cao → phù hợp với PD-L1 âm tính đại thể nhưng có một số vùng dương tính cục bộ. |
| `kurtosis` | Độ nhọn phân phối. Cao → phân phối "đuôi nặng" (heavy-tailed) → có các điểm nhuộm cực mạnh rải rác. Liên quan đến dị biệt không gian PD-L1. |
| `lognorm_fit_p0` | Tham số σ (shape) của phân phối lognormal khớp vào histogram pixel. Cao → phân phối lognormal rộng → biểu hiện PD-L1 không đồng nhất. |
| `lognorm_fit_p2` | Tham số scale của lognormal (xấp xỉ median). Liên quan đến mức độ biểu hiện PD-L1 trung bình theo phân phối lognormal. |

**Tại sao dùng lognormal fit?** Cường độ DAB pixel trên tiêu bản IHC thường tuân theo phân phối lognormal (phần lớn pixel tối màu/âm tính, đuôi dài về phía sáng/dương tính). Fit lognormal cho phép tóm tắt phân phối bằng chỉ 2 tham số (p0, p2) có ý nghĩa thống kê rõ ràng.

---

## 6. Thống kê mô tả

### Nhóm Autocorrelation

| Stat | Mean | Std | Min | Max |
|---|---|---|---|---|
| nobs | 55.5M | 147.9M | 355K | 1.25B |
| mean | 22.27 | 19.26 | 1.61 | 86.77 |
| variance | 1276.7 | 1510.6 | 10.4 | 6212.7 |
| skewness | 3.90 | 2.44 | 0.54 | 16.28 |
| kurtosis | 31.69 | 50.29 | -0.91 | 409.6 |
| lognorm_fit_p0 (σ) | 1.30 | 0.27 | 0.52 | 1.93 |
| lognorm_fit_p2 (scale) | 9.06 | 7.59 | 1.22 | 40.83 |

### Nhóm Raw Pixel Intensity

| Stat | Mean | Std | Min | Max |
|---|---|---|---|---|
| nobs | 51.0M | 140.6M | 306K | 1.22B |
| mean | 52.46 | 26.42 | 9.19 | 129.03 |
| variance | 2029.6 | 1554.5 | 102.6 | 6117.9 |
| skewness | 1.47 | 0.80 | -0.15 | 4.65 |
| kurtosis | 3.52 | 5.19 | -1.36 | 37.98 |
| lognorm_fit_p0 (σ) | 0.97 | 0.22 | 0.00 | 1.39 |
| lognorm_fit_p2 (scale) | 35.17 | 18.33 | 6.20 | 97.21 |

---

## 7. So sánh với Sauter PD-L1 Score

| Đặc điểm | Sauter PD-L1 Score | File này |
|---|---|---|
| Phương pháp | Bác sĩ đếm thủ công | Tự động toàn tiêu bản |
| Đơn vị | % tế bào dương tính (TPS) | Cường độ pixel + texture |
| Thông tin không gian | Không | Có (GLCM Autocorrelation) |
| Phân phối | Bimodal (0% hoặc >50%) | Liên tục, phong phú hơn |
| Bệnh nhân có | 201 | 105 (discovery) |
| Hạn chế | Subjective, semi-quantitative | Phụ thuộc chất lượng scan, color deconvolution |

Hai nguồn dữ liệu này **bổ sung cho nhau**:
- Sauter score: consensus lâm sàng, được FDA công nhận
- File này: khách quan hơn, nắm bắt spatial heterogeneity mà TPS bỏ qua

---

## 8. Cách tích hợp vào pipeline

```python
df_pdl1_glcm = pd.read_parquet(f"{BASE_DB_DIR}/new_lung_glcm_autocorrelation_v2_20x_stain1_pdl1.parquet")
prepare_other_modalities(modality_dict, df_pdl1_glcm, modality_MASK, name='pdl1_glcm')
```

**Quan trọng:** Cần truyền `no_scale=[]` (mặc định) – các features này là giá trị số thô, RobustScaler áp dụng bình thường.

Lưu ý khi kết hợp với Sauter score: chỉ có **105/201** bệnh nhân có cả hai nguồn → cần `get_training_data()` để align theo intersection của index.
