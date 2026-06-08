# Tài liệu dữ liệu: LUNG_PATHOLOGY_PDL1_GLCM_V3

## Tổng quan

| Thuộc tính | Giá trị |
|---|---|
| File discovery | `lung_pathology_pdl1_glcm_v3.parquet` |
| File validation | `lung_pathology_pdl1_glcm_v3_validation.parquet` |
| Loại dữ liệu | Giải phẫu bệnh lý (Pathology) – đặc trưng kết cấu hình ảnh mô học nhuộm PD-L1 |
| Index | `main_index` |
| Số bệnh nhân (discovery) | 105 |
| Số bệnh nhân (validation) | 52 |
| Số features | 150 |
| Kiểu dữ liệu | float64 (toàn bộ) |
| Giá trị thiếu | Không có (0 NaN, 0 Inf) |

## Mô tả nguồn dữ liệu

File này chứa các đặc trưng texture (kết cấu) được trích xuất từ ảnh mô học nhuộm PD-L1 (Programmed Death-Ligand 1) của bệnh nhân ung thư phổi. Đặc trưng được tính theo phương pháp GLCM (Gray-Level Co-occurrence Matrix) – ma trận đồng xuất hiện mức xám – là kỹ thuật phổ biến trong phân tích radiomics/pathomics.

- **PD-L1**: Dấu ấn sinh học miễn dịch, thường dùng để đánh giá khả năng đáp ứng với liệu pháp miễn dịch (immunotherapy)
- **GLCM**: Mô tả sự phân bố không gian của cường độ pixel trong vùng quan tâm (ROI) của tiêu bản mô học
- **Channel 1**: Kênh màu thứ nhất của ảnh hình ảnh (RGB/HSV – tùy pipeline xử lý)

## Phân vùng dữ liệu

| Tập dữ liệu | Số bệnh nhân | Cohort | Định dạng index |
|---|---|---|---|
| Discovery | 105 | `discovery` (xác nhận qua `final_cohort_listing.csv`) | `P-XXXXXXX` (MSK patient ID) |
| Validation | 52 | Không overlap với discovery | Số nguyên (ví dụ: `1012132`) – có thể là cohort ngoại viện |

> **Lưu ý:** Index của tập validation có định dạng khác (số nguyên thuần túy, không có prefix `P-`), cần xử lý riêng khi join với các bảng khác.

## Cấu trúc feature

### Quy tắc đặt tên

```
{image_type}_{feature_name}_channel_{c}_{aggregation}
```

| Thành phần | Giá trị | Mô tả |
|---|---|---|
| `image_type` | `original` | Ảnh gốc không qua biến đổi bộ lọc |
| `feature_name` | `pixels` hoặc `glcm_<Tên>` | Loại đặc trưng |
| `channel_c` | `channel_1` | Kênh màu (chỉ dùng kênh 1) |
| `aggregation` | Xem bảng bên dưới | Phương pháp tổng hợp theo bệnh nhân |

### Phương pháp tổng hợp (aggregation)

Mỗi đặc trưng được tổng hợp trên nhiều patch/tile từ cùng bệnh nhân bằng 6 phương pháp:

| Hậu tố | Ý nghĩa | Số cột |
|---|---|---|
| `mean` | Trung bình của phân bố | 25 |
| `variance` | Phương sai của phân bố | 25 |
| `skewness` | Độ lệch (skewness) của phân bố | 25 |
| `kurtosis` | Độ nhọn (kurtosis) của phân bố | 25 |
| `lognorm_fit_p0` | Tham số σ (shape) của fit lognormal | 25 |
| `lognorm_fit_p2` | Tham số loc (location) của fit lognormal | 25 |

**Tổng cộng: 25 đặc trưng × 6 aggregation = 150 cột**

### Phân loại đặc trưng

| Nhóm | Số đặc trưng gốc | Cột tương ứng |
|---|---|---|
| Pixel intensity (`pixels`) | 1 | 6 cột |
| GLCM texture | 24 | 144 cột |

### 24 đặc trưng GLCM

| STT | Tên đặc trưng | Ý nghĩa ngắn gọn |
|---|---|---|
| 1 | `Autocorrelation` | Tương quan tự động – đo tính lặp lại của texture |
| 2 | `ClusterProminence` | Độ nổi bật của cụm pixel |
| 3 | `ClusterShade` | Bất đối xứng của cụm pixel |
| 4 | `ClusterTendency` | Xu hướng tạo cụm |
| 5 | `Contrast` | Tương phản cục bộ giữa pixel lân cận |
| 6 | `Correlation` | Tương quan tuyến tính giữa pixel lân cận |
| 7 | `DifferenceAverage` | Trung bình hiệu giữa pixel lân cận |
| 8 | `DifferenceEntropy` | Entropy của phân bố hiệu |
| 9 | `DifferenceVariance` | Phương sai của phân bố hiệu |
| 10 | `Id` (Inverse Difference) | Đồng nhất cục bộ |
| 11 | `Idm` (Inverse Difference Moment) | Trọng số khoảng cách – đo sự đồng đều |
| 12 | `Idmn` (Normalized) | Phiên bản chuẩn hóa của Idm |
| 13 | `Idn` (Normalized Id) | Phiên bản chuẩn hóa của Id |
| 14 | `Imc1` (Info Measure Corr 1) | Thông tin tương quan, thước đo phi tuyến |
| 15 | `Imc2` (Info Measure Corr 2) | Thông tin tương quan, thước đo phi tuyến thứ 2 |
| 16 | `InverseVariance` | Nghịch đảo phương sai khoảng cách |
| 17 | `JointAverage` | Trung bình cấp độ xám chung |
| 18 | `JointEnergy` | Năng lượng/đồng nhất toàn cục (Angular Second Moment) |
| 19 | `JointEntropy` | Entropy toàn cục của GLCM |
| 20 | `MCC` (Maximal Correlation Coefficient) | Hệ số tương quan cực đại |
| 21 | `MaximumProbability` | Xác suất lớn nhất trong GLCM |
| 22 | `SumAverage` | Trung bình tổng pixel lân cận |
| 23 | `SumEntropy` | Entropy của phân bố tổng |
| 24 | `SumSquares` | Phương sai (sum of squares) |

## Thống kê mô tả (Discovery set, n=105)

### Pixel intensity

| Feature | Mean | Std | Min | Max |
|---|---|---|---|---|
| `pixels_mean` | 49.22 | 25.77 | 8.37 | 121.04 |
| `pixels_variance` | 1903.31 | 1529.27 | 97.18 | 5924.85 |
| `pixels_skewness` | 1.60 | 0.85 | −0.02 | 4.76 |
| `pixels_kurtosis` | 4.37 | 5.93 | −1.33 | 38.80 |

### Một số GLCM tiêu biểu (mean aggregation)

| Feature | Mean | Std | Range |
|---|---|---|---|
| `Autocorrelation_mean` | 21.27 | 18.71 | 1.50 – 82.08 |
| `Contrast_mean` | 1.39 | 0.74 | 0.35 – 3.11 |
| `Correlation_mean` | 0.194 | 0.184 | −0.003 – 0.867 |
| `Id_mean` | 0.745 | 0.092 | 0.610 – 0.977 |
| `Idmn_mean` | 0.994 | 0.002 | 0.988 – 0.999 |
| `JointEnergy_mean` | 0.446 | 0.156 | 0.256 – 0.932 |
| `JointEntropy_mean` | 1.630 | 0.519 | 0.176 – 2.313 |
| `MCC_mean` | 0.555 | 0.112 | 0.311 – 0.731 |
| `MaximumProbability_mean` | 0.519 | 0.147 | 0.334 – 0.947 |

## Quan sát và lưu ý quan trọng

### 1. Tương quan cao giữa pixel và GLCM
Các feature `SumAverage` và `JointAverage` (GLCM) có tương quan cực cao với `pixels_mean` (r ≈ 0.998), vì chúng đều đo cường độ trung bình của pixel theo cách khác nhau. Cần kiểm tra đa cộng tuyến trước khi dùng tất cả các feature cùng lúc.

### 2. Lognorm_fit_p0 bằng 0 cho một số feature
Các feature sau có `lognorm_fit_p0 = 0` và `lognorm_fit_p2 = 0` cho **tất cả** bệnh nhân — fit lognormal thất bại do giá trị âm hoặc zero trong dữ liệu:
- `pixels` (p0 = 0)
- `ClusterShade` (có giá trị âm)
- `Correlation` (có giá trị âm/zero)
- `DifferenceEntropy`
- `Imc1` (luôn âm)
- `JointEntropy`
- `SumEntropy`

Các cột này **không dùng được** với aggregation `lognorm_fit_p0/p2` và nên được loại bỏ hoặc xử lý trước.

### 3. Scale rất khác nhau giữa các feature
- `ClusterProminence_kurtosis` có thể đạt ~63,000
- `Idmn_mean` nằm trong khoảng 0.987 – 0.999
- **Bắt buộc** dùng `RobustScaler` hoặc chuẩn hóa trước khi đưa vào mô hình.

### 4. Phân bố lệch cao (high skewness)
Nhiều feature GLCM như `ClusterProminence`, `ClusterTendency`, `DifferenceVariance`, `SumSquares` có skewness rất cao (> 5), cho thấy phân bố đuôi dài. Cân nhắc log-transform nếu dùng với mô hình tuyến tính.

## Tích hợp với pipeline huấn luyện

### Cách nạp vào `modality_dict`

```python
import pandas as pd
from lung_helpers import prepare_other_modalities

df_pathology = pd.read_parquet(f"{BASE_DB_DIR}/lung_pathology_pdl1_glcm_v3.parquet")

prepare_other_modalities(
    modality_dict, df_pathology, modality_MASK,
    name='pathology_pdl1_glcm'
)
```

### Lưu ý về scaling
File này **không phải NLP embedding**, nên **KHÔNG** cần `no_scale`. `RobustScaler` mặc định được áp dụng — điều này phù hợp vì dữ liệu có nhiều outlier.

### Gợi ý L1 filter

```python
l1_filter = {
    idx_pathology: {
        'l1_selection_df': df_pathology,
        'kwargs': {
            'robustness_cutoff': 0.15,
            'outlier_cutoff': 6
        }
    }
}
```

Cân nhắc đặt `outlier_cutoff` cao hơn (ví dụ 8–10) cho các feature kurtosis vì chúng có phân bố đuôi cực dài.

## File liên quan

| File | Mô tả |
|---|---|
| `pdl1_score.parquet` | Điểm PD-L1 thô (Sauter PD-L1 Score, n=201, 13 mức: 0–95%) |
| `new_lung_glcm_autocorrelation_v2_20x_stain1_pdl1.parquet` | Phiên bản cũ hơn, chỉ có Autocorrelation, 18 cột, dùng tên cột khác (`pixel_original_glcm_...scale_None...`) |
| `lung_pathology_pdl1_glcm_v3_validation.parquet` | Tập validation tương ứng (52 bệnh nhân, cùng 150 cột) |

## Phân bố điểm PD-L1 (`pdl1_score.parquet`)

| Mức (%) | Số bệnh nhân |
|---|---|
| 0 | 91 |
| 1 | 6 |
| 5 | 14 |
| 10 | 11 |
| 20 | 3 |
| 30 | 6 |
| 50 | 11 |
| 60 | 9 |
| 70 | 6 |
| 75 | 1 |
| 80 | 20 |
| 90 | 10 |
| 95 | 13 |
| **Tổng** | **201** |

> PD-L1 score là biến liên tục rời rạc (0–95%). Nếu dùng làm nhãn phân loại, cần chọn ngưỡng cutoff (thường 1%, 50%, hoặc theo guideline lâm sàng).
