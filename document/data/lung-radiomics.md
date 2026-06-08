# Lung Radiomics – Phân tích dữ liệu và ý nghĩa y tế

**File:** `../datasets/lung_radiomics_spacing1.0_mirpon_window1350.250_allimagetypes_bw20.parquet`  
**Tham số trích xuất:** spacing=1.0mm, MirpOn (mirror padding), Window=[1350, 250] HU, binWidth=20 HU

---

## 1. Tổng quan cấu trúc file

| Thuộc tính | Giá trị |
|---|---|
| Tổng số hàng (lesions) | 3.996 |
| Tổng số cột (features) | 1.690 |
| Bệnh nhân duy nhất | 187 |
| Trung bình lesion/bệnh nhân | ~21.4 hàng |
| Null values | 0 |
| Index | `main_index` (P-XXXXXXX) |

### Cột metadata (không phải feature)

| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| `job_tag` | string | Loại radiomics: `filtered-radiomics` (333), `unfiltered-radiomics` (333), `pertubation-radiomics` (3.330) |
| `lesion_index` | int | Chỉ số tổn thương: 1–6 (theo thứ tự kích thước giảm dần) |

**Về `job_tag`:**
- `filtered-radiomics`: đặc trưng đã qua lọc robustness (ICC ≥ 0.85 với perturbation) → dùng trong training
- `unfiltered-radiomics`: toàn bộ đặc trưng chưa lọc
- `pertubation-radiomics`: 10 lần nhiễu loạn ngẫu nhiên (rotation, dilation) → dùng để tính ICC

---

## 2. Radiomics là gì? Ý nghĩa y tế tổng quát

**Radiomics** là phương pháp trích xuất hàng trăm đến hàng ngàn đặc trưng định lượng từ ảnh y tế (CT, MRI, PET) theo cách mà mắt người không thể nhận thấy. Trong NSCLC:

- CT scan hiển thị khối u dưới dạng vùng mật độ cao (Hounsfield Units – HU)
- Mỗi khối u có **phenotype ảnh** (imaging phenotype) đặc trưng: hình dạng, độ đồng nhất mật độ, cấu trúc bề mặt, texture nội bộ
- Các đặc trưng này phản ánh **sinh học phân tử bên dưới**: tốc độ tăng sinh tế bào, mức độ hoại tử, tân sinh mạch máu (angiogenesis), xâm lấn mô lành

Nghiên cứu này trích xuất radiomics từ ảnh CT ngực với **window [1350, 250] HU** – tức là window center 250 HU, width 1350 HU, tối ưu cho mô mềm/khối u phổi.

---

## 3. Cấu trúc tên cột: `{image_type}_{feature_family}_{feature_name}`

Mỗi feature có tên gồm 3 phần:

```
original_glcm_Contrast
│         │    └── tên đặc trưng cụ thể
│         └── nhóm feature (texture matrix)
└── loại ảnh đầu vào
```

---

## 4. Các loại ảnh đầu vào (Image Types) – 18 loại

File áp dụng các phép biến đổi toán học lên ảnh CT gốc trước khi trích feature, nhằm làm nổi bật các đặc tính khác nhau của mô khối u.

### 4.1 `original` – Ảnh CT gốc

Ảnh CT nguyên bản dạng Hounsfield Unit (HU). Đây là nền tảng; tất cả loại ảnh khác đều được tạo từ đây.

**Ý nghĩa lâm sàng:** Phản ánh trực tiếp mật độ vật lý của mô. Mô hoại tử có HU thấp hơn mô ung thư sống; vùng vôi hóa có HU rất cao.

### 4.2 `exponential` – Biến đổi mũ (e^x)

Áp dụng hàm e^x lên từng voxel. Khuếch đại sự khác biệt ở vùng mật độ cao, nén sự khác biệt ở vùng mật độ thấp.

**Ý nghĩa lâm sàng:** Làm nổi bật ranh giới giữa mô ung thư đặc (solid) và mô lành xung quanh; nhạy cảm hơn với vùng tăng sinh mạnh.

### 4.3 `gradient` – Gradient không gian (độ dốc cường độ)

Tính magnitude của gradient ảnh 3D – phát hiện biên/cạnh. Vùng chuyển tiếp đột ngột có gradient cao.

**Ý nghĩa lâm sàng:** Đặc trưng **rìa khối u** (tumor margin). Khối u có rìa sắc nét (spiculated margin) thường xâm lấn hơn. NSCLC hung hãn thường có gradient cao tại biên do mô khối u xâm lấn không đều vào mô lành.

### 4.4 `lbp-2D` – Local Binary Pattern 2D

Mã hóa cấu trúc texture cục bộ bằng cách so sánh mỗi pixel với 8 pixel lân cận trong từng lát cắt axial.

**Ý nghĩa lâm sàng:** Phân tích **cấu trúc vi mô** của khối u trên từng lát cắt CT. Phát hiện các pattern lặp lại trong nội bộ khối u – liên quan đến mức độ dị biệt mô học (heterogeneity).

### 4.5 `lbp-3D-m1`, `lbp-3D-m2`, `lbp-3D-k` – Local Binary Pattern 3D

Ba biến thể mã hóa LBP trong không gian 3D. m1 và m2 là hai phương pháp khác nhau để chọn neighbor points trên mặt cầu; k là biến thể rotation-invariant.

**Ý nghĩa lâm sàng:** Mở rộng phân tích texture lên 3 chiều, phản ánh tốt hơn cấu trúc không gian thực của khối u trong ảnh CT volumetric.

### 4.6 `logarithm` – Biến đổi logarithm (log(x))

Áp dụng log lên cường độ voxel. Nén dải động, làm đồng đều sự phân bố intensity.

**Ý nghĩa lâm sàng:** Làm nổi bật cấu trúc ở vùng mật độ thấp (ví dụ: vùng hoại tử trong khối u, ground-glass opacity). Nhạy cảm với các vùng dịch lỏng hoặc thoái hóa trong khối u.

### 4.7 `square` – Bình phương (x²)

Khuếch đại mạnh vùng mật độ cao, bỏ qua vùng mật độ thấp.

**Ý nghĩa lâm sàng:** Làm nổi bật mô đặc (solid component) trong khối u hỗn hợp (part-solid nodule). Tương phản rõ giữa vùng đặc và vùng bán trong (semi-solid).

### 4.8 `squareroot` – Căn bậc hai (√x)

Nén dải động theo hướng ngược lại với square.

**Ý nghĩa lâm sàng:** Làm nhạy cảm hơn với biến thiên ở vùng mật độ thấp; hữu ích phân tích ground-glass component.

### 4.9 `wavelet-HHH/HHL/HLH/HLL/LHH/LHL/LLH/LLL` – Phân tích wavelet 3D (8 sub-bands)

Phân rã tín hiệu CT thành 8 sub-band tần số theo 3 trục (x, y, z). Mỗi chữ cái là H (High-pass) hoặc L (Low-pass) cho một trục:

| Sub-band | Trục X | Trục Y | Trục Z | Nắm bắt |
|---|---|---|---|---|
| LLL | Low | Low | Low | Cấu trúc thô, hình dạng tổng thể |
| LLH | Low | Low | High | Biên theo trục Z (axial) |
| LHL | Low | High | Low | Biên theo trục Y (coronal) |
| HLL | High | Low | Low | Biên theo trục X (sagittal) |
| HHH | High | High | High | Texture mịn, chi tiết cực nhỏ |

**Ý nghĩa lâm sàng:** Wavelet phân tích khối u ở **nhiều tỷ lệ không gian** đồng thời. Sub-band tần số cao (HHH, HHL...) phát hiện bề mặt không đều, spiculation, vi calcification. Sub-band tần số thấp (LLL) mô tả hình dạng và mật độ tổng thể.

---

## 5. Nhóm đặc trưng hình dạng – `shape` (14 features)

**Chỉ có trong `original`** vì shape là đặc trưng hình học thuần túy, không phụ thuộc intensity.

| Feature | Ý nghĩa y tế |
|---|---|
| `Elongation` | Tỷ lệ giữa trục ngắn và trục trung gian. Khối u elongated (dài) thường xâm lấn dọc theo cấu trúc giải phẫu (mạch máu, phế quản). |
| `Flatness` | Tỷ lệ trục ngắn nhất / trục dài nhất. Khối u dẹt (flat) gợi ý phát triển dọc màng phổi hoặc thành ngực. |
| `LeastAxisLength` | Chiều dài trục nhỏ nhất (mm). |
| `MajorAxisLength` | Chiều dài trục lớn nhất (mm). Tương quan với T-stage trong TNM. |
| `MinorAxisLength` | Chiều dài trục trung gian (mm). |
| `Maximum2DDiameterColumn` | Đường kính lớn nhất trên mặt phẳng axial theo chiều cột. |
| `Maximum2DDiameterRow` | Đường kính lớn nhất trên axial theo chiều hàng. |
| `Maximum2DDiameterSlice` | Đường kính lớn nhất trên bất kỳ lát cắt 2D nào. Đây là số liệu bác sĩ dùng đo trên PACS (RECIST criteria). |
| `Maximum3DDiameter` | Đường kính 3D lớn nhất. Chính xác hơn 2D cho khối u bất đối xứng. |
| `MeshVolume` | Thể tích khối u (mm³) tính từ mesh bề mặt. Tương quan mạnh với gánh nặng khối u (tumor burden). |
| `VoxelVolume` | Thể tích tính bằng đếm voxel. Tương tự MeshVolume. |
| `SurfaceArea` | Diện tích bề mặt (mm²). Khối u bề mặt lớn (spiculated, lobulated) có SurfaceArea cao. |
| `SurfaceVolumeRatio` | Tỷ lệ bề mặt/thể tích. Cao → bề mặt phức tạp, không cầu → khối u xâm lấn. Thấp → hình cầu → thường lành tính hơn. |
| `Sphericity` | Độ cầu (0–1). Tiệm cận 1 = hình cầu hoàn hảo. NSCLC thường có Sphericity thấp do phát triển không đồng đều. |

**Ý nghĩa lâm sàng tổng hợp:** Shape features tương quan với T-stage, khả năng cắt bỏ phẫu thuật, và khả năng xâm lấn mạch bạch huyết.

---

## 6. Nhóm thống kê bậc nhất – `firstorder` (18 features)

Tính toán từ **histogram cường độ voxel** bên trong ROI – không quan tâm đến vị trí không gian của voxel.

| Feature | Ý nghĩa y tế |
|---|---|
| `Mean` | Mật độ trung bình (HU). Phản ánh mật độ chung của khối u. |
| `Median` | Mật độ trung vị. Ít bị ảnh hưởng bởi voxel ngoại vi. |
| `10Percentile` / `90Percentile` | Đo lường đuôi phân phối. 10P thấp → có vùng mật độ thấp (hoại tử, dịch). 90P cao → mô đặc nhiều. |
| `InterquartileRange` | IQR = Q75 - Q25. Đo **độ phân tán** mật độ bên trong khối u – chỉ số dị biệt nội tại (intra-tumor heterogeneity). |
| `Range` | Max - Min. Khoảng mật độ tổng thể trong khối u. |
| `Variance` | Phương sai mật độ. Cao → khối u không đồng nhất (heterogeneous) → tiên lượng xấu hơn. |
| `Skewness` | Độ lệch phân phối HU. Âm → lệch về mật độ thấp (nhiều hoại tử); dương → lệch về mật độ cao (đặc). |
| `Kurtosis` | Độ nhọn phân phối. Cao → nhiều voxel cực trị (hoại tử + vôi hóa cùng tồn tại). |
| `Entropy` | Độ hỗn loạn của histogram. Cao → mô học phức tạp, dị biệt cao. Liên quan đến tình trạng kháng điều trị. |
| `Uniformity` | Nghịch đảo của Entropy. Cao → mô đồng nhất. |
| `Energy` | Tổng bình phương intensity. |
| `TotalEnergy` | Energy × thể tích voxel. |
| `RootMeanSquared` | Căn bậc hai của trung bình bình phương intensity. |
| `MeanAbsoluteDeviation` | Trung bình độ lệch tuyệt đối so với Mean. |
| `RobustMeanAbsoluteDeviation` | MAD tính trên 10–90 percentile, loại bỏ outlier. |
| `Minimum` | Giá trị HU thấp nhất. Gần -1000 → có vùng không khí (cavitation). |
| `Maximum` | Giá trị HU cao nhất. >400 HU → vôi hóa. |

**Ý nghĩa lâm sàng tổng hợp:** Entropy và Variance là hai chỉ số mạnh nhất dự đoán tiên lượng. Khối u có Entropy cao phản ánh tình trạng tumor microenvironment phức tạp – nhiều nhánh tế bào (clonal heterogeneity) – liên quan đến kháng thuốc.

---

## 7. Ma trận đồng xuất hiện mức xám – `glcm` (24 features)

**GLCM (Gray Level Co-occurrence Matrix):** Đếm tần suất cặp voxel có cường độ (i, j) xuất hiện kề nhau theo hướng nhất định. Phân tích **texture thứ cấp** – mối quan hệ không gian giữa các voxel.

| Feature | Ý nghĩa y tế |
|---|---|
| `Contrast` | Sự khác biệt cục bộ giữa các vùng. Cao → mô không đồng nhất, ranh giới nội bộ rõ. |
| `Correlation` | Mức độ tương quan tuyến tính giữa cặp voxel. Cao → texture có hướng (anisotropic). |
| `Entropy` / `JointEntropy` | Độ hỗn loạn trong cặp intensity. Cao → texture phức tạp. |
| `JointEnergy` | Nghịch đảo entropy. Cao → texture đồng nhất, lặp lại. |
| `Autocorrelation` | Tính tự tương quan. Cao → texture mịn, đều đặn. |
| `Homogeneity` (Idm, Id) | Tính đồng nhất cục bộ. Idm (Inverse Difference Moment) cao → mô đồng nhất. |
| `ClusterProminence` | Phân cụm mạnh trong texture. Cao → mô có vùng phân cụm rõ ràng (e.g., vùng hoại tử bao quanh). |
| `ClusterShade` | Bất đối xứng phân cụm. Liên quan đến tính không đồng nhất theo hướng. |
| `ClusterTendency` | Xu hướng phân cụm tổng thể. |
| `DifferenceAverage` | Trung bình độ chênh lệch giữa cặp intensity. |
| `DifferenceEntropy` | Entropy của hiệu các cặp. |
| `DifferenceVariance` | Phương sai của hiệu các cặp. |
| `SumAverage` | Trung bình tổng cặp intensity. |
| `SumEntropy` | Entropy của tổng cặp. |
| `SumSquares` (Variance) | Phương sai GLCM. |
| `Imc1`, `Imc2` | Thước đo tương quan thông tin – đo sự phụ thuộc thống kê phi tuyến giữa cặp voxel. |
| `InverseVariance` | Nghịch đảo của variance GLCM. |
| `MCC` (Maximal Correlation Coefficient) | Tương quan cực đại. |
| `MaximumProbability` | Xác suất cặp voxel phổ biến nhất. Cao → mô rất đồng nhất. |

**Ý nghĩa lâm sàng tổng hợp:** GLCM là nhóm feature quan trọng nhất trong radiomics. Khối u NSCLC hung hãn thường có Contrast và Entropy cao, Homogeneity thấp – phản ánh tính không đồng nhất vi mô do tân sinh mạch không đều và hoại tử cục bộ.

---

## 8. Ma trận phụ thuộc mức xám – `gldm` (14 features)

**GLDM (Gray Level Dependence Matrix):** Đếm số voxel lân cận có cùng mức xám, tạo thành "dependency zone". Mô tả **mức độ đồng nhất cục bộ** theo neighborhood.

| Feature | Ý nghĩa y tế |
|---|---|
| `SmallDependenceEmphasis` | Ưu thế vùng phụ thuộc nhỏ. Cao → texture mịn, ít lặp lại → mô đồng nhất. |
| `LargeDependenceEmphasis` | Ưu thế vùng phụ thuộc lớn. Cao → texture thô → mô không đồng nhất. |
| `GrayLevelNonUniformity` | Phân phối không đều của mức xám. Cao → mật độ HU biến thiên nhiều. |
| `GrayLevelVariance` | Phương sai mức xám trong dependency zones. |
| `DependenceNonUniformity` | Phân phối không đều của kích thước zone. |
| `DependenceVariance` | Phương sai kích thước zone. |
| `DependenceEntropy` | Entropy của dependency distribution. |
| `HighGrayLevelEmphasis` | Ưu thế vùng mật độ cao. Liên quan đến mô đặc, tăng sinh mạnh. |
| `LowGrayLevelEmphasis` | Ưu thế vùng mật độ thấp. Liên quan đến hoại tử, dịch. |
| `LargeDependenceHighGrayLevelEmphasis` | Vùng lớn mật độ cao → khối u đặc, phát triển đồng đều. |
| `LargeDependenceLowGrayLevelEmphasis` | Vùng lớn mật độ thấp → hoại tử diện rộng. |
| `SmallDependenceHighGrayLevelEmphasis` | Vùng nhỏ mật độ cao → điểm vôi hóa vi thể. |
| `SmallDependenceLowGrayLevelEmphasis` | Vùng nhỏ mật độ thấp → vi hoại tử rải rác. |

---

## 9. Ma trận độ dài đường mức xám – `glrlm` (16 features)

**GLRLM (Gray Level Run Length Matrix):** Đếm số lần voxel cùng mức xám xuất hiện liên tiếp (run) theo một hướng nhất định.

| Feature | Ý nghĩa y tế |
|---|---|
| `ShortRunEmphasis` | Ưu thế đường ngắn. Cao → texture mịn. Mô ung thư đặc có nhiều đường ngắn. |
| `LongRunEmphasis` | Ưu thế đường dài. Cao → texture thô, đồng nhất theo hướng. Liên quan đến sợi xơ hóa (fibrosis). |
| `GrayLevelNonUniformity` | Phân phối không đều mức xám trong runs. |
| `GrayLevelVariance` | Phương sai mức xám theo runs. |
| `RunLengthNonUniformity` | Phân phối không đều độ dài runs. |
| `RunVariance` | Phương sai độ dài runs. |
| `RunEntropy` | Entropy của matrix. Cao → texture phức tạp. |
| `RunPercentage` | Tỷ lệ runs / tổng voxel. Cao (gần 1) → nhiều runs ngắn → mô mịn. |
| `HighGrayLevelRunEmphasis` | Ưu thế runs mật độ cao. |
| `LowGrayLevelRunEmphasis` | Ưu thế runs mật độ thấp. |
| `LongRunHighGrayLevelEmphasis` | Vùng đặc kéo dài → mô xơ hóa, đặc. |
| `LongRunLowGrayLevelEmphasis` | Vùng hoại tử kéo dài → cavitation, necrosis dọc. |
| `ShortRunHighGrayLevelEmphasis` | Điểm đặc nhỏ phân tán. |
| `ShortRunLowGrayLevelEmphasis` | Điểm lỏng nhỏ phân tán. |

---

## 10. Ma trận kích thước vùng mức xám – `glszm` (16 features)

**GLSZM (Gray Level Size Zone Matrix):** Tương tự GLRLM nhưng đếm vùng kết nối 3D (connected zone) thay vì đường 1D.

| Feature | Ý nghĩa y tế |
|---|---|
| `SmallAreaEmphasis` | Ưu thế vùng nhỏ → mô không đồng nhất, nhiều vùng nhỏ khác nhau. |
| `LargeAreaEmphasis` | Ưu thế vùng lớn → mô đồng nhất theo vùng. |
| `ZonePercentage` | Tỷ lệ zones / voxels. Cao → nhiều zones nhỏ → texture mịn, không đồng nhất. |
| `ZoneEntropy` | Entropy phân bố zones. |
| `ZoneVariance` | Phương sai kích thước zones. |
| `GrayLevelNonUniformity` | Phân tán mức xám giữa các zones. |
| `GrayLevelVariance` | Phương sai mức xám. |
| `SizeZoneNonUniformity` | Kích thước zones không đều → mô có vùng hoại tử xen kẽ mô đặc. |
| `HighGrayLevelZoneEmphasis` | Ưu thế vùng mật độ cao → nhiều mô đặc. |
| `LowGrayLevelZoneEmphasis` | Ưu thế vùng mật độ thấp → nhiều hoại tử/dịch. |
| `LargeAreaHighGrayLevelEmphasis` | Khối đặc lớn → khối u solid stage tiến triển. |
| `LargeAreaLowGrayLevelEmphasis` | Vùng hoại tử lớn → khối u có trung tâm hoại tử (central necrosis). |
| `SmallAreaHighGrayLevelEmphasis` | Điểm vôi hóa, mạch máu nhỏ phân tán. |
| `SmallAreaLowGrayLevelEmphasis` | Vi hoại tử, bào tương lỏng nhỏ. |

---

## 11. Ma trận độ chênh tông lân cận – `ngtdm` (5 features)

**NGTDM (Neighboring Gray Tone Difference Matrix):** Đo sự khác biệt trung bình giữa mỗi voxel và vùng lân cận 3D của nó.

| Feature | Ý nghĩa y tế |
|---|---|
| `Coarseness` | Nghịch đảo tổng chênh lệch. Cao → texture mịn, đồng nhất. Thấp → texture thô. Khối u hung hãn thường có Coarseness thấp. |
| `Contrast` | Cường độ biến thiên cục bộ. Cao → nhiều vùng chuyển tiếp đột ngột → mô không đồng nhất. |
| `Busyness` | Tần suất chuyển đổi mức xám từ voxel sang voxel. Cao → texture "bận rộn", cấu trúc vi mô phức tạp → liên quan tân sinh mạch và vi xâm lấn. |
| `Complexity` | Kết hợp busyness và contrast. Đo mức độ phức tạp tổng thể. |
| `Strength` | Phản ánh tính rõ ràng của cấu trúc texture. Cao → texture dễ phân biệt (regular patterns). |

---

## 12. Tổng hợp cấu trúc features

| Nhóm | Số features | Image types áp dụng | Tổng cột |
|---|---|---|---|
| shape | 14 | original (1) | 14 |
| firstorder | 18 | 18 image types | 324 |
| glcm | 24 | 18 image types | 432 |
| gldm | 14 | 18 image types | 252 |
| glrlm | 16 | 18 image types | 288 |
| glszm | 16 | 18 image types | 288 |
| ngtdm | 5 | 18 image types | 90 |
| **Tổng feature** | | | **1.688** |
| metadata (job_tag, lesion_index) | | | 2 |
| **Tổng cột** | | | **1.690** |

---

## 13. Ý nghĩa các tham số trích xuất trong tên file

`spacing1.0_mirpon_window1350.250_allimagetypes_bw20`

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `spacing1.0` | 1.0 mm isotropic | Tái lấy mẫu về voxel 1×1×1 mm để chuẩn hóa giữa các scanner khác nhau |
| `mirpon` (MirpOn) | Bật | Mirror padding – phản chiếu biên ảnh khi tính wavelet để tránh artifact biên |
| `window1350.250` | WL=250, WW=1350 HU | Cửa sổ CT tối ưu cho mô phổi: [-425, +925] HU |
| `allimagetypes` | 18 loại | Trích xuất trên tất cả biến đổi ảnh |
| `bw20` | binWidth=20 HU | Gộp 20 HU thành 1 bin khi tính texture matrix → giảm nhiễu, tăng ổn định |

---

## 14. Cách tích hợp vào pipeline

```python
df_rad = pd.read_parquet(f"{BASE_DB_DIR}/lung_radiomics_spacing1.0_mirpon_window1350.250_allimagetypes_bw20.parquet")

# Chỉ dùng filtered-radiomics
df_rad_filtered = df_rad[df_rad['job_tag'] == 'filtered-radiomics']

df_rad_by_site = decorate_with_site_index(df_rad_filtered)

# Tổn thương phổi chính (PC = Primary Cancer), lấy lớn nhất
prepare_rad_modality_by_size(modality_dict, df_rad_by_site, modality_MASK,
                              sites=['PC'], name='rad_lesion_pc')

# Hoặc chọn theo lesion_index cụ thể
prepare_rad_modality(modality_dict, df_rad_by_site, modality_MASK,
                     sites=['PC'], l_idx=1, name='rad_pc_lesion1')
```

**Lưu ý:** Bộ lọc robustness (`l1_filter`) sẽ tiếp tục loại bỏ features có ICC < 0.85 hoặc outlier > 6σ trong quá trình training.
