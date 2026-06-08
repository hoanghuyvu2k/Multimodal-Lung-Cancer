# README: Giải thích các cột trong file Excel lung_radiomics_spacing1.0_mirpon_window1350.250_allimagetypes_bw20

## Tổng quan

File Excel này chứa dữ liệu radiomics được trích xuất từ hình ảnh CT phổi. File có tổng cộng **1690 cột**, bao gồm 2 cột metadata và 1688 cột đặc trưng radiomics.

### Thông tin về file:
- **Spacing**: 1.0mm (khoảng cách giữa các voxel)
- **Window/Level**: 1350/250 (cửa sổ và mức độ cường độ để hiển thị hình ảnh)
- **Bin Width**: 20 (độ rộng bin cho discretization)
- **Image Types**: Tất cả các loại filter/transformation được áp dụng

---

## Cấu trúc cột

### 1. Các cột Metadata (2 cột)

#### `job_tag`
- **Ý nghĩa**: Nhãn công việc/xử lý, dùng để phân loại các loại radiomics khác nhau (ví dụ: filtered-radiomics, pertubation-radiomics)
- **Kiểu dữ liệu**: String/Categorical

#### `lesion_index`
- **Ý nghĩa**: Chỉ số của tổn thương (lesion) trong bệnh nhân. Một bệnh nhân có thể có nhiều tổn thương
- **Kiểu dữ liệu**: Integer

---

## 2. Các cột đặc trưng Radiomics (1688 cột)

Các đặc trưng radiomics được tổ chức theo cấu trúc: **`[filter_type]_[feature_class]_[feature_name]`**

### 2.1. Các loại Filter/Transformation

File này chứa các đặc trưng được trích xuất từ nhiều loại filter khác nhau:

1. **`original`**: Hình ảnh gốc, không có filter
2. **`exponential`**: Áp dụng phép biến đổi exponential
3. **`gradient`**: Áp dụng gradient filter
4. **`lbp-2D`**: Local Binary Pattern 2D
5. **`lbp-3D-m1`**: Local Binary Pattern 3D method 1
6. **`lbp-3D-m2`**: Local Binary Pattern 3D method 2
7. **`lbp-3D-k`**: Local Binary Pattern 3D kernel
8. **`logarithm`**: Áp dụng phép biến đổi logarithm
9. **`square`**: Áp dụng phép bình phương
10. **`squareroot`**: Áp dụng phép căn bậc hai
11. **`wavelet-LLH`**: Wavelet transform (Low-Low-High)
12. **`wavelet-LHL`**: Wavelet transform (Low-High-Low)
13. **`wavelet-LHH`**: Wavelet transform (Low-High-High)
14. **`wavelet-HLL`**: Wavelet transform (High-Low-Low)
15. **`wavelet-HLH`**: Wavelet transform (High-Low-High)
16. **`wavelet-HHL`**: Wavelet transform (High-High-Low)
17. **`wavelet-HHH`**: Wavelet transform (High-High-High)
18. **`wavelet-LLL`**: Wavelet transform (Low-Low-Low)

### 2.2. Các lớp đặc trưng (Feature Classes)

#### A. Shape Features (chỉ có trong `original`)
Các đặc trưng mô tả hình dạng 3D của tổn thương:

- **`Elongation`**: Độ dài dài (tỷ lệ giữa trục lớn và trục nhỏ)
- **`Flatness`**: Độ phẳng (tỷ lệ giữa trục nhỏ và trục trung bình)
- **`LeastAxisLength`**: Độ dài trục nhỏ nhất
- **`MajorAxisLength`**: Độ dài trục lớn nhất
- **`Maximum2DDiameterColumn`**: Đường kính lớn nhất 2D theo cột
- **`Maximum2DDiameterRow`**: Đường kính lớn nhất 2D theo hàng
- **`Maximum2DDiameterSlice`**: Đường kính lớn nhất 2D theo slice
- **`Maximum3DDiameter`**: Đường kính lớn nhất 3D
- **`MeshVolume`**: Thể tích mesh (tính từ bề mặt)
- **`MinorAxisLength`**: Độ dài trục nhỏ
- **`Sphericity`**: Độ cầu (tỷ lệ giữa thể tích và diện tích bề mặt)
- **`SurfaceArea`**: Diện tích bề mặt
- **`SurfaceVolumeRatio`**: Tỷ lệ diện tích bề mặt/thể tích
- **`VoxelVolume`**: Thể tích tính bằng số voxel

#### B. First Order Features
Các đặc trưng thống kê cơ bản về phân phối cường độ pixel:

- **`10Percentile`**: Giá trị percentile thứ 10
- **`90Percentile`**: Giá trị percentile thứ 90
- **`Energy`**: Năng lượng (tổng bình phương các giá trị pixel)
- **`Entropy`**: Entropy (đo độ ngẫu nhiên/phức tạp)
- **`InterquartileRange`**: Khoảng tứ phân vị (Q3 - Q1)
- **`Kurtosis`**: Độ nhọn (đo độ phân tán của phân phối)
- **`Maximum`**: Giá trị tối đa
- **`MeanAbsoluteDeviation`**: Độ lệch tuyệt đối trung bình
- **`Mean`**: Giá trị trung bình
- **`Median`**: Giá trị trung vị
- **`Minimum`**: Giá trị tối thiểu
- **`Range`**: Khoảng giá trị (max - min)
- **`RobustMeanAbsoluteDeviation`**: Độ lệch tuyệt đối trung bình mạnh (robust)
- **`RootMeanSquared`**: Căn bậc hai của trung bình bình phương
- **`Skewness`**: Độ lệch (đo độ bất đối xứng của phân phối)
- **`TotalEnergy`**: Tổng năng lượng
- **`Uniformity`**: Độ đồng nhất
- **`Variance`**: Phương sai

#### C. GLCM (Gray Level Co-occurrence Matrix) Features
Các đặc trưng mô tả mối quan hệ không gian giữa các pixel có cường độ tương tự:

- **`Autocorrelation`**: Tự tương quan
- **`ClusterProminence`**: Độ nổi bật của cluster
- **`ClusterShade`**: Độ bóng của cluster
- **`ClusterTendency`**: Xu hướng cluster
- **`Contrast`**: Độ tương phản
- **`Correlation`**: Tương quan
- **`DifferenceAverage`**: Trung bình của sự khác biệt
- **`DifferenceEntropy`**: Entropy của sự khác biệt
- **`DifferenceVariance`**: Phương sai của sự khác biệt
- **`Id`**: Inverse Difference
- **`Idm`**: Inverse Difference Moment
- **`Idmn`**: Inverse Difference Moment Normalized
- **`Idn`**: Inverse Difference Normalized
- **`Imc1`**: Informational Measure of Correlation 1
- **`Imc2`**: Informational Measure of Correlation 2
- **`InverseVariance`**: Nghịch đảo phương sai
- **`JointAverage`**: Trung bình chung
- **`JointEnergy`**: Năng lượng chung
- **`JointEntropy`**: Entropy chung
- **`MCC`**: Maximal Correlation Coefficient
- **`MaximumProbability`**: Xác suất tối đa
- **`SumAverage`**: Trung bình tổng
- **`SumEntropy`**: Entropy tổng
- **`SumSquares`**: Tổng bình phương

#### D. GLDM (Gray Level Dependence Matrix) Features
Các đặc trưng mô tả sự phụ thuộc giữa các mức xám:

- **`DependenceEntropy`**: Entropy của sự phụ thuộc
- **`DependenceNonUniformity`**: Độ không đồng nhất của sự phụ thuộc
- **`DependenceNonUniformityNormalized`**: Độ không đồng nhất chuẩn hóa
- **`DependenceVariance`**: Phương sai của sự phụ thuộc
- **`GrayLevelNonUniformity`**: Độ không đồng nhất mức xám
- **`GrayLevelVariance`**: Phương sai mức xám
- **`HighGrayLevelEmphasis`**: Nhấn mạnh mức xám cao
- **`LargeDependenceEmphasis`**: Nhấn mạnh sự phụ thuộc lớn
- **`LargeDependenceHighGrayLevelEmphasis`**: Nhấn mạnh sự phụ thuộc lớn với mức xám cao
- **`LargeDependenceLowGrayLevelEmphasis`**: Nhấn mạnh sự phụ thuộc lớn với mức xám thấp
- **`LowGrayLevelEmphasis`**: Nhấn mạnh mức xám thấp
- **`SmallDependenceEmphasis`**: Nhấn mạnh sự phụ thuộc nhỏ
- **`SmallDependenceHighGrayLevelEmphasis`**: Nhấn mạnh sự phụ thuộc nhỏ với mức xám cao
- **`SmallDependenceLowGrayLevelEmphasis`**: Nhấn mạnh sự phụ thuộc nhỏ với mức xám thấp

#### E. GLRLM (Gray Level Run Length Matrix) Features
Các đặc trưng mô tả độ dài của các chuỗi pixel liên tiếp có cùng mức xám:

- **`GrayLevelNonUniformity`**: Độ không đồng nhất mức xám
- **`GrayLevelNonUniformityNormalized`**: Độ không đồng nhất mức xám chuẩn hóa
- **`GrayLevelVariance`**: Phương sai mức xám
- **`HighGrayLevelRunEmphasis`**: Nhấn mạnh chuỗi mức xám cao
- **`LongRunEmphasis`**: Nhấn mạnh chuỗi dài
- **`LongRunHighGrayLevelEmphasis`**: Nhấn mạnh chuỗi dài với mức xám cao
- **`LongRunLowGrayLevelEmphasis`**: Nhấn mạnh chuỗi dài với mức xám thấp
- **`LowGrayLevelRunEmphasis`**: Nhấn mạnh chuỗi mức xám thấp
- **`RunEntropy`**: Entropy của chuỗi
- **`RunLengthNonUniformity`**: Độ không đồng nhất độ dài chuỗi
- **`RunLengthNonUniformityNormalized`**: Độ không đồng nhất độ dài chuỗi chuẩn hóa
- **`RunPercentage`**: Phần trăm chuỗi
- **`RunVariance`**: Phương sai độ dài chuỗi
- **`ShortRunEmphasis`**: Nhấn mạnh chuỗi ngắn
- **`ShortRunHighGrayLevelEmphasis`**: Nhấn mạnh chuỗi ngắn với mức xám cao
- **`ShortRunLowGrayLevelEmphasis`**: Nhấn mạnh chuỗi ngắn với mức xám thấp

#### F. GLSZM (Gray Level Size Zone Matrix) Features
Các đặc trưng mô tả kích thước của các vùng có cùng mức xám:

- **`GrayLevelNonUniformity`**: Độ không đồng nhất mức xám
- **`GrayLevelNonUniformityNormalized`**: Độ không đồng nhất mức xám chuẩn hóa
- **`GrayLevelVariance`**: Phương sai mức xám
- **`HighGrayLevelZoneEmphasis`**: Nhấn mạnh vùng mức xám cao
- **`LargeAreaEmphasis`**: Nhấn mạnh vùng lớn
- **`LargeAreaHighGrayLevelEmphasis`**: Nhấn mạnh vùng lớn với mức xám cao
- **`LargeAreaLowGrayLevelEmphasis`**: Nhấn mạnh vùng lớn với mức xám thấp
- **`LowGrayLevelZoneEmphasis`**: Nhấn mạnh vùng mức xám thấp
- **`SizeZoneNonUniformity`**: Độ không đồng nhất kích thước vùng
- **`SizeZoneNonUniformityNormalized`**: Độ không đồng nhất kích thước vùng chuẩn hóa
- **`SmallAreaEmphasis`**: Nhấn mạnh vùng nhỏ
- **`SmallAreaHighGrayLevelEmphasis`**: Nhấn mạnh vùng nhỏ với mức xám cao
- **`SmallAreaLowGrayLevelEmphasis`**: Nhấn mạnh vùng nhỏ với mức xám thấp
- **`ZoneEntropy`**: Entropy của vùng
- **`ZonePercentage`**: Phần trăm vùng
- **`ZoneVariance`**: Phương sai kích thước vùng

#### G. NGTDM (Neighboring Gray Tone Difference Matrix) Features
Các đặc trưng mô tả sự khác biệt giữa mức xám của một pixel và trung bình của các pixel lân cận:

- **`Busyness`**: Độ bận rộn (đo độ thay đổi nhanh của cường độ)
- **`Coarseness`**: Độ thô (đo độ mịn/thô của texture)
- **`Complexity`**: Độ phức tạp
- **`Contrast`**: Độ tương phản
- **`Strength`**: Độ mạnh

---

## Cách đọc tên cột

Ví dụ: `original_firstorder_Mean`

- **`original`**: Filter type (hình ảnh gốc)
- **`firstorder`**: Feature class (đặc trưng bậc nhất)
- **`Mean`**: Feature name (giá trị trung bình)

Ví dụ: `wavelet-LLH_glcm_Contrast`

- **`wavelet-LLH`**: Filter type (wavelet transform Low-Low-High)
- **`glcm`**: Feature class (Gray Level Co-occurrence Matrix)
- **`Contrast`**: Feature name (độ tương phản)

---

## Lưu ý quan trọng

1. **Shape features** chỉ có trong `original` filter (không có trong các filter khác)
2. Mỗi filter type có thể có các feature classes khác nhau tùy thuộc vào tính chất của filter
3. Các giá trị trong file đã được tính toán với:
   - Spacing: 1.0mm
   - Window/Level: 1350/250
   - Bin Width: 20
4. File này được tạo từ PyRadiomics library, tuân theo Image Biomarker Standardization Initiative (IBSI)

---

## Ứng dụng

Các đặc trưng radiomics này có thể được sử dụng để:
- Phân loại tổn thương (benign vs malignant)
- Dự đoán đáp ứng điều trị
- Phân tích tiên lượng bệnh
- Nghiên cứu mối quan hệ giữa hình ảnh và kết quả lâm sàng

---

---

## Quy trình chuyển đổi ảnh CT scan thành file Excel

File Excel `lung_radiomics_spacing1.0_mirpon_window1350.250_allimagetypes_bw20.xlsx` là kết quả của một quy trình xử lý hình ảnh CT phổi phức tạp sử dụng thư viện PyRadiomics. Dưới đây là các bước chính trong quy trình:

### Bước 1: Thu thập và nạp dữ liệu ảnh CT

1. **Đọc file DICOM/NIfTI**: 
   - Ảnh CT scan được lưu trữ dưới dạng DICOM (Digital Imaging and Communications in Medicine) hoặc NIfTI
   - Sử dụng thư viện như `pydicom`, `nibabel`, hoặc `SimpleITK` để đọc dữ liệu ảnh 3D

2. **Chuyển đổi sang định dạng SimpleITK**:
   - PyRadiomics sử dụng SimpleITK để xử lý ảnh
   - Ảnh được chuyển đổi thành đối tượng `SimpleITK.Image`

### Bước 2: Tiền xử lý ảnh (Image Preprocessing)

1. **Resampling (Tái lấy mẫu)**:
   - **Spacing 1.0mm**: Tất cả ảnh được resample về khoảng cách voxel đồng nhất 1.0mm × 1.0mm × 1.0mm
   - Đảm bảo tính nhất quán giữa các ảnh từ các máy CT khác nhau
   - Sử dụng interpolation (thường là linear hoặc B-spline)

2. **Áp dụng Window/Level**:
   - **Window: 1350, Level: 250**: Áp dụng cửa sổ cường độ để tối ưu hóa hiển thị mô phổi
   - Window/Level giúp tập trung vào phạm vi cường độ HU (Hounsfield Units) quan trọng
   - Công thức: `pixel_value = (HU - Level + Window/2) / Window`

3. **Discretization (Rời rạc hóa)**:
   - **Bin Width: 20**: Chia phạm vi cường độ thành các bin có độ rộng 20 HU
   - Giảm số lượng mức xám để tính toán hiệu quả hơn
   - Ví dụ: nếu cường độ từ -1000 đến 1000 HU, sẽ có khoảng 100 bins

### Bước 3: Phân đoạn vùng quan tâm (ROI Segmentation)

1. **Xác định tổn thương (Lesion)**:
   - Bác sĩ X-quang (Radiologist) khoanh vùng các tổn thương trong phổi
   - Mỗi tổn thương được đánh số bằng `lesion_index`
   - Một bệnh nhân có thể có nhiều tổn thương
   - Các loại tổn thương: Parenchymal (PC), Pleural (PL), Nodal (LN)

2. **Tạo mask nhị phân**:
   - Tạo mask 3D cho mỗi tổn thương (1 = trong vùng tổn thương, 0 = ngoài vùng)
   - Mask được sử dụng để giới hạn vùng tính toán đặc trưng

3. **Generate Perturbations (Tạo biến thể segmentation)**:
   - Từ segmentation gốc của bác sĩ, tạo ra nhiều biến thể (perturbations) của mask
   - Mục đích: Đánh giá tính robust (ổn định) của các đặc trưng radiomics
   - Các perturbations được tạo bằng cách thêm nhiễu hoặc thay đổi nhỏ ranh giới segmentation
   - Mỗi tổn thương có thể có nhiều perturbations (thường 10-100 perturbations)

### Bước 4: Trích xuất đặc trưng Radiomics

PyRadiomics trích xuất đặc trưng theo các bước sau:

1. **Generate Scan và Resample Voxels**:
   - Từ CT images DICOM, tạo scan 3D
   - Resample về spacing 1.0mm để đảm bảo tính nhất quán

2. **Áp dụng các Image Filters**:
   - **Original**: Ảnh gốc sau khi resample và discretize
   - **Exponential**: `exp(x)`
   - **Logarithm**: `log(x + 1)`
   - **Square**: `x²`
   - **Square Root**: `√x`
   - **Gradient**: Tính gradient của ảnh
   - **LBP (Local Binary Pattern)**: 2D và 3D với các phương pháp khác nhau
   - **Wavelet**: 8 biến thể (LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH)

3. **Tính toán các Feature Classes**:
   - **Shape Features**: Từ mask 3D (chỉ có trong original)
   - **First Order**: Thống kê về phân phối cường độ
   - **GLCM**: Ma trận đồng xuất hiện mức xám
   - **GLDM**: Ma trận phụ thuộc mức xám
   - **GLRLM**: Ma trận độ dài chuỗi mức xám
   - **GLSZM**: Ma trận kích thước vùng mức xám
   - **NGTDM**: Ma trận khác biệt mức xám lân cận

4. **Hai loại Radiomics được tạo**:
   - **Original Radiomics**: Trích xuất từ segmentation gốc của bác sĩ
   - **Perturbation Radiomics**: Trích xuất từ các perturbations của segmentation
   - Được đánh dấu bằng `job_tag`: `'filtered-radiomics'` (original) và `'pertubation-radiomics'` (perturbations)

5. **Tổ chức dữ liệu**:
   - Mỗi tổn thương (và mỗi perturbation) → một hàng trong DataFrame
   - Mỗi đặc trưng → một cột
   - Định dạng: `[filter]_[feature_class]_[feature_name]`

### Bước 5: Feature Selection (Lựa chọn đặc trưng)

Quá trình lựa chọn đặc trưng được thực hiện qua 3 bước:

1. **Remove Non-Robust Features (Loại bỏ đặc trưng không ổn định)**:
   - Sử dụng **Perturbation Radiomics** để đánh giá tính robust
   - Tính toán inter-lesion variance: `mean(intra-lesion std) / std(intra-lesion mean)`
   - Loại bỏ các đặc trưng có variance cao (cutoff = 0.15)
   - Mục đích: Giữ lại các đặc trưng ổn định giữa các perturbations của cùng một tổn thương

2. **Reject Outlier Prone Features (Loại bỏ đặc trưng dễ bị outlier)**:
   - Phát hiện các đặc trưng có nhiều giá trị ngoại lai (outliers)
   - Sử dụng phương pháp IQR (Interquartile Range) hoặc Z-score
   - Loại bỏ các đặc trưng có quá nhiều outliers (cutoff = 6 lần)
   - Mục đích: Giữ lại các đặc trưng có phân phối ổn định

3. **Feature Selection using L1 Regularization (Lựa chọn bằng L1 regularization)**:
   - Sử dụng **Original Radiomics** (filtered-radiomics) đã được lọc qua 2 bước trên
   - Áp dụng Elastic Net Logistic Regression với L1 regularization
   - L1 ratio = 0.5, C = 0.1 (l1_strength)
   - Chỉ giữ lại các đặc trưng có hệ số khác 0 sau khi fit model
   - Mục đích: Chọn các đặc trưng có khả năng dự đoán tốt nhất

4. **Kết quả**:
   - **Reduced Feature Set**: Tập đặc trưng cuối cùng đã được lọc và chọn lọc
   - Từ 1688 đặc trưng ban đầu → giảm xuống còn vài chục đến vài trăm đặc trưng quan trọng

### Bước 6: Xử lý và lưu trữ

1. **Thêm metadata**:
   - `job_tag`: Phân loại loại radiomics (filtered-radiomics, pertubation-radiomics, etc.)
   - `lesion_index`: Chỉ số tổn thương
   - `main_index`: ID bệnh nhân (thường là index của DataFrame)

2. **Lưu vào Parquet**:
   - Dữ liệu ban đầu được lưu dưới dạng Parquet (hiệu quả hơn CSV)
   - File: `lung_radiomics_spacing1.0_mirpon_window1350.250_allimagetypes_bw20.parquet`

3. **Chuyển đổi sang Excel**:
   - Đọc từ Parquet file
   - Chuyển đổi sang DataFrame pandas
   - Lưu vào Excel với `pandas.ExcelWriter`
   - File: `lung_radiomics_spacing1.0_mirpon_window1350.250_allimagetypes_bw20.xlsx`

### Sơ đồ quy trình tổng quan

```
INPUTS:
├── Feature extraction parameters (settings)
├── CT images in DICOM format
└── Radiologist's segmentations (ROI masks)
    ↓
[1] Generate Scan & Resample Voxels
    ├── Đọc CT images DICOM
    ├── Chuyển đổi sang SimpleITK
    └── Resample về spacing 1.0mm
    ↓
[2] Generate Perturbations
    └── Tạo nhiều biến thể từ segmentation gốc
    ↓
[3] Extract Radiomics
    ├── Áp dụng Window/Level (1350/250)
    ├── Discretization (Bin Width = 20)
    ├── Áp dụng 18 loại Image Filters
    └── Trích xuất 7 Feature Classes cho mỗi filter
    ↓
OUTPUT: Hai loại Radiomics
├── Original Radiomics (từ segmentation gốc)
└── Perturbation Radiomics (từ perturbations)
    ↓
[4] Feature Selection
    ├── Step 1: Remove non-robust features
    │   └── Dựa trên inter-lesion variance từ perturbations
    ├── Step 2: Reject outlier prone features
    │   └── Loại bỏ features có nhiều outliers
    └── Step 3: Feature selection using L1 regularization
        └── Elastic Net Logistic Regression
    ↓
OUTPUT: Reduced Feature Set
    ↓
[5] Tổ chức thành DataFrame (1690 cột ban đầu → giảm sau feature selection)
    ↓
[6] Lưu Parquet → Chuyển Excel
    ↓
File Excel cuối cùng
```

### Các thông số kỹ thuật quan trọng

- **Spacing 1.0mm**: Đảm bảo tính nhất quán không gian
- **Window/Level 1350/250**: Tối ưu cho hiển thị mô phổi
- **Bin Width 20**: Cân bằng giữa độ chi tiết và hiệu suất tính toán
- **MirpOn**: Có thể là một flag hoặc setting đặc biệt trong PyRadiomics

### Lưu ý về quy trình

1. **Tính tái tạo (Reproducibility)**:
   - Tất cả các thông số được cố định để đảm bảo kết quả có thể tái tạo
   - Tuân theo tiêu chuẩn IBSI (Image Biomarker Standardization Initiative)

2. **Tính Robust (Ổn định)**:
   - Việc sử dụng perturbations giúp đánh giá tính ổn định của đặc trưng
   - Các đặc trưng được chọn phải ổn định giữa các perturbations của cùng một tổn thương
   - Điều này đảm bảo kết quả không phụ thuộc quá nhiều vào ranh giới segmentation

3. **Hiệu suất**:
   - Quy trình này có thể mất nhiều thời gian cho một dataset lớn
   - Việc tạo perturbations và trích xuất features cho mỗi perturbation làm tăng thời gian xử lý
   - Thường được chạy trên server/cluster với nhiều CPU cores

4. **Chất lượng dữ liệu**:
   - Chất lượng segmentation của bác sĩ X-quang ảnh hưởng lớn đến kết quả
   - Cần kiểm tra và validate dữ liệu đầu vào
   - Perturbations giúp giảm thiểu ảnh hưởng của sai số trong segmentation

5. **Feature Selection**:
   - Quy trình 3 bước đảm bảo chỉ giữ lại các đặc trưng:
     - Ổn định (robust) giữa các perturbations
     - Không có nhiều outliers
     - Có khả năng dự đoán tốt (predictive power)

---

## Tham khảo

- PyRadiomics Documentation: https://pyradiomics.readthedocs.io/
- Image Biomarker Standardization Initiative (IBSI): https://theibsi.github.io/
- SimpleITK Documentation: https://simpleitk.org/
- DICOM Standard: https://www.dicomstandard.org/

