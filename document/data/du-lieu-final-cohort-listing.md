# Tài liệu dữ liệu: `final_cohort_listing.csv`

## Tổng quan

| Thuộc tính | Giá trị |
|---|---|
| Đường dẫn | `../datasets/final_cohort_listing.csv` |
| Kích thước | 368 dòng × 2 cột |
| Mục đích | Danh sách toàn bộ bệnh nhân trong dự án, phân chia theo cohort (tập huấn luyện / tập kiểm định) |
| Không có giá trị null | Đúng – dữ liệu đầy đủ 100% |

File này là **bản đồ phân vùng dữ liệu** – nó không chứa thông tin lâm sàng, mà chỉ xác định mỗi bệnh nhân thuộc tập nào. Mọi script huấn luyện đều load file này trước để lọc cohort phù hợp.

---

## Cấu trúc cột

### `main_index` (string, khóa chính)

Mã định danh duy nhất của bệnh nhân trong toàn bộ dự án. Đây là cột index được dùng để join với tất cả các file dữ liệu khác.

Định dạng ID khác nhau tùy theo cohort:

| Cohort | Định dạng | Ví dụ | Độ dài |
|---|---|---|---|
| `discovery` | Tiền tố `P-` + 7 số | `P-0013653` | 9 ký tự cố định |
| `rad_valid` | Số nguyên thuần túy | `190868` | 6 chữ số |
| `path_valid` | Số nguyên thuần túy | `3369788` | 6–7 chữ số |

- Không có bệnh nhân trùng lặp giữa các cohort (giao nhau = 0).
- Trong code, `main_index` được dùng làm `.set_index('main_index')` khi load DataFrame.
- Với cohort `discovery`, `main_index` tương ứng với cột `dmp_pt_id` trong file omnibus inventory (MSK-MIND project ID).

### `cohort` (string, categorical)

Phân loại cohort của bệnh nhân. Có 3 giá trị:

| Giá trị | Số lượng | Tỉ lệ | Ý nghĩa |
|---|---|---|---|
| `discovery` | 247 | 67.1% | Tập khám phá – dùng để huấn luyện và cross-validation (10-fold CV) |
| `path_valid` | 71 | 19.3% | Tập kiểm định độc lập về **pathology** (dữ liệu giải phẫu bệnh / PD-L1) |
| `rad_valid` | 50 | 13.6% | Tập kiểm định độc lập về **radiomics** (dữ liệu CT scan) |

---

## Phân tích từng cohort

### Cohort `discovery` (247 bệnh nhân)

Đây là tập dữ liệu chính cho việc huấn luyện mô hình.

- **Nguồn dữ liệu chính:** File omnibus inventory (`18193mskmindprojectm-...csv`) – chứa đầy đủ thông tin lâm sàng (366 dòng, bao phủ 247 bệnh nhân discovery).
- **Dữ liệu radiomics:** `lung_radiomics_spacing1.0_mirpon_window1350.250_allimagetypes_bw20.parquet`
- **Dữ liệu pathology:** `lung_pathology_pdl1_glcm_v3.parquet`
- **Dữ liệu genomics:** `genomic_data_v3.parquet`
- **Cách load trong code:**
  ```python
  df_cohort = pd.read_csv('../datasets/final_cohort_listing.csv').set_index('main_index')
  df_cohort_disc = df_cohort[df_cohort['cohort'] == 'discovery']
  df_clinical = get_clinical_table_v2(path=..., main_index_col='dmp_pt_id', cohort=df_cohort_disc)
  ```

**Đặc điểm lâm sàng của cohort discovery** (từ omnibus inventory):

| Đặc điểm | Phân bố |
|---|---|
| Tuổi | 30–93 tuổi, trung bình 66.8 |
| Giới tính | Nam (sex=1): 168 (45.7%), Nữ (sex=2): 198 (53.8%) |
| Mô học | Adenocarcinoma: 268 (73%), Squamous: 56 (15.2%), NOS: 26 (7.1%) |
| ECOG | 0: 46 (12.5%), 1: 289 (78.7%), 2: 29 (7.9%), 3: 2 |
| Nhãn kết quả | label=1 (không đáp ứng SD/POD): 273 (74.2%), label=0 (đáp ứng PR/CR): 93 (25.3%) |
| Phác đồ điều trị | Anti-PD-1: 197 (78.8%), Anti-PD-L1: 50 (20%), Combo: 12 |

### Cohort `rad_valid` (50 bệnh nhân)

Tập kiểm định độc lập cho mô hình sử dụng dữ liệu **CT radiomics**.

- **Nguồn dữ liệu:** `lung_radiomics_spacing1.0_mirpon_window1350.250_allimagetypes_bw20_validation.parquet`
- **Số bệnh nhân thực tế có dữ liệu radiomics:** 46 (4 bệnh nhân không có CT segmentation)
- **ID format:** 6 chữ số (ví dụ: `190868`) – đây là radiology accession number
- **Mục đích:** Đánh giá khả năng tổng quát hóa của mô hình radiomics trên dữ liệu CT từ nguồn khác

### Cohort `path_valid` (71 bệnh nhân)

Tập kiểm định độc lập cho mô hình sử dụng dữ liệu **pathology (giải phẫu bệnh)**.

- **Nguồn dữ liệu:** `lung_pathology_pdl1_glcm_v3_validation.parquet`
- **Số bệnh nhân thực tế có dữ liệu pathology:** 52 (19 bệnh nhân không có slide)
- **ID format:** 6–7 chữ số (ví dụ: `3369788`) – đây là pathology case ID
- **Mục đích:** Đánh giá khả năng tổng quát hóa của mô hình pathology/PD-L1

---

## Mối quan hệ với các file dữ liệu khác

```
final_cohort_listing.csv
        │
        ├── cohort='discovery'  ──► dmp_pt_id join ──► omnibus_inventory.csv (lâm sàng)
        │                                         ──► lung_radiomics_...parquet (CT)
        │                                         ──► lung_pathology_...parquet (giải phẫu bệnh)
        │                                         ──► genomic_data_v3.parquet (genomics)
        │
        ├── cohort='rad_valid'  ──► main_index join ──► lung_radiomics_..._validation.parquet
        │
        └── cohort='path_valid' ──► main_index join ──► lung_pathology_..._validation.parquet
```

**Lưu ý quan trọng:**
- Cohort `discovery` dùng `dmp_pt_id` (dạng `P-XXXXXXX`) để join với omnibus inventory, nhưng dùng `main_index` (dạng `R-XXX`) để join với một số file nội bộ khác.
- Cohort `rad_valid` và `path_valid` dùng trực tiếp `main_index` (số nguyên) để join với file validation tương ứng.
- Không có sự chồng lấp bệnh nhân giữa các cohort (mỗi bệnh nhân chỉ thuộc đúng 1 cohort).

---

## Cách sử dụng trong code

```python
import pandas as pd

# Load và phân chia cohort
df_cohort = pd.read_csv('../datasets/final_cohort_listing.csv').set_index('main_index')

# Lấy tập discovery để train
df_disc = df_cohort[df_cohort['cohort'] == 'discovery']

# Lấy tập validation radiomics
df_rad_val = df_cohort[df_cohort['cohort'] == 'rad_valid']

# Lấy tập validation pathology
df_path_val = df_cohort[df_cohort['cohort'] == 'path_valid']

# Kiểm tra số lượng
print(df_disc.shape)      # (247, 1)
print(df_rad_val.shape)   # (50, 1)
print(df_path_val.shape)  # (71, 1)
```

---

## Thống kê tóm tắt

| Metric | Giá trị |
|---|---|
| Tổng số bệnh nhân | 368 |
| Số cột | 2 |
| Giá trị null | 0 |
| Bệnh nhân trùng lặp | 0 |
| Cohort để train (discovery) | 247 (67.1%) |
| Cohort kiểm định pathology | 71 (19.3%) |
| Cohort kiểm định radiomics | 50 (13.6%) |
