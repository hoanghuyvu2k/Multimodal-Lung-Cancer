# Phân tích dữ liệu: Genomic Data v3

**File:** `genomic_data_v3.parquet`  
**Ngày tạo tài liệu:** 2026-06-01  
**Nguồn:** MSK IMPACT panel genomics – được xử lý từ `genomic_inventory_v3.csv`

---

## 1. Tổng quan

| Thuộc tính | Giá trị |
|---|---|
| Số bệnh nhân | 247 |
| Số đặc trưng | 11 |
| Dữ liệu thiếu | Không có (0%) |
| Kiểu dữ liệu | Tất cả `int64` |
| Khóa chính (`index`) | `main_index` (định dạng `P-XXXXXXX`, là `dmp_pt_id`) |

File không có giá trị thiếu – toàn bộ 247 bệnh nhân có đủ thông tin cho 11 đặc trưng.

**Lưu ý về index:** `main_index` trong file này dùng định dạng `P-XXXXXXX` (MSK DMP patient ID), khác với định dạng `R-XXX` trong file lâm sàng Omnibus. Khi join với bảng lâm sàng cần dùng cột `dmp_pt_id` làm khóa join (không phải cột `main_index` của bảng lâm sàng).

---

## 2. Các đặc trưng

### 2.1 Gánh nặng đột biến khối u – `TMB` (Tumor Mutational Burden)

| Thống kê | Giá trị |
|---|---|
| Trung bình | 10.2 mut/Mb |
| Trung vị | 7.0 mut/Mb |
| Độ lệch chuẩn | 11.3 |
| Khoảng | 0 – 90 |
| Tứ phân vị 25% / 75% | 4 / 12 |

**Phân bố theo khoảng TMB:**

| Khoảng | Số bệnh nhân |
|---|---|
| 0–4 | 74 (30.0%) |
| 5–9 | 80 (32.4%) |
| 10–14 | 49 (19.8%) |
| 15–19 | 16 (6.5%) |
| 20–29 | 15 (6.1%) |
| 30–49 | 10 (4.0%) |
| 50+ | 3 (1.2%) |

Phân bố lệch phải rõ rệt; ~62% bệnh nhân có TMB < 10 (TMB-Low). Điểm ngưỡng TMB ≥ 10 thường dùng trong lâm sàng.

### 2.2 Đột biến driver (7 gen)

Các cột dạng `<gene> driver_binarized`: 1 = có đột biến driver, 0 = không.

| Gen | # Dương tính | Tỷ lệ | Vai trò lâm sàng |
|---|---|---|---|
| `STK11` | 44 | **17.8%** | Chỉ thị kháng PD-1/PD-L1 |
| `EGFR` | 22 | **8.9%** | Đột biến sensitizing → TKI |
| `ERBB2` | 19 | 7.7% | Khuếch đại / đột biến HER2 |
| `ARID1A` | 16 | 6.5% | Biểu sinh chromatin remodeling |
| `MET` | 12 | 4.9% | MET exon 14 skipping |
| `BRAF` | 7 | 2.8% | BRAF V600E / non-V600 |
| (tổng có ≥1 driver) | 111 | **44.9%** | |

### 2.3 Khuếch đại gen (4 gen)

Các cột dạng `<gene>: AMP_binarized`: 1 = có khuếch đại (copy number gain), 0 = không.

| Gen | # Dương tính | Tỷ lệ |
|---|---|---|
| `EGFR: AMP` | 10 | 4.0% |
| `ERBB2: AMP` | 6 | 2.4% |
| `MET: AMP` | 5 | 2.0% |
| `RET: AMP` | 1 | 0.4% |
| (tổng có ≥1 AMP) | 21 | **8.5%** |

---

## 3. Phân bố theo nhãn điều trị

Sau khi join với bảng lâm sàng qua `dmp_pt_id` → 247 bệnh nhân, trong đó label=0 (đáp ứng): 62, label=1 (không đáp ứng): 185.

### 3.1 TMB theo kết quả điều trị

| Nhóm | n | Trung bình | Trung vị | TMB ≥ 10 |
|---|---|---|---|---|
| Đáp ứng (`label=0`) | 62 | 13.1 | 10.5 | 53.2% |
| Không đáp ứng (`label=1`) | 185 | 9.2 | 7.0 | 32.4% |

Bệnh nhân đáp ứng có TMB trung bình cao hơn và tỷ lệ TMB-High (~53% vs ~32%), phù hợp với y văn về mối liên hệ TMB–đáp ứng miễn dịch.

### 3.2 Tỷ lệ đột biến driver theo kết quả điều trị

| Gen | label=0 (đáp ứng) | label=1 (không đáp ứng) | Nhận xét |
|---|---|---|---|
| `EGFR driver` | 0.0% (0/62) | 11.9% (22/185) | EGFR đột biến → không đáp ứng ICI |
| `STK11 driver` | 8.1% (5/62) | 21.1% (39/185) | STK11 là dấu hiệu kháng trị nổi bật |
| `ERBB2 driver` | 4.8% (3/62) | 8.6% (16/185) | Nhẹ hơn, hướng không đáp ứng |
| `ARID1A driver` | 4.8% (3/62) | 7.0% (13/185) | Không rõ xu hướng |
| `MET driver` | 8.1% (5/62) | 3.8% (7/185) | MET có thể liên quan đáp ứng (n nhỏ) |
| `BRAF driver` | 3.2% (2/62) | 2.7% (5/185) | Tương đương |
| `EGFR: AMP` | 0.0% (0/62) | 5.4% (10/185) | Chỉ xuất hiện ở không đáp ứng |
| `ERBB2: AMP` | 0.0% (0/62) | 3.2% (6/185) | Tương tự EGFR AMP |
| `MET: AMP` | 4.8% (3/62) | 1.1% (2/185) | |
| `RET: AMP` | 1.6% (1/62) | 0.0% (0/185) | n=1, không có ý nghĩa thống kê |

**Nhận xét nổi bật:**
- **EGFR driver** và **EGFR AMP** hoàn toàn vắng mặt ở nhóm đáp ứng → đây là biomarker âm tính mạnh với ICI.
- **STK11** là đột biến phổ biến nhất và có tỷ lệ cao hơn ~2.6× ở nhóm không đáp ứng → phù hợp cơ chế ức chế miễn dịch khối u qua LKB1.

---

## 4. Phân tích đồng đột biến

### 4.1 Số lượng driver trên mỗi bệnh nhân

| # Driver | # Bệnh nhân |
|---|---|
| 0 | 136 (55.1%) |
| 1 | 102 (41.3%) |
| 2 | 9 (3.6%) |

Phần lớn bệnh nhân có 0 hoặc 1 driver mutation. 9 bệnh nhân có 2 driver đồng thời.

### 4.2 Các cặp đồng đột biến đáng chú ý

| Cặp | # Bệnh nhân |
|---|---|
| `EGFR` mut + `EGFR` AMP | 10 – toàn bộ ca EGFR AMP đều có EGFR driver mut |
| `EGFR` + `STK11` | **0** – loại trừ lẫn nhau (EGFR thường Adenocarcinoma không hút thuốc, STK11 thường hút thuốc nhiều) |
| `BRAF` + `ARID1A` | 1 |
| `BRAF` + `STK11` | 2 |
| `ERBB2` driver + `MET` driver | 1 |
| `ERBB2` driver + `ARID1A` | 1 |

---

## 5. Mối liên hệ với pipeline huấn luyện

### 5.1 Cách sử dụng trong notebook

```python
df_genomic = pd.read_parquet(f"{BASE_DB_DIR}/genomic_data_v3.parquet")
df_tmb     = df_genomic[['TMB']]
df_nontmb  = df_genomic.loc[:, ~df_genomic.columns.str.contains("TMB")]

prepare_other_modalities(modality_dict, df_genomic, modality_MASK, 'gen_driver_mut_amp')
prepare_other_modalities(modality_dict, df_nontmb,  modality_MASK, 'gen_driver_non_tmb')
prepare_other_modalities(modality_dict, df_tmb,     modality_MASK, 'gen_driver_tmb')
```

### 5.2 Ba cách dùng genomic trong thí nghiệm

| Tên modality | Đặc trưng | Mô tả |
|---|---|---|
| `gen_driver_mut_amp` | Tất cả 11 cột | Toàn bộ: driver mutations + amplifications + TMB |
| `gen_driver_non_tmb` | 10 cột (không TMB) | Chỉ driver mutations + amplifications |
| `gen_driver_tmb` | 1 cột (`TMB`) | Chỉ gánh nặng đột biến |

### 5.3 Các mô hình baseline sử dụng genomic

- `train_LR(df_tmb, ...)` → `LR Gen-Only-TMB`
- `train_LR(df_nontmb, ...)` → `LR Gen-No-TMB`
- `train_LR(df_genomic, ...)` → `LR Gen-Combined`
- Ensemble: `LR Gen-Average` = trung bình của TMB-Only và Combined

---

## 6. Vấn đề và lưu ý

| Vấn đề | Chi tiết |
|---|---|
| **Mismatch index** | `main_index` là `P-XXXXXXX` (DMP patient ID) – không khớp trực tiếp với `main_index` (`R-XXX`) trong bảng lâm sàng; cần join qua `dmp_pt_id` |
| **Tập con** | Chỉ 247/366 bệnh nhân lâm sàng có dữ liệu genomic (67.5%) |
| **Mất cân bằng nhãn** | Sau khi join: label=0: 62 (25.1%), label=1: 185 (74.9%) – tương tự tỷ lệ toàn bộ cohort |
| **RET: AMP** | Chỉ 1 ca dương tính – gần như không có tín hiệu thống kê |
| **TMB đơn vị** | Giá trị là số đột biến nguyên (integer), không phải mut/Mb chuẩn hoá; cần xác nhận nếu so sánh với ngưỡng lâm sàng 10 mut/Mb |
| **Tiền xử lý** | File đã binarized sẵn từ CSV gốc `genomic_inventory_v3.csv` qua bộ lọc `df.columns.str.contains("bin\|TMB")` |
