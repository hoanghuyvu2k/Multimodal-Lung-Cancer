# Phân tích dữ liệu: MSK MIND Omnibus Inventory

**File:** `18193mskmindprojectm-omnibusinventory_data_2021-12-20_1540-with-tb-and-scanner.csv`  
**Ngày tạo tài liệu:** 2026-06-01  
**Nguồn:** MSK MIND Project M – bản xuất ngày 2021-12-20

---

## 1. Tổng quan

| Thuộc tính | Giá trị |
|---|---|
| Số hàng (bệnh nhân/scan) | 366 |
| Số cột (đặc trưng) | 53 |
| Số bệnh nhân duy nhất (`dmp_pt_id`) | 332 |
| Số `main_index` duy nhất | 366 |

`main_index` là khóa chính duy nhất cho mỗi hàng (ví dụ `R-304`, `R-308`, …). Một số bệnh nhân (`dmp_pt_id`) xuất hiện nhiều hơn một lần do có nhiều lần scan hoặc nhiều đợt điều trị.

---

## 2. Nhãn mục tiêu (`label`)

| Nhãn | Ý nghĩa | Số bệnh nhân | Tỷ lệ |
|---|---|---|---|
| `0` | PR/CR – đáp ứng điều trị | 93 | 25.4% |
| `1` | SD/POD – không đáp ứng | 273 | 74.6% |

Dữ liệu **mất cân bằng rõ rệt** (~3:1). Các mô hình huấn luyện trên tập này cần dùng class-weighting hoặc kỹ thuật cân bằng nhãn.

### Biến `bor` (Best Overall Response – mã hóa thô)

| Mã | Số lượng |
|---|---|
| 4 (SD – Stable Disease) | 141 |
| 2 (PR – Partial Response) | 82 |
| 3 (POD – Progressive Disease) | 73 |
| 8 (khác / NE) | 41 |
| 5, 1, 7, 6 | 29 (tổng) |

`label` được dẫn xuất từ `bor`: PR/CR → 0, SD/POD → 1.

---

## 3. Đặc điểm lâm sàng

### 3.1 Nhân khẩu học

| Biến | Giá trị |
|---|---|
| Tuổi trung bình | 66.8 ± 10.2 (30–93 tuổi) |
| Giới tính Nam (`sex=1`) | 168 (45.9%) |
| Giới tính Nữ (`sex=2`) | 198 (54.1%) |

### 3.2 Hút thuốc (`smoking_status`)

| Mã | Ý nghĩa giả định | Số lượng |
|---|---|---|
| 1 | Từng hút | 230 (62.8%) |
| 2 | Đang hút | 86 (23.5%) |
| 0 | Không hút | 50 (13.7%) |

- **Pack-years** (trung bình): 29.0 ± 26.4, dải 0–165 (n=359, thiếu 7 giá trị).

### 3.3 Mô bệnh học (`histo` / `hist_adeno`)

| Loại | Số lượng |
|---|---|
| Adenocarcinoma | 268 (73.2%) |
| Squamous | 56 (15.3%) |
| NOS / Khác | 42 (11.5%) |

Cột `hist_adeno` (bool): `True` = 268, tương ứng hoàn toàn với Adenocarcinoma.

### 3.4 Tình trạng thể lực ECOG

| ECOG | Số lượng |
|---|---|
| 0 | 46 (12.6%) |
| 1 | 289 (78.9%) |
| 2 | 29 (7.9%) |
| 3 | 2 (0.5%) |

Phần lớn bệnh nhân có ECOG 1 – đủ điều kiện tham gia thử nghiệm lâm sàng.

### 3.5 Di căn

| Vị trí | Có di căn | Không có |
|---|---|---|
| Gan (`liver_mets`) | 64 (25.9%) | 183 | (n=247)
| Não (`brain_mets`) | 68 (27.5%) | 179 | (n=247)

---

## 4. Các biến sinh học và điều trị

### 4.1 PD-L1 (`clinical_pdl1_score`)

| Nhóm | n | Trung bình | Trung vị |
|---|---|---|---|
| Đáp ứng (`label=0`) | 82 | 57.4% | 70.0% |
| Không đáp ứng (`label=1`) | 235 | 26.0% | 1.0% |

PD-L1 cao liên quan rõ rệt với đáp ứng điều trị. Thiếu 49 giá trị (13.4%).

Cột `js_pdl1_score` (đọc từ HALO/AI): trung bình 28.6 ± 36.4, thiếu 165 giá trị (45.1%).

### 4.2 Loại liệu pháp miễn dịch

| Loại | Số bệnh nhân dùng |
|---|---|
| Anti-PD-L1 (`recieves_pdl1_therapy`) | 50 |
| Anti-PD-1 (`recieves_pd1_therapy`) | 197 |
| Phối hợp (`recieves_combo_therapy`) | 12 |

Tổng 259 bệnh nhân có thông tin về loại thuốc (thiếu 119 – 32.5%).

### 4.3 Dòng điều trị (`therapy_line`)

| Dòng | Số lượng |
|---|---|
| 1 | 78 (32.0%) |
| 2 | 136 (55.7%) |
| 3+ | 33 (13.5%) |

(n=247; thiếu 119 giá trị)

---

## 5. Kết cục lâm sàng

### 5.1 Thời gian sống không tiến triển (`pfs`, tháng)

| Nhóm | n | Trung bình | Trung vị | Khoảng |
|---|---|---|---|---|
| Đáp ứng (`label=0`) | 93 | 16.3 | 14.8 | 3.5–59.7 |
| Không đáp ứng (`label=1`) | 273 | 3.4 | 1.9 | 0.1–23.5 |

- `pfs_censor`: 1 = bị censored (308 ca), 0 = event xảy ra (58 ca).

### 5.2 Thời gian sống toàn bộ (`os_int`, tháng)

| Nhóm | n | Trung bình | Trung vị |
|---|---|---|---|
| Đáp ứng (`label=0`) | 22 | 18.8 | 17.9 |
| Không đáp ứng (`label=1`) | 210 | 7.6 | 5.2 |

Thiếu nhiều (134 giá trị, 36.6%) – OS chưa đủ trưởng thành tại thời điểm thu thập.

---

## 6. Dữ liệu hình ảnh và scan CT

### 6.1 Thông tin scan

| Biến | n hợp lệ | Trung bình | Khoảng |
|---|---|---|---|
| Exposure (mAs) | 237 | 16.4 | 7–40 |
| TableSpeed (mm/rot) | 237 | 63.1 | 55–110 |
| TotalCollimationWidth (mm) | 237 | 38.8 | 20–40 |
| XRayTubeCurrent_min (mA) | 237 | 149.8 | 118–380 |
| XRayTubeCurrent_max (mA) | 237 | 294.3 | 119–645 |
| XRayTubeCurrent_range (mA) | 237 | 144.5 | 0–444 |

Thiếu 129 giá trị (35.2%) trong toàn bộ nhóm biến scanner.

### 6.2 Loại máy CT (n=237)

| Máy | Số lượng |
|---|---|
| GE Discovery CT750 HD | 108 (45.6%) |
| GE Revolution | 70 (29.5%) |
| GE LightSpeed VCT | 59 (24.9%) |

Ba cột `scanner_discovery`, `scanner_lightspeed`, `scanner_revolution` là **one-hot encoding** cho loại máy.

### 6.3 Phân đoạn khối u

| Biến | n hợp lệ | Ghi chú |
|---|---|---|
| `has_radiology_segmentation` | 246 | 1 = có mask phân đoạn |
| `radiologysegmentationpath` | 189 | Đường dẫn file mask NIfTI |
| `n_lesions` | 189 | TB = 1.76, dải 0–4 |

---

## 7. Dữ liệu mô học / PD-L1 slide

| Biến | n hợp lệ | Mô tả |
|---|---|---|
| `pdl1_tiss_site` | 317 | Vị trí sinh thiết (phổi 45%, hạch 18%, khác) |
| `impact_pdl1_same` | 246 | Flag: PDL1 từ cùng mẫu IMPACT |
| `pdl1_image_id` | 269 | ID slide WSI PD-L1 |
| `slide_id` | 262 | ID slide hình ảnh |
| `halo_tumor_quality` | 235 | Đánh giá chất lượng phân tích HALO |

**Vị trí sinh thiết PD-L1 chính:** phổi (n=142, 44.8%), hạch bạch huyết (n=56, 17.7%), xương (n=19), màng phổi (n=18).

---

## 8. Vấn đề chất lượng dữ liệu

| Vấn đề | Chi tiết |
|---|---|
| Dữ liệu thiếu cao (>30%) | `os_int`, `de_i_id`, `dicom_path`, `tumor_burden`, `therapy_line`, tất cả biến scanner |
| Dữ liệu thiếu trung bình (13–32%) | `clinical_pdl1_score`, `pdl1_tiss_site`, `site_lung`, `pdl1_acc` |
| `js_pdl1_score` thiếu 45.1% | Cần cân nhắc khi dùng làm đặc trưng |
| `n_lesions` thiếu 48.4% | Chỉ có ở tập con có phân đoạn |
| Bệnh nhân lặp | 34 bệnh nhân có >1 hàng (multi-scan hoặc multi-timepoint) |
| `pack_years` kiểu object | Cần chuyển đổi `pd.to_numeric(..., errors='coerce')` trước khi dùng |

---

## 9. Mối liên hệ với pipeline huấn luyện

File này là **bảng lâm sàng trung tâm** của dự án, tương ứng với kết quả của `get_clinical_table_v2()` trong `lung_helpers.py`. Các cột quan trọng:

| Cột trong file | Vai trò trong pipeline |
|---|---|
| `main_index` | Khóa join với tất cả tập dữ liệu khác (radiomics, genomics, PD-L1) |
| `label` | Biến mục tiêu nhị phân cho mô hình |
| `albumin`, `dnlr`, `age`, `ecog`, `sex`, `smoking_status`, `pack_years`, `clinical_pdl1_score`, `pfs`, `os_int`, `n_lesions`, `tumor_burden`, `liver_mets`, `brain_mets`, `therapy_line` | 13 cột lâm sàng số dùng cho NLP embedding (`clinical_nlp_embedding.py`) |
| `scanner_*` | Hiệu chỉnh biến động thiết bị CT trong radiomics |
| `pdl1_tiss_site`, `js_pdl1_score` | Tích hợp dữ liệu PD-L1 từ slide mô học |

**Lưu ý:** Khi dùng các cột lâm sàng làm NLP embedding, luôn truyền `no_scale=[idx]` trong `model_params` để tránh RobustScaler làm hỏng vector cosine-normalized.
