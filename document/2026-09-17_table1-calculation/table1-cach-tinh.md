# Table 1 — Cách tính và nguồn dữ liệu từng dòng

> **Bảng:** Table 1 *"Patient and cohort characteristics"* của bài báo
> **File bảng:** `paper/tables/table1_patients.tex` (sinh tự động — **không sửa tay**)
> **Script sinh bảng:** `paper/make_table1.py`
> **Cập nhật tài liệu:** 2026-09-17

---

## 1. Tổng quan

Table 1 mô tả đặc điểm bệnh nhân của 3 cohort:

| Cohort | n | Vai trò |
|---|---|---|
| `discovery` | 247 | Tập huấn luyện — dùng cho 10-fold cross-validation |
| `rad_valid` | 50 | Tập kiểm định ngoài cho **radiomics** (CT) |
| `path_valid` | 71 | Tập kiểm định ngoài cho **pathology** (PD-L1 IHC) |

Bảng có 2 cột số: **Discovery** và **Validation** (cột Validation ghi `rad_valid; path_valid`,
ngăn cách bằng dấu chấm phẩy).

### Cách chạy

Chạy từ thư mục `code/`:

```bash
python paper/make_table1.py            # tính số liệu + ghi đè paper/tables/table1_patients.tex
python paper/make_table1.py --dry-run  # chỉ in kết quả ra màn hình, không ghi file
```

Sau khi sinh lại bảng, phải chép số tương ứng sang `paper/word_export/build_manuscript_docx.py`
(bản Word), rồi compile lại PDF.

---

## 2. Các file dữ liệu được dùng

Tất cả nằm trong `D:\code\master\doan\datasets\` (tức `../datasets/` tính từ `code/`).

| # | File | Kích thước | Dùng cho dòng nào |
|---|---|---|---|
| **F1** | `final_cohort_listing.csv` | 368 dòng × 2 cột | Xác định bệnh nhân thuộc cohort nào |
| **F2** | `18193mskmindprojectm-omnibusinventory_data_2021-12-20_1540-with-tb-and-scanner.csv` | 366 dòng × 53 cột | **Toàn bộ dòng nhân khẩu học, mô học, ECOG, ICI, đáp ứng, PFS** |
| **F3** | `lung_radiomics_spacing1.0_mirpon_window1350.250_allimagetypes_bw20.parquet` | — | Modality: CT radiomics (discovery) |
| **F4** | `lung_pathology_pdl1_glcm_v3.parquet` | — | Modality: Pathology IHC (discovery) |
| **F5** | `genomic_data_v3.parquet` | — | Modality: Genomics (discovery) |
| **F6** | `pdl1_score.parquet` | — | Modality: PD-L1 TPS (discovery) |
| **F7** | `lung_radiomics_spacing1.0_mirpon_window1350.250_allimagetypes_bw20_validation.parquet` | 1176 dòng | Modality: CT radiomics (rad_valid) |
| **F8** | `lung_pathology_pdl1_glcm_v3_validation.parquet` | 52 dòng × 150 cột | Modality: Pathology IHC (path_valid) |

> F3–F6 không được script đọc trực tiếp: chúng được nạp qua
> `experiments/common/data_setup.py::load_all()`, tức **đúng pipeline mà mô hình dùng khi
> huấn luyện**. Nhờ vậy số "modality availability" trong bảng khớp chính xác với số bệnh
> nhân thực sự đưa vào mô hình.

---

## 3. Bước 1 — Xác định bệnh nhân của từng cohort (join)

**Code:** `load_cohorts()` — `make_table1.py` dòng 68.

File F2 (omnibus) chứa dữ liệu lâm sàng của cả 3 cohort, nhưng **mỗi cohort phải join bằng
một cột ID khác nhau**. Cột `main_index` trong F1 mang ý nghĩa khác nhau tùy cohort:

| Cohort | `main_index` trong F1 là gì | Join với cột nào của F2 | Kết quả |
|---|---|---|---|
| `discovery` | Mã bệnh nhân dạng `P-0013653` | `dmp_pt_id` | khớp 247/247 |
| `rad_valid` | Mã chụp CT dạng `190868` | `radiology_accession_number` (dự phòng: `did_acc`) | khớp 50/50 |
| `path_valid` | Mã slide dạng `3369788` | `pdl1_image_id` (dự phòng: `slide_id`, `pdl1_acc`) | khớp 71/71 |

Cách làm: lấy tập ID của cohort từ F1 → lọc các dòng F2 có giá trị cột join nằm trong tập đó.
Trước khi so sánh, bỏ đuôi `.0` (pandas đọc số nguyên có NaN thành float, ví dụ `190868.0`).
Script thử lần lượt các cột dự phòng và chọn cột khớp nhiều dòng nhất.

**Script tự kiểm tra:** nếu số dòng khớp ≠ số bệnh nhân của cohort trong F1, script **dừng
ngay và báo lỗi** — không cho sinh bảng với dữ liệu thiếu.

> ### ⚠️ Bẫy quan trọng nhất
> File F2 có **366 dòng** nhưng cohort discovery chỉ có **247** bệnh nhân. Nếu tính thống kê
> trên toàn bộ F2 mà không lọc, số sẽ trông rất hợp lý nhưng **sai**. Table 1 trước đây (gõ
> tay) đã mắc đúng lỗi này — xem mục 8.
>
> Hàm `get_clinical_table_v2()` trong `lung_helpers.py` chỉ join bằng `dmp_pt_id`, nên chỉ
> lấy được discovery; không thể dùng nó để tính cột Validation.

---

## 4. Bảng tra cứu nhanh — từng dòng lấy từ đâu

| Dòng trong bảng | File | Cột | Công thức | Discovery | rad_valid | path_valid |
|---|---|---|---|---|---|---|
| Cỡ cohort (tiêu đề cột) | F1 + F2 | cột join | số dòng sau join | 247 | 50 | 71 |
| Age, mean (range) | F2 | `age` | mean (min–max) | 66.9 (38–93) | 64.4 (45–86) | 68.0 (30–89) |
| Male sex | F2 | `sex` | đếm `sex == 1` | 113 (45.7%) | 24 (48.0%) | 32 (45.1%) |
| Adenocarcinoma | F2 | `histo` | đếm `== "Adenocarcinoma"` | 184 (74.5%) | 39 (78.0%) | 47 (66.2%) |
| Squamous cell | F2 | `histo` | đếm `== "Squamous"` | 36 (14.6%) | 7 (14.0%) | 13 (18.3%) |
| Other / NOS | F2 | `histo` | n − Adeno − Squamous | 27 (10.9%) | 4 (8.0%) | 11 (15.5%) |
| ECOG 0 | F2 | `ecog` | đếm `== 0` | 28 (11.3%) | 10 (20.0%) | 8 (11.3%) |
| ECOG 1 | F2 | `ecog` | đếm `== 1` | 194 (78.5%) | 39 (78.0%) | 58 (81.7%) |
| ECOG ≥2 | F2 | `ecog` | n − ECOG0 − ECOG1 − thiếu | 25 (10.1%) | 1 (2.0%) | 5 (7.0%) |
| Anti-PD-1 | F2 | `recieves_pd1_therapy` | đếm `== 1` | 197 (79.8%) | không ghi nhận | không ghi nhận |
| Anti-PD-L1 | F2 | `recieves_pdl1_therapy` | đếm `== 1` | 50 (20.2%) | không ghi nhận | không ghi nhận |
| Combination | F2 | `recieves_combo_therapy` | đếm `== 1` | 12 (4.9%) | không ghi nhận | không ghi nhận |
| Response (PR/CR) | F2 | `label` | đếm `== 0` | 62 (25.1%) | 11 (22.0%) | 21 (29.6%) |
| No response (SD/PD) | F2 | `label` | n − Response | 185 (74.9%) | 39 (78.0%) | 50 (70.4%) |
| Events (PFS) | F2 | `pfs_censor` | đếm `== 1` | 209 (84.6%) | 46 (92.0%) | 55 (77.5%) |
| Median PFS (range) | F2 | `pfs` | median thô (min–max) | 2.7 (0.1–49.1) | 2.6 (1.0–59.7) | 2.7 (0.1–28.4) |
| CT radiomics | F3 → mask | `rad_lesion_pc/pl/ln` | có ≥1 trong 3 vị trí | 187 (75.7%) | 46/50 (F7) | — |
| Pathology IHC | F4 → mask | `path_ihc_glcm` | có dữ liệu | 105 (42.5%) | — | 52/71 (F8) |
| Genomics | F5 → mask | `gen_driver_mut_amp` | có dữ liệu | 247 (100.0%) | — | — |
| PD-L1 TPS | F6 → mask | `cnl_pdl1_score` | có dữ liệu | 201 (81.4%) | — | — |
| Clinical labs | F2 → mask | `cnl_dem_labs` | có dòng dữ liệu | 247 (100.0%) | — | — |

Mẫu số của mọi tỷ lệ % là **n của cohort đó** (247 / 50 / 71).

---

## 5. Chi tiết cách tính từng nhóm

**Code chung:** hàm `stats(df)` — `make_table1.py` dòng 127–150. `df` là các dòng F2 thuộc một
cohort (kết quả của Bước 1).

### 5.1 Demographics — Tuổi và giới tính

```python
s['age']  = f'{r1(df.age.mean())} ({g(df.age.min())}--{g(df.age.max())})'   # dòng 130
s['male'] = pct(int((df.sex == 1).sum()), n)                                 # dòng 131
```

- **Tuổi:** trung bình cộng của cột `age`, kèm tuổi nhỏ nhất – lớn nhất trong ngoặc.
  Hàm `g()` bỏ phần `.0` thừa (38.0 → 38).
- **Giới:** cột `sex` chỉ có 2 giá trị: `1` = nam, `2` = nữ. Đếm số dòng `sex == 1`.
  - Kiểm tra: discovery 113 nam + 134 nữ = 247 ✓

### 5.2 Histology — Mô bệnh học

```python
h = df.histo.value_counts()
adeno, squa = int(h.get('Adenocarcinoma', 0)), int(h.get('Squamous', 0))    # dòng 133
s['other'] = pct(n - adeno - squa, n)                                        # dòng 135
```

Cột `histo` là **chuỗi văn bản thô** trong F2 (không phải biến đã mã hóa). Cách tính:

1. Đếm **khớp chính xác** chuỗi `"Adenocarcinoma"` và `"Squamous"`.
2. `Other / NOS` = **phần bù** `n − Adeno − Squamous`, không phải cộng các loại còn lại.
   → Bảo đảm 3 dòng luôn cộng đúng bằng n (100%).

**Giá trị thô trong dữ liệu và kiểm chứng phần bù:**

| Cohort | Adenocarcinoma | Squamous | Các giá trị rơi vào "Other" | Other |
|---|---|---|---|---|
| discovery | 184 | 36 | NOS 18, Large cell 5, Adenosquamous 1, Large Cell 1, Pleomorphic 1, Giant Cell 1 | 27 ✓ |
| rad_valid | 39 | 7 | Large cell 2, NOS 1, squamous/small cell 1 | 4 ✓ |
| path_valid | 47 | 13 | NOS 7, Adenosquamous 2, Pleomorphic 2 | 11 ✓ |

**Vì sao khớp chính xác, không khớp chuỗi con:** dữ liệu có `Adenosquamous` — chuỗi này chứa
cả `adeno` lẫn `squamous`. Nếu viết `'squamous' in x.lower()` thì Adenosquamous sẽ bị đếm nhầm
thành Squamous.

**Quy ước cần biết để giải trình:**
- `squamous/small cell` (1 ca, rad_valid) được xếp vào **Other**, vì là mô học hỗn hợp. Nếu tính
  là Squamous thì rad_valid Squamous sẽ là 8 (16.0%) thay vì 7 (14.0%).
- `Large cell` và `Large Cell` là cùng một loại nhưng khác hoa/thường (cột chưa chuẩn hóa) —
  vô hại ở đây vì cả hai đều vào Other.

### 5.3 Performance status — ECOG

```python
e = df.ecog.value_counts()
e0, e1 = int(e.get(0, 0)), int(e.get(1, 0))                                   # dòng 137
s['ecog2'] = pct(n - e0 - e1 - int(df.ecog.isna().sum()), n)                  # dòng 139
```

- Cột `ecog` có giá trị `0, 1, 2, 3`.
- ECOG 0 và ECOG 1 đếm trực tiếp.
- ECOG ≥2 = phần còn lại **trừ đi số ca thiếu dữ liệu** (hiện = 0).
  - Discovery: 24 ca ECOG 2 + 1 ca ECOG 3 = 25.

### 5.4 ICI therapy — Loại thuốc ức chế miễn dịch

```python
for key, col in [('pd1',   'recieves_pd1_therapy'),
                 ('pdl1t', 'recieves_pdl1_therapy'),
                 ('combo', 'recieves_combo_therapy')]:                         # dòng 140–144
    s[key] = pct(int((df[col] == 1).sum()), n) if col in df and df[col].notna().any() \
             else '---\\textsuperscript{c}'
```

- Lấy từ **3 cột riêng biệt** trong F2 (tên cột viết sai chính tả `recieves` từ dữ liệu gốc —
  giữ nguyên).
- Đếm số dòng có giá trị `== 1` ở từng cột.
- **3 cột là cờ độc lập, không loại trừ nhau** — một bệnh nhân dùng phối hợp có thể mang nhiều cờ.
  Vì vậy 79.8% + 20.2% + 4.9% = 104.9% > 100% là **đúng**, không phải lỗi. Điều này được ghi ở
  chú thích **b** của bảng.
- 2 cohort validation: cả 3 cột **100% trống** (50/50 và 71/71 NaN) — dữ liệu thực sự không
  được ghi nhận trong registry (xác nhận ở `document/data/omnibus-inventory-analysis.md` §1.1),
  nên script tự in `---` kèm chú thích **c** *"not recorded"*.

### 5.5 Treatment outcome — Đáp ứng điều trị

```python
resp = int((df.label == 0).sum())                                             # dòng 145
s['resp'], s['noresp'] = pct(resp, n), pct(n - resp, n)
```

- Cột `label` là biến mục tiêu của mô hình:
  - `label = 0` → **PR/CR** (đáp ứng một phần / hoàn toàn) → dòng *Response*
  - `label = 1` → **SD/PD** (bệnh ổn định / tiến triển) → dòng *No response*
- `label` được dẫn xuất từ cột `bor` (Best Overall Response): PR/CR → 0, SD/POD → 1.
- No response tính bằng phần bù `n − Response`.
- Discovery 62/185 = đúng tỷ lệ mất cân bằng ~1:3 mà mô hình xử lý bằng class-weighting.

### 5.6 Survival (PFS) — Sống không tiến triển

```python
s['events'] = pct(int((df.pfs_censor == 1).sum()), n)                          # dòng 147
s['pfs'] = f'{r1(df.pfs.median())} ({r1(df.pfs.min())}--{r1(df.pfs.max())})'  # dòng 148
```

Dùng 2 cột trong F2:

| Cột | Ý nghĩa |
|---|---|
| `pfs` | Thời gian (tháng) từ lúc bắt đầu ICI đến khi tiến triển hoặc tử vong; nếu chưa xảy ra thì là thời điểm theo dõi cuối |
| `pfs_censor` | **1 = đã quan sát được biến cố** (tiến triển/tử vong); **0 = censored** (hết theo dõi mà chưa có biến cố) |

- **Events, n (%)** = số ca `pfs_censor == 1`. Cho biết bao nhiêu % bệnh nhân đã thực sự
  tiến triển/tử vong → quyết định sức mạnh thống kê của phân tích sống còn.
- **Median PFS** = thời điểm một nửa bệnh nhân đã tiến triển. **Range** = min–max của `pfs`.

> ### ⚠️ Quy ước `pfs_censor` — không được đảo ngược
> `pfs_censor = 1` nghĩa là **CÓ biến cố**, vì code truyền thẳng cột này vào
> `event_observed=` của thư viện lifelines (`lung_helpers.py` dòng 2272–2320).
> Tài liệu `document/data/omnibus-inventory-analysis.md` từng ghi ngược ("1 = censored") và
> đã được đính chính ngày 2026-09-16.

> ### ⚠️ Hạn chế đã biết của dòng Median PFS (CHƯA sửa trong script)
> Script dùng `df.pfs.median()` — **median thô**, gộp cả ca censored. Cách chuẩn trong báo cáo
> ung thư là **median Kaplan–Meier**, vì thời gian của ca censored chỉ là cận dưới của PFS thật.
>
> | Cohort | Median thô (đang in trong bảng) | Median Kaplan–Meier [95% CI] | Số ca censored |
> |---|---|---|---|
> | discovery | 2.7 | 2.7 [2.2–3.6] | 38 (15.4%) |
> | rad_valid | 2.6 | 2.6 [1.9–4.7] | 4 (8.0%) |
> | path_valid | 2.7 | **3.0** [2.1–4.7] | 16 (22.5%) |
>
> Discovery và rad_valid trùng nhau vì hầu hết đều có biến cố. **path_valid lệch** (2.7 → 3.0)
> vì tỷ lệ censor cao hơn. Ngoài ra, **range** hiện lấy cả ca censored, nên giá trị lớn nhất
> (vd. 49.1 ở discovery) là của một người **được theo dõi lâu nhất mà chưa tiến triển**, không
> phải "PFS dài nhất". Báo cáo chuẩn thường thay range bằng 95% CI của median KM.

### 5.7 Modality availability — Tỷ lệ có từng loại dữ liệu

#### Cột Discovery

**Code:** `modality_counts_discovery()` — `make_table1.py` dòng 90–103.

Gọi `load_all()` của `experiments/common/data_setup.py` để lấy `modality_MASK` — bảng 247 dòng
× 11 cột True/False, cho biết bệnh nhân nào có dữ liệu modality nào. **Đây chính là mask mô hình
dùng khi huấn luyện.**

| Dòng | Cột mask | File gốc | Cách đếm | Kết quả |
|---|---|---|---|---|
| CT radiomics (PC/PL/LN) | `rad_lesion_pc`, `rad_lesion_pl`, `rad_lesion_ln` | F3 | Có **ít nhất 1** trong 3 vị trí tổn thương (`.any(axis=1)`) | 187 |
| Pathology IHC | `path_ihc_glcm` | F4 | `True` | 105 |
| Genomics | `gen_driver_mut_amp` | F5 | `True` | 247 |
| PD-L1 TPS | `cnl_pdl1_score` | F6 | `True` | 201 |
| Clinical labs | `cnl_dem_labs` | F2 (13 cột lâm sàng) | `True` | 247 |

Chi tiết từng vị trí radiomics (có trong Supplementary Table S1): PC 163 · PL 21 · LN 67 · hợp
của cả ba = 187. (Tổng 163+21+67 = 251 > 187 vì một bệnh nhân có thể có tổn thương ở nhiều vị trí.)

> **Lưu ý về "Clinical labs 247 (100%)":** con số này nghĩa là **mọi bệnh nhân đều có dòng dữ
> liệu lâm sàng**, không có nghĩa cả 13 biến đều đầy đủ. Thực tế trong discovery có
> `pack_years` thiếu 7 ca và `site_lung` thiếu 1 ca.

#### Cột Validation

**Code:** `modality_counts_validation()` — `make_table1.py` dòng 106–124.

Lấy tập ID của cohort từ F1, giao với **index** (`main_index`) của file validation tương ứng:

| Dòng | File | Công thức | Kết quả |
|---|---|---|---|
| CT radiomics — rad_valid | F7 | số ID của rad_valid có mặt trong F7 | 46/50 (4 ca không có CT segmentation) |
| Pathology IHC — path_valid | F8 | số ID của path_valid có mặt trong F8 | 52/71 (19 ca không có slide) |

Các ô `—` là modality không tồn tại ở cohort đó (rad_valid không có pathology, path_valid không
có radiomics).

---

## 6. Quy ước làm tròn

**Code:** hàm `r1()` — `make_table1.py` dòng 49.

Mọi số thập phân làm tròn **1 chữ số theo kiểu half-up** bằng `Decimal(...).quantize(...,
ROUND_HALF_UP)`.

**Không dùng** `f'{x:.1f}'` vì lỗi biểu diễn nhị phân: median PFS thật của rad_valid là đúng
**2.55**, nhưng trong máy tính nó là `2.5499999…` nên `f'{2.55:.1f}'` cho ra **2.5** (sai), còn
half-up cho **2.6** (đúng quy ước báo cáo y khoa).

---

## 7. Dữ liệu thiếu

Kiểm tra trên 3 cohort, các cột dùng cho Table 1:

| Cột | discovery | rad_valid | path_valid |
|---|---|---|---|
| `age`, `sex`, `histo`, `ecog`, `label`, `pfs`, `pfs_censor` | 0 thiếu | 0 thiếu | 0 thiếu |
| `recieves_pd1/pdl1/combo_therapy` | 0 thiếu | **50/50 thiếu** | **71/71 thiếu** |

→ Hiện tại **không có ca thiếu** ở các dòng số của bảng.

> **Bẫy tiềm ẩn chưa xảy ra:** `value_counts()` mặc định bỏ qua NaN. Nếu sau này cột `histo`
> có ca thiếu, ca đó sẽ **âm thầm bị gộp vào "Other / NOS"** thay vì báo là missing. Tương tự
> với dòng No response (`n − Response`) nếu `label` có NaN. Script hiện chưa có kiểm tra chặn.

---

## 8. Lịch sử — vì sao phải viết script

Trước 2026-09-16, Table 1 được **gõ tay**. Đối chiếu với dữ liệu phát hiện:

| Dòng | Bản gõ tay cũ | Giá trị đúng | Nguyên nhân |
|---|---|---|---|
| Age | 66.8 (30–93) | 66.9 (38–93) | Chép từ thống kê của 366 dòng F2 |
| Male sex | 168 (45.9%) | 113 (45.7%) | 168 là số của 366 dòng — **bất khả thi** với n=247 (168/247 = 68%) |
| Adeno / Squamous / Other | 73.1 / 15.2 / 11.7% | 74.5 / 14.6 / 10.9% | Như trên |
| ECOG 0 / 1 / ≥2 | 12.5 / 78.7 / 8.8% | 11.3 / 78.5 / 10.1% | Như trên |
| Anti-PD-1 / PD-L1 / Combo | 78.8 / 20.0 / 4.9% | 79.8 / 20.2 / 4.9% | Nguồn khác, lại trình bày như 3 nhóm loại trừ nhau |
| Response / No response | ≈62 / ≈185 | 62 / 185 | Số đúng, chỉ bỏ dấu "≈" |

**Gốc lỗi:** bảng *"Đặc điểm lâm sàng của cohort discovery"* trong
`document/data/du-lieu-final-cohort-listing.md` thực chất chứa số của **toàn bộ 366 dòng file
omnibus** nhưng lại dán nhãn là discovery. Table 1 chép thẳng từ bảng đó. Đã sửa cả hai ngày
2026-09-16.

Các phần **đúng từ trước** (được giữ nguyên, script tính lại ra cùng kết quả): toàn bộ cột
Validation, dòng Events PFS, Median PFS, và các dòng Modality availability.

Cùng đợt đối chiếu cũng phát hiện Supplementary Table S1 sai 5/8 dòng (radiomics PC/PL/LN và
pathology — số `71` ở dòng pathology thực chất là cỡ cohort `path_valid`) và Supplementary
Table S2 còn 3 ô `[TO VERIFY]`; cả hai đã được sửa.

---

## 9. Tóm tắt các file liên quan

| File | Vai trò |
|---|---|
| `paper/make_table1.py` | Script tính toán và sinh bảng |
| `paper/tables/table1_patients.tex` | Bảng LaTeX (output của script) |
| `paper/word_export/build_manuscript_docx.py` | Bản Word — số Table 1 được chép thủ công vào đây |
| `experiments/common/data_setup.py` | `load_all()` — nguồn mask modality |
| `lung_helpers.py` dòng 2272–2320 | Chứng cứ quy ước `pfs_censor = 1` là biến cố |
| `document/data/omnibus-inventory-analysis.md` §1.1 | Hướng dẫn cột join cho từng cohort |
| `document/data/du-lieu-final-cohort-listing.md` | Mô tả F1 (đã sửa bảng dán nhãn nhầm) |
| `CLAUDE.md` mục *"Table 1 (patient characteristics) rule"* | Quy tắc: không sửa tay bảng |
