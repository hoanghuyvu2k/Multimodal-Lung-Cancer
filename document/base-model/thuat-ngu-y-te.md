# Thuật ngữ y tế trong dự án DyAM

Dự án dự đoán phản ứng điều trị miễn dịch ở bệnh nhân **ung thư phổi không tế bào nhỏ (NSCLC)** đang điều trị bằng **liệu pháp ức chế điểm kiểm soát miễn dịch (ICI)**. Tài liệu này giải thích các thuật ngữ lâm sàng, hình ảnh học, mô bệnh học, và sinh học phân tử xuất hiện trong code và dữ liệu.

---

## Mục lục

1. [Bệnh và bệnh nhân](#1-bệnh-và-bệnh-nhân)
2. [Phân loại điều trị và thuốc](#2-phân-loại-điều-trị-và-thuốc)
3. [Kết quả điều trị (outcomes)](#3-kết-quả-điều-trị-outcomes)
4. [Các biến lâm sàng (clinical features)](#4-các-biến-lâm-sàng-clinical-features)
5. [Hình ảnh học và Radiomics](#5-hình-ảnh-học-và-radiomics)
6. [Giải phẫu bệnh và IHC (Pathology)](#6-giải-phẫu-bệnh-và-ihc-pathology)
7. [Biomarker phân tử (Genomics)](#7-biomarker-phân-tử-genomics)
8. [Phân tầng rủi ro và thống kê sống còn](#8-phân-tầng-rủi-ro-và-thống-kê-sống-còn)

---

## 1. Bệnh và bệnh nhân

### NSCLC — Non-Small Cell Lung Cancer (Ung thư phổi không tế bào nhỏ)

Nhóm ung thư phổi chiếm ~85% tổng số ca. Được phân loại theo mô học:

| Giá trị `histo` trong data | Loại mô học | Tần suất |
|----------------------------|-------------|----------|
| `Adenocarcinoma` | Ung thư biểu mô tuyến — hay gặp nhất, thường ngoại vi phổi | ~70% trong cohort |
| `Squamous` | Ung thư biểu mô vảy — hay gặp ở trung tâm phổi, liên quan hút thuốc | ~15% |
| `Adenosquamous` | Hỗn hợp tuyến + vảy | hiếm |
| `Large cell` | Ung thư tế bào lớn | hiếm |
| `NOS` | Not Otherwise Specified — không phân loại được | — |

Biến `hist_adeno` (0/1) trong data: `1` = Adenocarcinoma, `0` = các loại khác. Dùng như một feature nhị phân trong model.

### Cohort — Discovery và Validation

Dữ liệu được chia thành 2 nhóm:

- **Discovery cohort (~280 bệnh nhân):** Dùng để **huấn luyện và đánh giá nội bộ** bằng 10-fold cross-validation. Model học trên tập này.
- **Validation cohort:** Tập **độc lập hoàn toàn**, dùng để xác nhận model không bị overfitting. Bệnh nhân từ cơ sở khác hoặc giai đoạn sau.

Trong code: `df_cohort[df_cohort['cohort'] == 'discovery']`

### Metastasis — Di căn

Ung thư phổi giai đoạn muộn thường di căn sang các cơ quan khác. Các biến di căn trong data:

| Biến | Cơ quan | Ý nghĩa lâm sàng |
|------|---------|-----------------|
| `brain_mets` | Não | Di căn não — tiên lượng rất xấu, ảnh hưởng quyết định điều trị |
| `liver_mets` | Gan | Di căn gan — gánh nặng khối u cao |
| `pleural_mets` | Màng phổi | Di căn màng phổi — gây tràn dịch |
| `adrenal_mets` | Tuyến thượng thận | Di căn thượng thận — hay gặp ở NSCLC |
| `bone_mets` | Xương | Di căn xương — gây đau, gãy xương bệnh lý |
| `other_visceral_mets` | Cơ quan nội tạng khác | Thận, lách, v.v. |

Tất cả là biến nhị phân (0/1). Được dùng trong `clinical_predictors` và `NLP embedding`.

---

## 2. Phân loại điều trị và thuốc

### ICI — Immune Checkpoint Inhibitor (Thuốc ức chế điểm kiểm soát miễn dịch)

Đây là nhóm thuốc **trung tâm** của toàn bộ dự án. ICI hoạt động bằng cách "mở khóa" hệ miễn dịch để nhận diện và tiêu diệt tế bào ung thư.

**Cơ chế hoạt động:**

Tế bào T (bạch cầu lympho) có thể nhận diện và tiêu diệt tế bào ung thư, nhưng bình thường bị "tắt" bởi các protein điểm kiểm soát (PD-1, PD-L1, CTLA-4) để tránh tự miễn. Tế bào ung thư lợi dụng cơ chế này để "trốn" hệ miễn dịch. ICI chặn các protein đó → hệ miễn dịch hoạt động trở lại.

```
Tế bào T ──[PD-1]──────[PD-L1]── Tế bào ung thư
                 ↑
           ICI chặn kết nối này
           → T cell tấn công ung thư
```

| Biến trong data | Loại thuốc | Ví dụ thuốc |
|----------------|------------|-------------|
| `recieves_pdl1_therapy` | Anti-PD-L1 | Atezolizumab, Durvalumab |
| `recieves_pd1_therapy` | Anti-PD-1 | Pembrolizumab, Nivolumab |
| `recieves_combo_therapy` | Kết hợp ICI + chemo hoặc ICI + ICI | — |

### therapy_line — Bậc điều trị

Số lần điều trị trước đó mà bệnh nhân đã trải qua:

| Giá trị | Ý nghĩa |
|---------|---------|
| `1` | Điều trị lần đầu (first-line) — chưa từng chữa trị hệ thống trước |
| `2` | Bậc 2 — đã thất bại với 1 phác đồ trước |
| `3+` | Bậc 3 trở lên — đã thất bại nhiều lần, khó điều trị hơn |

Trong cohort: phần lớn là bậc 2 (~136/247 bệnh nhân). Bậc điều trị cao hơn thường liên quan đến tiên lượng xấu hơn.

---

## 3. Kết quả điều trị (Outcomes)

### BOR — Best Overall Response (Đáp ứng tốt nhất)

Đánh giá đáp ứng khối u tốt nhất ghi nhận được trong suốt quá trình điều trị, theo tiêu chuẩn **RECIST** (Response Evaluation Criteria In Solid Tumors):

| Mã BOR | Ý nghĩa | `label` |
|--------|---------|---------|
| `1` | CR — Complete Response (Đáp ứng hoàn toàn): khối u biến mất hoàn toàn | 0 |
| `2` | PR — Partial Response (Đáp ứng một phần): khối u thu nhỏ ≥30% | 0 |
| `3` | SD — Stable Disease (Bệnh ổn định): không thu nhỏ, không tiến triển | 1 |
| `4` | POD — Progressive Disease (Bệnh tiến triển): khối u tăng ≥20% hoặc có tổn thương mới | 1 |
| `5–8` | Các trường hợp đặc biệt (tiến triển sớm, tử vong, v.v.) | 1 |

**`label` trong model:**
- `label = 0` → CR hoặc PR → **Có đáp ứng** (bệnh nhân hưởng lợi từ ICI)
- `label = 1` → SD hoặc POD → **Không đáp ứng** (bệnh nhân không có lợi)

Tỷ lệ trong cohort: 273 không đáp ứng / 93 có đáp ứng (~3:1 mất cân bằng lớp — lý do dùng `class_weight='balanced'`).

### PFS — Progression-Free Survival (Thời gian sống không tiến triển)

Thời gian (tính bằng tháng) từ khi bắt đầu điều trị đến khi **bệnh tiến triển** hoặc **bệnh nhân tử vong**.

- **Trung vị PFS trong cohort:** 2.65 tháng
- **Phạm vi:** 0.1 – 59.7 tháng

`pfs_censor`: biến chỉ thị sự kiện trong phân tích sống còn:
- `pfs_censor = 1` → **Sự kiện thực sự xảy ra** (tiến triển hoặc tử vong được ghi nhận)
- `pfs_censor = 0` → **Bị kiểm duyệt (censored)** — bệnh nhân rời nghiên cứu, mất liên lạc, hoặc kết thúc nghiên cứu mà chưa có sự kiện (308/366 bệnh nhân có sự kiện)

### OS — Overall Survival (Thời gian sống toàn bộ)

`os_int`: Thời gian từ điều trị đến **tử vong** (bất kể nguyên nhân). Chỉ có ở ~232 bệnh nhân vì nhiều người còn sống hoặc mất dữ liệu.

---

## 4. Các biến lâm sàng (Clinical Features)

Đây là 13 cột tạo thành modality `cnl_dem_labs` (raw) hoặc `cnl_nlp_embedding` (NLP):

### age — Tuổi bệnh nhân

Tuổi tại thời điểm bắt đầu điều trị ICI. Giá trị trong cohort: phổ biến 50–85 tuổi.

### pack_years — Số gói-năm thuốc lá

Đơn vị đo lường lượng thuốc lá tích lũy:

```
pack_years = (số gói/ngày) × (số năm hút)
Ví dụ: hút 1 gói/ngày trong 30 năm = 30 pack-years
```

Liên quan đến TMB (đột biến do khói thuốc), từ đó liên quan đến đáp ứng ICI. `pack_years = 0` là người không hút thuốc.

### ECOG — Eastern Cooperative Oncology Group Performance Status

Thang điểm đánh giá **khả năng hoạt động** (chức năng thể chất) của bệnh nhân:

| ECOG | Mô tả |
|------|-------|
| 0 | Bình thường hoàn toàn, hoạt động đầy đủ |
| 1 | Có triệu chứng nhẹ nhưng tự chăm sóc được, đi lại được |
| 2 | Nằm nghỉ <50% thời gian ban ngày, tự chăm sóc hạn chế |
| 3 | Nằm nghỉ >50% thời gian ban ngày, cần hỗ trợ |
| 4 | Hoàn toàn phụ thuộc, không thể tự chăm sóc |

Trong cohort: hầu hết là ECOG 0–1 (điều kiện nhận vào ICI thường ≤2). ECOG cao → tiên lượng xấu, dung nạp thuốc kém.

### albumin — Albumin huyết thanh (g/dL)

Protein chính trong máu, phản ánh **tình trạng dinh dưỡng và viêm hệ thống**. Giảm albumin (<3.5 g/dL) là dấu hiệu viêm mạn tính hoặc suy dinh dưỡng — liên quan tiên lượng xấu trong ung thư.

### dNLR — Derived Neutrophil-to-Lymphocyte Ratio

Tỷ lệ bạch cầu trung tính / bạch cầu lympho, đo từ xét nghiệm máu thường quy:

```
dNLR = Neutrophil count / (WBC - Neutrophil count)
     ≈ Neutrophil / Lymphocyte
```

**Ý nghĩa miễn dịch:**
- Neutrophil cao + Lymphocyte thấp → viêm mạnh, miễn dịch chống khối u yếu → đáp ứng ICI kém
- dNLR thấp → cân bằng miễn dịch tốt hơn → đáp ứng ICI tốt hơn

Ngưỡng thường dùng trong nghiên cứu: dNLR > 3 là tiên lượng xấu. Trong cohort: median = 2.5, max = 33.1.

### tumor_burden — Gánh nặng khối u

Tổng kích thước/thể tích tất cả tổn thương đo được trên CT. Đơn vị không chuẩn hóa (có thể là tổng đường kính mm). Giá trị trong cohort: trung vị ~51, phạm vi 0–287.

Gánh nặng khối u lớn → nhiều tổn thương hoặc khối u lớn → tiên lượng xấu hơn, nhưng paradox là TMB cao (đột biến nhiều) đôi khi đi kèm khối u lớn và lại đáp ứng ICI tốt.

### therapy_line — Xem [Bậc điều trị](#therapy_line--bậc-điều-trị) ở mục 2

### site_lung — Vị trí tổn thương phổi

Biến nhị phân: `True` = khối u nguyên phát nằm trong phổi (không phải hạch hoặc di căn xa là vị trí chính được xét). Ảnh hưởng đến lựa chọn phương pháp điều trị bổ trợ.

---

## 5. Hình ảnh học và Radiomics

### CT — Computed Tomography (Chụp cắt lớp vi tính)

Phương pháp chẩn đoán hình ảnh chính trong dự án. CT ngực được dùng để:
1. Xác định vị trí và kích thước khối u
2. Phân tích hình ảnh radiomics (định lượng đặc trưng từ pixel)
3. Đánh giá đáp ứng điều trị theo RECIST

### Radiomics — Trích xuất đặc trưng định lượng từ ảnh

Radiomics là quá trình chuyển ảnh y tế thành **hàng trăm đặc trưng số** bằng thuật toán. Không cần bác sĩ phân tích bằng mắt — máy tính tự trích xuất.

**Các nhóm đặc trưng radiomics trong data:**

| Nhóm | Tiền tố cột | Mô tả |
|------|-------------|-------|
| Shape | `original_shape_*` | Hình dạng 3D: thể tích, diện tích, độ cầu, trục dài/ngắn |
| First-order | `original_firstorder_*` | Phân bố cường độ pixel: mean, median, entropy, kurtosis |
| GLCM | `original_glcm_*` | Gray Level Co-occurrence Matrix — texture (kết cấu bề mặt) |
| GLRLM | `original_glrlm_*` | Gray Level Run Length Matrix — texture theo hướng |
| GLSZM | `original_glszm_*` | Gray Level Size Zone Matrix — vùng đồng nhất |
| Wavelet | `wavelet-*` | Phân tích tần số đa cấp bằng wavelet |
| LoG | `log-sigma-*` | Laplacian of Gaussian — phát hiện cạnh, độ thô/mịn |

**`job_tag` trong radiomics:**
- `filtered-radiomics` — Dữ liệu thực sự dùng để train (sau lọc độ ổn định)
- `unfiltered-radiomics` — Toàn bộ features trước khi lọc
- `pertubation-radiomics` — Dữ liệu nhiễu để kiểm tra độ ổn định (ICC test)

### Lesion Sites — Vị trí tổn thương

Mỗi bệnh nhân có nhiều tổn thương. Code chia theo 3 vị trí giải phẫu:

| Mã | Tên đầy đủ | Ý nghĩa lâm sàng |
|----|-----------|-----------------|
| **PC** | Primary Cancer (Khối u nguyên phát) | Khối u chính trong phổi, nguồn gốc của ung thư |
| **PL** | Pulmonary (Tổn thương phổi khác) | Các nốt/tổn thương phổi khác ngoài khối chính |
| **LN** | Lymph Node (Hạch bạch huyết) | Hạch lympho bị xâm lấn — chỉ điểm di căn vùng |

Trong `decorate_with_site_index()` (lung_helpers.py:516):
- `lesion_index` 1–2 → PC
- `lesion_index` 3–4 → PL
- `lesion_index` 5–6 → LN

Modality tương ứng: `rad_lesion_pc`, `rad_lesion_pl`, `rad_lesion_ln`.

### Robustness / ICC — Độ ổn định của features

**ICC (Intraclass Correlation Coefficient):** Đo mức độ một feature radiomics **tái hiện nhất quán** khi scan lại hoặc thêm nhiễu nhỏ. ICC từ 0 (không ổn định) đến 1 (hoàn toàn ổn định).

Trong code: `robustness_cutoff=0.15` — loại bỏ features có phương sai thay đổi >15% khi perturbation. Đảm bảo chỉ dùng features thực sự đo được từ khối u, không phải nhiễu từ máy scan.

---

## 6. Giải phẫu bệnh và IHC (Pathology)

### IHC — Immunohistochemistry (Hóa mô miễn dịch)

Kỹ thuật nhuộm mô bằng kháng thể đặc hiệu để định vị protein trong mô. Trong dự án dùng để định lượng **PD-L1** trên tiêu bản mô.

**Quy trình:** Lấy mẫu sinh thiết → cắt lát mỏng → nhuộm kháng thể anti-PD-L1 → chụp ảnh → phân tích số.

### PD-L1 — Programmed Death-Ligand 1

Protein "điểm kiểm soát" trên bề mặt tế bào ung thư. Khi kết hợp với PD-1 trên tế bào T → ức chế miễn dịch → ung thư trốn thoát.

**Liên quan đến điều trị:** Bệnh nhân có PD-L1 cao thường đáp ứng tốt hơn với thuốc anti-PD-1/PD-L1.

### TPS — Tumor Proportion Score (Điểm tỷ lệ khối u)

Phần trăm **tế bào ung thư** nhuộm dương tính với PD-L1:

```
TPS = (Số tế bào ung thư PD-L1+) / (Tổng tế bào ung thư) × 100%
```

| TPS | Ý nghĩa lâm sàng |
|-----|-----------------|
| 0% | PD-L1 âm tính — ít có khả năng đáp ứng |
| 1–49% | Dương tính thấp |
| ≥50% | Dương tính cao — tốt nhất để dùng pembrolizumab đơn trị |

Trong data: `Sauter PD-L1 Score` (0–100). `cnl_pdl1_score` là modality dùng score này. Phần lớn bệnh nhân có TPS = 0 (91/201 ca).

### IHC-A và IHC-G — Hai hướng phân tích IHC

| Tên | Nguồn data | Đặc trưng | File |
|-----|-----------|-----------|------|
| **IHC-A** (IHC Antibody) | Cường độ pixel kênh PD-L1 | `original_pixels_channel_1_*` — cường độ nhuộm trung bình | `lung_pathology_pdl1_glcm_v3.parquet` |
| **IHC-G** (IHC GLCM) | Kết cấu texture từ ảnh IHC | `original_glcm_*` — pattern không gian của nhuộm | `lung_pathology_pdl1_glcm_v3.parquet` |

**IHC-A** đo "mức độ" PD-L1 (có nhiều hay ít). **IHC-G** đo "cách sắp xếp" PD-L1 trong mô (phân bố đồng đều hay cụm). Hai thông tin bổ sung nhau.

### GLCM — Gray Level Co-occurrence Matrix

Ma trận đồng xuất hiện mức xám — phương pháp phân tích **texture** (kết cấu) của ảnh. Tính tần suất xuất hiện đồng thời của các cặp giá trị pixel ở khoảng cách nhất định.

Các đặc trưng GLCM trong data:
- `Autocorrelation` — tự tương quan cục bộ
- `ClusterProminence` — độ nổi bật của cụm
- `ClusterShade` — bất đối xứng cụm
- `Contrast` — tương phản giữa pixel lân cận
- `Correlation` — tương quan tuyến tính
- `Entropy` — độ hỗn loạn/phức tạp texture

---

## 7. Biomarker phân tử (Genomics)

### TMB — Tumor Mutational Burden (Gánh nặng đột biến khối u)

Số lượng đột biến soma trên 1 megabase DNA của khối u:

```
TMB = (Số đột biến không đồng nghĩa trong khối u) / (Kích thước genome sequenced, Mb)
```

**Tại sao TMB quan trọng với ICI:**
- Đột biến nhiều → protein bất thường nhiều → hệ miễn dịch nhận ra nhiều → đáp ứng ICI tốt hơn
- TMB cao (≥10 mut/Mb) là biomarker FDA-approved cho pembrolizumab

Trong data: `TMB` (liên tục). Modality `gen_driver_tmb` dùng riêng TMB. Modality `gen_driver_mut_amp` dùng TMB + các gene driver.

### Driver Mutations — Đột biến gene driver

Đột biến ở gene kiểm soát tăng trưởng tế bào, **gây ra** ung thư (không chỉ là hành khách):

| Gene | Vai trò | Liên quan ICI |
|------|---------|---------------|
| **EGFR** | Epidermal Growth Factor Receptor — thụ thể tăng trưởng biểu bì | Đột biến EGFR → đáp ứng ICI **kém** (TMB thấp, vi môi trường khối u ức chế miễn dịch) |
| **ERBB2** (HER2) | Họ thụ thể tăng trưởng — khuếch đại → khối u tăng sinh | Ít đáp ứng ICI |
| **BRAF** | Kinase trong con đường MAPK — khuếch tán tín hiệu tăng trưởng | BRAF V600E đôi khi kết hợp với ICI |
| **MET** | Thụ thể yếu tố tăng trưởng gan — khuếch đại hoặc exon 14 skip | Đáp ứng ICI trung bình |
| **STK11** (LKB1) | Kinase ức chế khối u — mất chức năng → kháng ICI | Đột biến STK11 → kháng ICI mạnh, dù TMB cao |
| **ARID1A** | Gene sửa chữa chromatin | Đột biến → MSI-H → đáp ứng ICI tốt |

### AMP — Amplification (Khuếch đại gene)

Tăng số bản copy của gene → protein tương ứng được tổng hợp nhiều hơn → tín hiệu tăng sinh mạnh hơn. Trong data:

- `EGFR: AMP_binarized` — EGFR khuếch đại (0/1)
- `ERBB2: AMP_binarized` — ERBB2/HER2 khuếch đại
- `MET: AMP_binarized` — MET khuếch đại
- `RET: AMP_binarized` — RET khuếch đại

Cột `_binarized` = đã chuyển thành 0/1, không phải số bản copy liên tục.

### MSI / MMR — Microsatellite Instability / Mismatch Repair

Không có trong data trực tiếp nhưng liên quan qua TMB:
- **MSI-High (MSI-H)** hoặc **dMMR** → hệ thống sửa chữa DNA bị hỏng → đột biến tích lũy nhiều → TMB rất cao → đáp ứng ICI tốt nhất

---

## 8. Phân tầng rủi ro và thống kê sống còn

### Kaplan-Meier (KM) — Đường cong sống còn

Phương pháp ước tính **xác suất sống sót** theo thời gian, tính đến bệnh nhân kiểm duyệt (censored). Trục X = thời gian (tháng), trục Y = xác suất còn sống/chưa tiến triển.

**Các loại KM trong dự án:**
- **5B (Binary):** Chia bệnh nhân thành 2 nhóm theo ngưỡng score. Đường nằm trên = nhóm nguy cơ thấp (AI dự đoán đáp ứng tốt).
- **6B (Quantile):** Chia thành 4 nhóm Q1/Q2/Q3/Q4 theo phân vị score — Q1 nguy cơ thấp nhất, Q4 cao nhất.

### Log-rank Test — Kiểm định log-rank

Kiểm định thống kê so sánh **xem hai đường KM có thực sự khác nhau hay không**:

- **H₀ (giả thuyết không):** Hai nhóm có cùng phân phối sống sót
- **p-value:** Xác suất quan sát được sự khác biệt lớn như vậy nếu H₀ đúng
- **p < 0.05:** Khác biệt có ý nghĩa thống kê

Trong dự án dùng **-log₂(p)** (âm logarithm cơ số 2):
- p = 0.05 → -log₂(p) ≈ 4.3
- p = 0.001 → -log₂(p) ≈ 10.0
- p = 1.1×10⁻⁷ (DyAM tốt nhất) → -log₂(p) ≈ 23.1

Giá trị **càng lớn** → sự phân tầng càng mạnh → model càng tốt.

### Cox Proportional Hazards Model — Mô hình hồi quy Cox

Phân tích **độc lập** ảnh hưởng của nhiều biến đến nguy cơ tiến triển bệnh:

```
HR (Hazard Ratio) = nguy cơ tiến triển của nhóm A so với nhóm B
HR > 1 → nhóm A tiến triển nhanh hơn
HR < 1 → nhóm A tiến triển chậm hơn (bảo vệ)
```

Figure trong paper dùng Cox để chứng minh DyAM score là **predictor độc lập** sau khi đã điều chỉnh cho age, albumin, ECOG, PD-L1 TPS — nghĩa là AI học được tín hiệu thực sự, không chỉ phản ánh lại các biến lâm sàng cơ bản.

### AUC-ROC — Area Under the Receiver Operating Characteristic Curve

Đo khả năng **phân biệt** bệnh nhân đáp ứng vs không đáp ứng:

| AUC | Ý nghĩa |
|-----|---------|
| 0.5 | Đoán ngẫu nhiên |
| 0.6–0.7 | Yếu |
| 0.7–0.8 | Tốt |
| 0.8–0.9 | Rất tốt |
| > 0.9 | Xuất sắc |

Trong cohort discovery: DyAM đạt AUC ≈ 0.788, cao hơn tất cả baseline đơn lẻ.

### CI — Confidence Interval (Khoảng tin cậy)

Phạm vi giá trị mà AUC thực sự nằm trong đó với xác suất 95%. Trong code dùng **DeLong method** (`auc_roc_ci()`):

```
AUC = 0.788  [95% CI: 0.731 – 0.845]
```

Nếu CI của hai model không chồng lên nhau → sự khác biệt có ý nghĩa thống kê.

---

## Bảng tra nhanh — Tên biến trong code

| Biến trong code | Thuật ngữ y tế | Đơn vị / Giá trị |
|----------------|----------------|------------------|
| `pfs` | Progression-Free Survival | Tháng |
| `pfs_censor` | Censoring indicator | 1=sự kiện, 0=kiểm duyệt |
| `os_int` | Overall Survival | Tháng |
| `bor` | Best Overall Response | 1=CR, 2=PR, 3=SD, 4=POD |
| `label` | Nhãn nhị phân | 0=đáp ứng (CR/PR), 1=không đáp ứng |
| `ecog` | ECOG Performance Status | 0–4 |
| `albumin` | Albumin huyết thanh | g/dL |
| `dnlr` | Derived NLR | Tỷ số (không đơn vị) |
| `pack_years` | Gói-năm thuốc lá | Gói × Năm |
| `tumor_burden` | Gánh nặng khối u | mm tổng (không chuẩn hóa) |
| `therapy_line` | Bậc điều trị | 1, 2, 3, ... |
| `TMB` | Tumor Mutational Burden | Mut/Mb |
| `Sauter PD-L1 Score` | TPS PD-L1 | 0–100% |
| `hist_adeno` | Mô học Adenocarcinoma | 0/1 |
| `site_lung` | Tổn thương tại phổi | True/False |
| `brain_mets` | Di căn não | 0/1 |
| `liver_mets` | Di căn gan | 0/1 |
| `PC` / `rad_lesion_pc` | Khối u nguyên phát | Radiomics features |
| `PL` / `rad_lesion_pl` | Tổn thương phổi khác | Radiomics features |
| `LN` / `rad_lesion_ln` | Hạch bạch huyết | Radiomics features |
