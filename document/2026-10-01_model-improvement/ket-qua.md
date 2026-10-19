# Bốn hướng cải thiện mô hình — kết quả

> 2026-10-01 → 2026-10-03 · Kế hoạch: [ke-hoach.md](ke-hoach.md)
> Mã: `experiments/modelimp/` · Kết quả thô: `experiments/results/modelimp.json`,
> `combat_external.json`

## Con số chốt

| Cấu hình (backbone `uniform_avg`) | AUC 5-seed | + ensemble theo seed |
|---|---|---|
| BM1 Rad+IHC-G+Gen+PDL1 — nền | 0,7730 ± 0,019 | 0,7838 |
| **BM1 + NLP + miss_pattern** | **0,7896 ± 0,011** | **0,7970** |
| Rad+IHC-G+Gen — nền | 0,7510 ± 0,015 | 0,7631 |
| Rad+IHC-G+Gen + NLP + miss_pattern | 0,7871 ± 0,014 | 0,7956 |

**0,7970** là con số cao nhất dự án từng đạt (mốc cũ 0,7832), và xấp xỉ mức công bố của bài gốc
trên cùng bộ dữ liệu (0,80). Chỉ gồm những thành phần đã chứng minh được: mã hoá NLP cho dữ liệu
lâm sàng, chỉ báo mẫu khuyết, và ensemble theo seed.

---

## Tổng kết năm hướng

| Hướng | Kết quả | Kết luận |
|---|---|---|
| **2 — ensemble theo seed** | **+0,0091**, thắng **19/21** tổ hợp | ✅ GIỮ — cải thiện duy nhất vừa rộng vừa bền, lại miễn phí |
| 4 — dò siêu tham số | BM1/3-seed: +0,008 → BM1/5-seed: **−0,0035** | ⚠️ Không đứng vững ở tổ hợp chủ lực |
| 3 — pairwise loss | +0,0045 (BM1), +0,0016 (BM4); gộp với tham số mới thì triệt tiêu | ⚠️ Quá nhỏ |
| 3 — modality dropout | −0,0028 (BM1), +0,0035 (BM4) | ❌ Không giúp |
| 1 — ComBat radiomics | nội bộ −0,023/−0,029; ngoại kiểm 0,42 → 0,51 | ❌ Âm |

---

## Hướng 2 — ensemble theo seed (GIỮ)

Trung bình **điểm từng bệnh nhân** qua 5 seed rồi mới tính AUC, thay vì lấy trung bình 5 giá trị AUC.
Tính được trực tiếp từ điểm đã lưu, **không phải train lại**.

| Tổ hợp | 1 seed (TB) | Ensemble | Δ |
|---|---|---|---|
| Rad | 0,6661 | 0,6925 | +0,0264 |
| Rad+IHC-A | 0,6836 | 0,7051 | +0,0215 |
| Rad+IHC-G | 0,7089 | 0,7298 | +0,0209 |
| Rad+Gen | 0,7160 | 0,7291 | +0,0132 |
| BM1 | 0,7730 | 0,7838 | +0,0108 |

Cải thiện 19/21 tổ hợp, trung bình +0,0091. Lợi ích lớn nhất rơi đúng vào tổ hợp radiomics — nơi
phương sai giữa các seed lớn nhất, nên gộp lại giảm nhiễu được nhiều nhất.

## Hướng 4 — dò siêu tham số (KHÔNG ĐỨNG VỮNG)

Dò theo từng trục trên BM1 với 3 seed cho kết quả rất hứa hẹn — cả ba trục đều cải thiện khi đi về
cùng một phía "huấn luyện nhẹ tay hơn":

| Cấu hình | AUC (3 seed) | Δ |
|---|---|---|
| epochs = 75 | 0,7903 | +0,0080 |
| lr = 0,005 | 0,7897 | +0,0073 |
| alpha = 0,0001 | 0,7885 | +0,0062 |
| alpha = 0,01 | 0,7384 | **−0,0440** |

Nhưng khi kiểm lại bằng **5 seed trên 6 tổ hợp**, lợi ích ở BM1 biến mất:

| Tổ hợp | nền | tham số mới | Δ |
|---|---|---|---|
| **BM1** | 0,7730 | 0,7695 | **−0,0035** |
| PDL1+Gen | 0,7191 | 0,7517 | **+0,0326** |
| IHC-G+Gen | 0,7184 | 0,7410 | **+0,0226** |
| Rad+Gen | 0,7160 | 0,7241 | +0,0082 |
| Rad+IHC-G+Gen | 0,7510 | 0,7591 | +0,0081 |
| Rad+IHC-A+Gen+PDL1 | 0,7547 | 0,7468 | −0,0079 |

**Bài học phương pháp:** +0,008 đo trên 3 seed của một tổ hợp là nhiễu, không phải tín hiệu. Dò tham
số rồi chọn theo chính tổ hợp đã dò là cách chắc chắn tự lừa mình.

Nhưng bức tranh không hoàn toàn âm: cấu hình nhẹ tay giúp **rất nhiều** ở tổ hợp ít chiều
(PDL1+Gen +0,033; IHC-G+Gen +0,023) và hại ở tổ hợp giàu radiomics. Lại đúng mẫu hình "radiomics
cư xử khác phần còn lại" đã gặp ở TabPFN, cooperative learning và XGBoost.

Một con số đáng giữ cho bài báo: **alpha = 0,01 làm AUC sụp 0,044** — mô hình rất nhạy với mức phạt L1
trên trọng số attention.

## Hướng 3 — pairwise loss + modality dropout (QUÁ NHỎ / KHÔNG GIÚP)

Pairwise loss (tối ưu trực tiếp AUC qua 62 × 185 = 11.470 cặp): +0,0045 ở BM1, +0,0016 ở BM4.
Khi gộp với cấu hình tham số mới thì phần lợi ích này biến mất (cột `hp` và `hp_pairwise` gần trùng).

Modality dropout (che ngẫu nhiên nhánh đang có khi train): −0,0028 ở BM1, +0,0035 ở BM4. Giả thuyết
"mô hình cần được huấn luyện có chủ đích cho việc thiếu nhánh" **không được số liệu ủng hộ**.

## Hướng 1 — ComBat hài hoà radiomics theo máy chụp (ÂM)

Hài hoà 1.609 đặc trưng radiomics trên 3.996 dòng; batch: unknown 212 / revolution 69 / lightspeed 52.

| | AUC nội bộ |
|---|---|
| BM1 gốc → ComBat | 0,7730 → 0,7501 (**−0,0228**) |
| Rad+Gen gốc → ComBat | 0,7160 → 0,6869 (**−0,0291**) |

**Ngoại kiểm trên `rad_valid`** (phép thử quyết định — xem `combat_external.py`):

| Biến thể | k=30 neural | k=30 LR | full neural | full LR |
|---|---|---|---|---|
| gốc | 0,425 | 0,461 | 0,401 | 0,569 |
| ComBat theo máy chụp | 0,422 | 0,475 | 0,556 | 0,458 |
| ComBat theo cohort | 0,514 | 0,511 | 0,497 | 0,581 |

ComBat theo cohort kéo radiomics từ "kém hơn ngẫu nhiên" lên **đúng mức ngẫu nhiên** (0,51, CI
[0,27–0,74]) — không phải cải thiện có ý nghĩa.

**Giá trị của kết quả âm này:** nó **loại trừ** giả thuyết đơn giản nhất cho việc radiomics không
khái quát được. Nếu nguyên nhân là hiệu ứng máy chụp thì khử nó phải làm ngoại kiểm nhích lên rõ rệt;
ở đây không. Nguyên nhân khả dĩ hơn nằm ở khác biệt cách chọn/phân vùng tổn thương giữa hai cohort,
hoặc tín hiệu radiomics vốn không chuyển được. Đáng đưa vào phần thảo luận.

---

## Sự cố kỹ thuật đã xử lý

1. **Cache dùng chung bị nhiễm.** `load_all(use_cache=False)` GHI ĐÈ file cache 98 MB dùng chung.
   Khi `load_harmonized()` gọi nó, dữ liệu đã ComBat ngấm vào cache → lần chạy sau, nhánh "gốc" đọc
   ra chính dữ liệu đã hài hoà (Δ = 0,0000 tuyệt đối, nền tụt 0,7730 → 0,7501). Đã sửa: ComBat ghi
   vào file cache riêng, và đã xoá + dựng lại cache sạch. Dòng thời gian cho thấy hướng 2, 3, 4 chạy
   **trước** khi cache bị nhiễm nên không bị ảnh hưởng.
2. **ComBat hài hoà cả `lesion_index`** (vì nó là cột số) → giá trị không còn là 1–6, bước gán vị trí
   tổn thương PC/PL/LN không tìm thấy dòng nào. Đã loại cột siêu dữ liệu khỏi ComBat.
3. **NaN trong điểm đã lưu** (seed nào bỏ fold 0-feature thì bệnh nhân không được chấm) làm
   `roc_auc_score` văng lỗi. Đã xử lý bỏ qua NaN khi tính AUC và khi gộp ensemble.
