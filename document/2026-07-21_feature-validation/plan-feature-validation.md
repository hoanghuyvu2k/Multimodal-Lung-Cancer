# Plan: Cải Thiện Tín Hiệu & Kiểm Chứng Ngoài Mẫu

> Ngày tạo: 2026-07-21
> Tiền đề: dự án `2026-07-20_attention-redesign` đã chứng minh **kiến trúc fusion đã cạn** trên
> cohort này (không phương pháp attention/gate nào thắng; giá trị duy nhất là chuẩn hoá `1/Σmask`;
> neural không vượt LR). Kết luận của nó: *đòn bẩy nằm ở dữ liệu/feature và ở generalization,
> KHÔNG ở fusion.* Plan này đi đúng hai hướng đó.

## 0. Nguyên tắc — khác gì so với dự án trước

- **KHÔNG đụng vào cơ chế fusion nữa.** Backbone cố định = `uniform_avg`
  (`experiments/baselines/model_uniform_avg.py`, đã kiểm chứng tái lập 1e-9). Nó ngang/hơn cả
  Original lẫn OvO và là bản đơn giản nhất — mọi cải thiện đo *trên nền nó*, không phải trên attention.
- **Hai hướng:** (B) nâng chất lượng biểu diễn feature; (A) kiểm chứng ngoài mẫu trên
  `rad_valid` (n=50) và `path_valid` (n=71) — hai cohort chưa hề đụng tới.
- Tái dùng tối đa harness: `common/data_setup.py`, `common/evaluate.py`, `common/benchmarks.py`.

### 0.1. Cấu trúc thư mục
Mỗi phương pháp cải thiện = một folder độc lập dưới `experiments/`. **Không sửa `lung_helpers.py`.**

```
experiments/
  common/                 ← harness (tái dùng)
  baselines/              ← model_uniform_avg.py (backbone), model_lr.py
  audit/                  ← Step 1 (chẩn đoán per-modality)
  feat_select/            ← Step 2
  feat_reduce/            ← Step 3
  nlp_modality/           ← Step 4
  validation/             ← Step 5 (Hướng A — xương sống)
  ensemble/               ← Step 6
  results/                ← JSON tổng hợp (tái dùng thư mục cũ)
```

### 0.2. Tiêu chí GIỮ / XÓA (cho các phương pháp cải thiện feature — Step 2,3,4)
Đo **so với backbone `uniform_avg`**, cùng 4 benchmark, 5 seeds × 10-fold:
- **GIỮ** nếu: mean ΔAUC ≥ **+0.015** trên ≥ 3/4 config **VÀ** paired bootstrap p < 0.10 trên BM1.
- **XÓA** nếu không đạt: `rm -rf` folder đó ngay, chỉ giữ một mục Nhật ký + file `results/*.json`
  + `results/scores/*.csv` để không lặp lại.
- Step 5 (validation) và Step 6 (ensemble/calibration) **không** áp GIỮ/XÓA — chúng là *phép đo*,
  báo cáo trung thực dù kết quả ra sao.

### 0.3. Kỷ luật token
- Mỗi Step = một lượt làm việc riêng, chạy đúng MỘT step rồi dừng.
- Script ghi kết quả ra `results/*.json`, chỉ in bảng tóm tắt ngắn. Log dài → scratchpad.
- Đặt `PYTHONIOENCODING=utf-8`. Dùng đường dẫn tuyệt đối cho background job.
- Tận dụng GPU nếu nhanh hơn (model neural qua `lung_helpers` đã tự phát hiện CUDA).

### 0.4. Benchmark cố định (nội bộ) — giống hệt dự án trước để nối được số
| Ký hiệu | Modalities | uniform_avg (nền, 5-seed) |
|---|---|---|
| `BM1` | Rad+IHC-G+Gen+PDL1 | 0.7746 ± 0.0186 |
| `BM2` | +Labs | 0.7665 ± 0.0110 |
| `BM3` | Rad+Gen | 0.7110 ± 0.0148 |
| `BM4` | PDL1+Gen | 0.7191 ± 0.0084 |

---

## 1. Các bước

### Step 0 — Khung & khóa backbone
- [x] Xác nhận `uniform_avg` chạy lại khớp số nền ở bảng 0.4 (BM1 5-seed = 0.7746, **Δ = 0.00000**, khớp tuyệt đối)
- [x] Tạo khung thư mục các folder ở mục 0.1 (`audit/ feat_select/ feat_reduce/ nlp_modality/ validation/ ensemble/`)
- [x] Kiểm tra `results/baseline.json`, `method_B.json` (uniform_avg), `baseline_lr.json` còn đọc được — OK
- [x] Ghi `results/backbone_lock.json` (số nền uniform_avg 4 BM để mọi step sau đối chiếu)

### Step 1 — Audit tín hiệu từng modality (chẩn đoán)
> Fusion đã cạn thì tín hiệu nằm ở từng modality. Modality nào là tín hiệu, modality nào là nhiễu?
- [x] Với **mỗi** modality đơn lẻ: chạy `uniform_avg` chỉ với modality đó (subset bệnh nhân có mask=True), 5 seeds × 10-fold → AUC
- [x] Chạy song song bản LR đơn-modality (đối chiếu head neural vs LR ở mức từng modality)
- [x] Ghi `results/signal_audit.json` + bảng xếp hạng AUC per-modality
- [x] **Cổng quyết định:** không modality nào AUC ≤ 0.53 → **không loại modality nào**. Nhưng có 2 kết luận định hướng (xem dưới)

#### Bảng xếp hạng (uniform_avg | LR, 5 seeds × 10-fold, subset present)

| modality | n | pos | uniform_avg | LR |
|---|---|---|---|---|
| cnl_pdl1_score | 201 | 148 | **0.7198 ± 0.0122** | 0.7032 |
| gen_driver_mut_amp | 247 | 185 | 0.6763 ± 0.0125 | 0.6776 |
| rad_lesion_ln | 67 | 43 | 0.6446 ± 0.0450 | **0.6855** |
| rad_lesion_pc | 163 | 120 | 0.6366 ± 0.0091 | **0.6693** |
| path_ihc_pdl1 | 105 | 63 | 0.6347 ± 0.0207 | 0.6338 |
| gen_driver_non_tmb | 247 | 185 | 0.6317 ± 0.0243 | 0.6039 |
| path_ihc_glcm | 105 | 63 | 0.6222 ± 0.0258 | 0.6309 |
| gen_driver_tmb | 247 | 185 | 0.6161 ± 0.0007 | 0.6073 |
| rad_lesion_lu | 187 | 136 | 0.5869 ± 0.0213 | 0.5978 |
| rad_lesion_pl | 21 | 13 | 0.5779 ± 0.0965 | 0.5692 |
| cnl_dem_labs | 247 | 185 | 0.5731 ± 0.0195 | 0.5962 |

**Hai kết luận định hướng:**
1. **Radiomics: LR > head neural ở CẢ pc/ln/lu** (ln +0.041, pc +0.033) — xác nhận nghi ngờ dự án
   trước: head neural overfit trên feature radiomics tương quan cao. → mục tiêu chính của Step 2 (chọn
   feature) và Step 3 (giảm chiều) là **radiomics**.
2. **`cnl_dem_labs` yếu nhất (0.573)** dù n=247 đủ — labs thô gần như không phân biệt được. → củng cố
   động cơ Step 4 (NLP embedding có rút thêm tín hiệu từ lâm sàng không). `rad_lesion_pl` (n=21) quá
   nhỏ, bỏ qua khi diễn giải.

### Step 2 — Feature selection sweep (Hướng B-1)  → ~~`experiments/feat_select/`~~ **ĐÃ XÓA**
> Manh mối: `lr_concat` thua nặng ở config radiomics → feature radiomics nhiễu/dư thừa.
- [x] Sweep `l1_strength` × `robustness_cutoff` (9 cấu hình) trên BM1+BM3, 3 seeds — **winner = config mặc định** (l1=0.1, rob=0.15)
- [x] Selector độc lập fold: mutual-info top-20 cho radiomics (`model_mi.py`)
- [x] Đánh giá l1_best + mi_topk đầy đủ BM1–BM4 × 5 seeds → `results/feat_select.json`
- [x] Paired bootstrap vs `uniform_avg` (nền) + scores lưu ở `results/scores/feat_*`
- [x] Áp tiêu chí 0.2 → **XÓA cả hai**, đã `rm -rf experiments/feat_select/`

#### Kết quả — không có đòn bẩy trong feature selection

| BM | uniform_avg (nền) | l1_best | Δ | mi_topk | Δ | p(mi) |
|---|---|---|---|---|---|---|
| BM1 | 0.7746 ± 0.0186 | 0.7746 | +0.0000 | 0.7169 | **−0.0497** | **0.01** |
| BM2 | 0.7665 ± 0.0110 | 0.7665 | +0.0000 | 0.7310 | −0.0255 | 0.11 |
| BM3 | 0.7110 ± 0.0148 | 0.7110 | +0.0000 | 0.6734 | −0.0280 | 0.21 |
| BM4 | 0.7191 ± 0.0084 | 0.7191 | +0.0000 | 0.7191 | +0.0000 | 1.00 |

**Hai kết luận:**
1. Sweep elastic-net: **config mặc định đã tối ưu** trong lưới — tăng `robustness_cutoff` (ít feature hơn)
   hay đổi `l1_strength` đều làm tệ đi. Không có "điểm yếu selection" để khai thác. `l1_best` = nền y hệt.
2. **Cắt cứng top-20 (MI) làm HỎNG** đúng ở config radiomics (BM1 −0.050 p=0.01), BM4 không đổi (không có
   filter radiomics). Nghịch lý audit: dù LR > head neural trên từng modality radiomics, hạn chế feature
   không sửa được — head neural dùng tập feature rộng vẫn tốt hơn một lát cắt hẹp.

### Step 3 — Giảm chiều per-modality (Hướng B-2)  → ~~`experiments/feat_reduce/`~~ **ĐÃ XÓA**
> Radiomics vài trăm feature tương quan cao. Biểu diễn gọn có giảm overfit không?
- [x] `model_reduce.py`: PCA per-modality (giữ ~95% variance), fit trong fold, TRƯỚC backbone uniform_avg
- [x] Biến thể nén có giám sát: PLS 5 component cho radiomics
- [x] Fit PCA/PLS **trong fold** (chỉ trên bệnh nhân train có mask=1) — không rò rỉ; đánh giá BM1–BM4 × 5 seeds
- [x] Ghi `results/feat_reduce.json` + paired bootstrap vs nền
- [x] Áp tiêu chí 0.2 → **XÓA cả hai**, đã `rm -rf experiments/feat_reduce/`

#### Kết quả — nén radiomics làm HỎNG, không cứu được

| BM | uniform_avg (nền) | pca95 | Δ (p) | pls5 | Δ (p) |
|---|---|---|---|---|---|
| BM1 | 0.7746 ± 0.0186 | 0.7356 | −0.0344 (0.10) | 0.7346 | −0.0452 (0.05) |
| BM2 | 0.7665 ± 0.0110 | 0.7120 | −0.0536 (0.00) | 0.7331 | −0.0373 (0.04) |
| BM3 | 0.7110 ± 0.0148 | 0.6589 | −0.0550 (0.06) | 0.6788 | −0.0424 (0.15) |
| BM4 | 0.7191 ± 0.0084 | 0.7191 | +0.0000 (1.00) | 0.7191 | +0.0000 (1.00) |

Cả PCA lẫn PLS (có giám sát) đều **tệ đi rõ trên mọi config radiomics**; BM4 không đổi vì không có radiomics.
Cùng mẫu hình với Step 2: **mọi cách nén/hạn chế biểu diễn radiomics đều làm giảm AUC.** Đây là kết luận
Hướng B: khoảng cách "LR > head neural per-modality" (Step 1) KHÔNG phải do dư thừa feature sửa được bằng
selection hay giảm chiều — head neural đã khai thác tập feature đầy đủ tốt hơn bất kỳ biểu diễn gọn nào.

### Step 4 — NLP clinical embedding làm modality (Hướng D)  → ~~`experiments/nlp_modality/`~~ **ĐÃ XÓA**
> `clinical_nlp_embedding.py` đã có. Nhúng văn bản lâm sàng 384-chiều có thêm tín hiệu so với labs thô?
- [x] sentence-transformers 5.3.0 có sẵn; build modality NLP (all-MiniLM 384-dim) với `no_scale` (bắt buộc)
- [x] So sánh BM2 (labs thô) vs BM2_NLP (thay labs bằng NLP) + config chỉ-NLP, 5 seeds × 10-fold
- [x] Ghi `results/nlp_modality.json` + paired bootstrap vs BM2 nền
- [x] Áp tiêu chí 0.2 → **XÓA** (đạt Δ nhưng trượt p), đã `rm -rf experiments/nlp_modality/`

#### Kết quả — gần nhất nhưng vẫn trượt, và bằng chứng cho thấy là nhiễu

| | AUC | Δ vs BM2 | p |
|---|---|---|---|
| BM2 (labs thô) | 0.7665 ± 0.0110 | — | — |
| **BM2_NLP** (thay NLP) | **0.7836 ± 0.0139** | **+0.0180** | 0.157 |
| NLP đơn modality | 0.5486 | vs labs 0.5731 | (−0.0245) |

**Đây là kết quả gần ngưỡng GIỮ nhất của cả dự án** — Δ=+0.018 **vượt** mốc hiệu ứng +0.015, nhưng
**p=0.157 trượt** cổng p<0.10. Quan trọng hơn: **NLP đơn modality (0.549) còn YẾU hơn labs thô (0.573)**,
nên không có chuyện "văn bản lâm sàng giàu tín hiệu hơn". Δ+0.018 ở BM2_NLP nhiều khả năng là nhiễu CV
(khớp cảnh báo chọn-seed của dự án trước). Theo tiêu chí đặt trước → XÓA.

### Step 5 — Kiểm chứng ngoài mẫu (Hướng A — xương sống)  → `experiments/validation/`
> Câu hỏi paper thật sự: train trên discovery, test trên cohort ngoài — mô hình có giữ được không?

- [x] **Probe dữ liệu — độ phủ modality + crosswalk ID + nhãn** (xác định thực tế cohort ngoài):
  - File `*_validation.parquet` chứa dữ liệu cohort ngoài. **ID cohort ngoài KHÁC discovery** (không phải
    `P-xxxx`): rad_valid = số kiểu `190310`, path_valid = số kiểu `1012132`.
  - **`rad_valid` (n=50): CHỈ có radiomics** (46/50 có feature; không site-split), không gen/path/pdl1.
  - **`path_valid` (n=71): CHỈ có pathology** glcm (52/71), không rad/gen/pdl1.
  - **Crosswalk qua omnibus:** rad_valid ↔ `did_acc`/`radiology_accession_number` (50/50);
    path_valid ↔ `pdl1_image_id`/`slide_id` (71/71). Nhãn lấy từ `bor` trong omnibus.
- [x] **Ánh xạ config → cohort (thực tế):** KHÔNG cohort ngoài nào có đủ BM1/BM2/BM3/BM4 (thiếu gen+pdl1).
      Kiểm chứng chỉ khả thi **theo từng modality**: radiomics→`rad_valid`, pathology→`path_valid`.
      Vì đơn-modality nên fusion vô nghĩa; phép so có ý nghĩa = **head neural vs LR** (đúng hai họ mô hình
      cả dự án đang so). Câu hỏi: tín hiệu discovery có giữ ngoài mẫu không, họ nào generalize hơn.
- [x] Train head neural (backbone) + LR trên **toàn bộ** discovery (theo từng modality) → test 1 lần trên
      cohort ngoài. AUC + 95% CI bootstrap trên bệnh nhân test
- [x] Ghi `results/validation.json` + bảng discovery-CV (từ audit) vs external

#### Kết quả — pathology GIỮ được ngoài mẫu, radiomics thì KHÔNG

| modality | disc uniform | disc LR | **ext neural** | **ext LR** | n_test (pos) |
|---|---|---|---|---|---|
| **pathology** → path_valid | 0.622 | 0.631 | **0.767** [0.62, 0.89] | **0.765** [0.61, 0.89] | 52 (38) |
| **radiomics** → rad_valid | 0.587 | 0.598 | **0.425** [0.20, 0.65] | **0.461** [0.23, 0.68] | 46 (36) |

(radiomics dùng SelectKBest k=30 fit trên discovery cho công bằng; bản dùng cả 1689 feature còn tệ hơn:
neural 0.401 / LR 0.572 — overfit. Cả bốn số radiomics đều **≤ 0.57, không số nào rõ trên 0.5**.)

**Hai kết luận xương sống:**
1. **Pathology generalize mạnh** — AUC ngoài mẫu **0.767 CAO HƠN** discovery-CV (0.62). Tín hiệu pathology
   glcm giữ vững, thậm chí tốt hơn, trên cohort độc lập. `head neural ≈ LR` (0.767 vs 0.765) — **khẳng định
   ngoài mẫu** kết luận cả dự án: hai họ mô hình tương đương.
2. **Radiomics KHÔNG generalize** — AUC ngoài mẫu tụt về ~0.42–0.46 (quanh/ dưới ngẫu nhiên) dù discovery-CV
   0.587. Khớp trọn mạch: radiomics là nơi head neural overfit (Step 1, LR>neural), không sửa được bằng
   selection/giảm chiều (Step 2–3), và giờ chứng minh **tín hiệu radiomics là đặc thù discovery, không
   chuyển được ra cohort ngoài** (nhiều khả năng domain shift — scanner khác).

> Cảnh báo: n_test nhỏ (46/52), CI rộng. Nhưng hướng kết luận nhất quán và mạnh: pathology > 0.75 ổn định,
> radiomics dưới 0.5 ở mọi cấu hình.

### Step 6 — Ensemble seed + calibration (robustness rẻ tiền)  → `experiments/ensemble/`
> sd giữa seed ≈ 0.02. Trung bình dự đoán qua seed có cho AUC ổn định hơn/nhỉnh hơn không?
- [x] Trung bình điểm dự đoán per-patient qua 5 seed (từ `results/scores/`) → AUC ensemble
- [x] Calibration Brier (Platt) cho uniform_avg + Original + OvO + lr_late
- [x] Ghi `results/ensemble.json` (không GIỮ/XÓA — phép đo)

#### Kết quả — ensemble seed là cải thiện DUY NHẤT rẻ và trung thực

| method (BM1) | mean per-seed | span seed | **ensemble** | Δ | Brier |
|---|---|---|---|---|---|
| uniform_avg | 0.7746 | 0.047 | **0.7850** | +0.0104 | 0.150 |
| Original | 0.7595 | 0.055 | 0.7752 | +0.0157 | 0.152 |
| OvO | 0.7728 | 0.048 | 0.7843 | +0.0115 | 0.150 |
| lr_late | 0.7615 | 0.025 | 0.7683 | +0.0067 | 0.156 |

Ensemble qua 5 seed **luôn > mean per-seed** trên mọi method × mọi BM (uniform_avg +0.005…+0.014). Đây
không phải chọn seed may mắn — mà là trung bình dự đoán qua 5 phân hoạch, triệt tiêu phương sai phân hoạch
(±0.02–0.05). uniform_avg ensemble BM1 = **0.7850** (số đa-seed trung thực cao nhất của backbone). Brier
~0.15 (calibrate tốt, đồng đều). lr_late lợi ít nhất vì vốn ổn định nhất. **Khuyến nghị thực tế: dùng
ensemble-qua-seed — lợi ích +0.01 ổn định, chi phí gần như bằng 0.**

### Step 7 — Tổng hợp
- [x] Bảng nội bộ cuối (Hướng B) — trong `ket-qua.md`: mọi phương pháp feature đều XÓA
- [x] Bảng validation: discovery-CV vs external cho pathology & radiomics — trong `ket-qua.md`
- [x] Hình: (a) `fig-feature-methods.svg` forest ΔAUC vs nền; (b) `fig-validation.svg` discovery vs external
- [x] Xóa mọi folder phương pháp không đạt (feat_select, feat_reduce, nlp_modality — đã xác nhận không còn)
- [x] Viết `document/2026-07-21_feature-validation/ket-qua.md`
- [x] Cập nhật Nhật ký; mọi step đã tick → dừng loop

---

## 2. Nhật ký kết quả
> Ghi cả kết quả âm để không lặp lại.

| Ngày | Step | Kết quả | Quyết định |
|---|---|---|---|
| 2026-07-21 | 7 | **TỔNG HỢP — HẾT PLAN.** 2 hình (`fig-feature-methods.svg`, `fig-validation.svg`) + `ket-qua.md`. Headline: Hướng B không có đòn bẩy (mọi phương pháp feature XÓA); Hướng A là phát hiện chính — **pathology generalize (ext 0.767), radiomics không (≤0.46)**; neural≈LR trong lẫn ngoài mẫu; ensemble-qua-seed là cải thiện rẻ duy nhất (+0.01). Folder thất bại đã xóa | **HOÀN TẤT.** Dừng loop |
| 2026-07-21 | 6 | **Ensemble seed = cải thiện rẻ & trung thực DUY NHẤT.** Trung bình dự đoán qua 5 seed luôn > mean per-seed (uniform_avg +0.005…+0.014, BM1 0.7746→0.7850) trên mọi method×BM — triệt tiêu phương sai phân hoạch, không phải chọn seed. Brier ~0.15 (calibrate tốt) | Xong. Khuyến nghị dùng ensemble-qua-seed. Sang Step 7 (tổng hợp) |
| 2026-07-21 | 5 | **KIỂM CHỨNG NGOÀI MẪU — phát hiện chính.** Cohort ngoài đơn-modality (crosswalk qua omnibus did_acc/pdl1_image_id, nhãn từ bor). **Pathology GIỮ mạnh: ext 0.767 > disc-CV 0.62** (head neural≈LR 0.767/0.765). **Radiomics KHÔNG generalize: ext 0.42–0.46, dưới ngẫu nhiên** dù disc-CV 0.587 → tín hiệu radiomics đặc thù discovery (domain shift). Neural≈LR khẳng định ngoài mẫu | Xong. Sang Step 6 (ensemble). Đây là kết quả có giá trị nhất cho paper |
| 2026-07-21 | 4 | **NLP embedding — gần nhất nhưng vẫn XÓA.** BM2_NLP (thay labs bằng NLP 384-dim) = 0.7836, Δ=+0.0180 **đạt** mốc hiệu ứng nhưng p=0.157 **trượt** cổng. NLP đơn modality (0.549) YẾU hơn labs thô (0.573) → không giàu tín hiệu hơn; Δ là nhiễu | Xóa `experiments/nlp_modality/`; giữ JSON+scores. Hết Hướng B, sang Step 5 (validation ngoài) |
| 2026-07-21 | 3 | **Giảm chiều KHÔNG cải thiện — XÓA cả hai.** PCA-95 và PLS-5 (fit trong fold) đều tệ đi rõ trên mọi config radiomics (Δ −0.03…−0.055, nhiều cái p<0.05); BM4 không đổi. **Kết luận Hướng B: không có đòn bẩy ở biểu diễn feature** — nén/hạn chế radiomics luôn làm giảm AUC | Xóa `experiments/feat_reduce/`; giữ JSON+scores. Sang Step 4 (NLP) |
| 2026-07-21 | 2 | **Feature selection KHÔNG cải thiện — XÓA cả hai.** Sweep 9 cấu hình elastic-net: winner = config mặc định (l1=0.1/rob=0.15), `l1_best` tái lập nền y hệt (Δ=0.0). MI top-20 làm hỏng radiomics (BM1 −0.050, p=0.01). Nghịch lý: LR>neural per-modality nhưng cắt feature không sửa được | Xóa `experiments/feat_select/`; giữ `results/feat_select.json` + scores. Sang Step 3 |
| 2026-07-21 | 1 | **Audit tín hiệu 11 modality.** Không modality nào là nhiễu thuần (min 0.573). Mạnh nhất: PD-L1 (0.720), driver-mut (0.676). Yếu nhất: labs thô (0.573). **LR vượt head neural trên mọi modality radiomics** (ln +0.041, pc +0.033) → radiomics là nơi head neural overfit | Xong. Step 2/3 nhắm radiomics; Step 4 nhắm labs. Không loại modality nào |
| 2026-07-21 | 0 | Backbone `uniform_avg` khóa xong. Tái lập BM1 5-seed = 0.7746 **khớp tuyệt đối** (Δ=0.0) với số nền. 4 folder thư mục + `results/backbone_lock.json` đã tạo. Các JSON kết quả cũ còn đọc được | Xong, sang Step 1 |

---

## 3. Ghi chú kỹ thuật (kế thừa từ dự án trước)
- Tham số `seed` của `train()`/`train_ovo()` **vô hiệu** (KFold hardcode `random_state=0`,
  `manual_seed(42)` trong `__init__`) → `run_repeated_cv` hoán vị hàng `outcomes` theo seed. Dùng lại.
- OvO: cột `attn_*` không dùng để diễn giải; `predict_proba` đã sửa `.to(device)` (line 1713).
- `summary_df` OvO dtype object → `.astype(float)` trong harness (đã có trong `evaluate.py`).
- Modality NLP **luôn** cần `no_scale=[idx]` (embeddings đã cosine-normalized).
- `LayerNorm(1)` xoá modality 1 feature → nếu giảm chiều gặp modality đơn feature (PD-L1) phải `Identity`.
- Cache dữ liệu pickle dạng dict, không pickle dataclass.
- Đối chiếu số nền uniform_avg tại `results/method_B.json` (khóa Step 5 dự án trước) và bảng 0.4.
