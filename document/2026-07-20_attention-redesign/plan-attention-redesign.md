# Plan: Thiết Kế Lại Cơ Chế Attention

> Ngày tạo: 2026-07-20
> Mục tiêu: xây dựng cơ chế fusion tốt hơn `AttentionMatrix` / `AttentionMatrixOvO` hiện tại,
> với đánh giá đủ mạnh để kết luận không bị nhiễu chi phối.

---

## 0. Bối cảnh — vì sao cần làm

Phân tích code (`lung_helpers.py:976-1356`) và kết quả trong `document/ovo-model/ovo-complete.md`:

**Kết quả OvO hiện tại nhiều khả năng là nhiễu.** 20 test case cho mean improvement **-0.32%**,
dải -4.94% → +3.27%. Với cohort vài trăm bệnh nhân, SE của AUC ≈ 0.03–0.04 → toàn bộ dải nằm
trong nhiễu. Thêm nữa `torch.manual_seed(42)` nằm trong `__init__` nên mỗi con số chỉ là 1 seed,
không có ước lượng phương sai.

**Ba điểm nghẽn kiến trúc:**

| # | Vấn đề | Vị trí | Hệ quả |
|---|---|---|---|
| 1 | `tanh(risk)` ∈ [-1,1] **và** `Σaᵢ = 1` | `:1096`, `:1113`, `:1126` | logit đầu ra bị chặn cứng trong [-1,1] → OR tối đa ≈ 7.4; tanh bão hòa làm chết gradient |
| 2 | Ràng buộc `Σaᵢ = 1` (L1-normalize) | `:1113`, `:1251` | thêm modality yếu sẽ **hút** trọng số khỏi modality mạnh → giải thích vì sao 3+ modalities Original thắng 6/10 |
| 3 | Gate và expert dùng chung một nguồn thông tin | `:1077` vs `:1086` | gate không biết gì hơn expert, không trả lời được "modality này có đáng tin cho bệnh nhân này không" |

**Nghi ngờ bổ sung (cần kiểm chứng ở Step 1):** `attn_score / l_feature_factor[i]` (`:1209`)
chia logit cho **số lượng feature**. Với radiomics vài trăm–nghìn feature, `score_i - mean_others`
co về ~0 → `sigmoid(≈0) ≈ 0.5` cho mọi modality → **OvO gate có thể đang gần như không hoạt động**.

---

## 1. Nguyên tắc thực thi

### 1.1. Cấu trúc thư mục

Mỗi phương pháp = một folder độc lập dưới `experiments/`. **Không sửa `lung_helpers.py`** —
mọi model mới đều standalone, import `lung_helpers` để dùng lại tiện ích.

```
code/
  experiments/
    common/                      ← harness dùng chung (data loading, evaluation)
      data_setup.py
      evaluate.py
    diagnostics/                 ← Step 1
    A_gated_logodds/             ← Phương pháp A
    B_uncertainty_fusion/        ← Phương pháp B
    results/                     ← bảng kết quả tổng hợp (JSON/Excel)
```

### 1.2. Tiêu chí GIỮ / XÓA

Áp dụng sau khi hoàn tất đánh giá đầy đủ của mỗi phương pháp:

- **GIỮ** nếu: mean ΔAUC ≥ **+0.015** trên ≥ 3/4 benchmark config **VÀ** paired bootstrap
  p < 0.10 trên config chuẩn (`Rad+IHC-G+Gen+PDL1`).
- **XÓA** nếu không đạt: `rm -rf` folder phương pháp đó ngay, chỉ giữ lại một mục ghi kết quả
  âm trong phần *Nhật ký* của file này (để không lặp lại thí nghiệm đã thất bại).

### 1.3. Kỷ luật token

- Mỗi Step là **một lượt làm việc riêng**, không gộp.
- Script ghi kết quả ra file (`results/*.json`, `*.xlsx`), **chỉ in ra một bảng tóm tắt ngắn**.
- Log dài redirect vào scratchpad, không đọc lại trừ khi debug.
- Bước đọc kết quả chỉ đọc file JSON tóm tắt, không đọc log thô.

### 1.4. Benchmark config cố định

Dùng đúng 4 config xuyên suốt để mọi phương pháp so sánh được với nhau:

| Ký hiệu | Modalities | Original AUC (tham chiếu cũ) |
|---|---|---|
| `BM1` | `Rad+IHC-G+Gen+PDL1` (config tốt nhất) | 0.7839 |
| `BM2` | `Rad+IHC-G+Gen+PDL1+Labs` | 0.7879 |
| `BM3` | `Rad+Gen` | 0.7384 |
| `BM4` | `PDL1+Gen` | 0.6931 |

Đánh giá: **5 seeds × 10-fold CV**, báo cáo `mean ± sd`.

---

## 2. Các bước

### Step 0 — Phân tích kiến trúc hiện tại
- [x] Đọc và phân tích `AttentionMatrix`, `AttentionMatrixOvO`, `AttentionMatrixOvO_Softmax`
- [x] Xác định 3 điểm nghẽn kiến trúc + nghi ngờ về `l_feature_factor`
- [x] Xác nhận kết quả OvO cũ nằm trong biên nhiễu
- [x] Viết file plan này

---

### Step 1 — Harness dùng chung
> Mục tiêu: tách phần load dữ liệu (hiện đang lặp trong mọi `compare_*.py`) thành module tái dùng.

- [x] Tạo `experiments/common/data_setup.py`: rút phần `[1/7]`–`[4/7]` của
      `compare_ovo_attention_full.py` thành `load_all()` → `(modality_dict, modality_MASK,
      df_outcomes, rad_filters)`; cache ra pickle trong scratchpad để các step sau không load lại
- [x] Tạo `experiments/common/evaluate.py`:
      - `run_repeated_cv(train_fn, cfg, seeds)` → `dict(mean_auc, sd_auc, per_seed_scores)`
      - `paired_bootstrap(scores_a, scores_b, labels, n=2000)` → `p_value`
      - `save_result(name, dict)` → ghi `results/<name>.json`
- [x] Tạo `experiments/common/benchmarks.py`: định nghĩa 4 config `BM1`–`BM4` ở mục 1.4
- [x] Smoke test: chạy `load_all()`, in shape của mask + phân bố label

**Kết quả:** 247 bệnh nhân, 11 modalities, phân bố label `{1: 185, 0: 62}` (25% responder).
Với `n₁=62, n₂=185`, SE của AUC theo Hanley–McNeil ≈ **0.032** — khớp với ước lượng ban đầu và
xác nhận dải ±5% của bảng OvO cũ nằm gọn trong nhiễu.

---

### Step 2 — Chẩn đoán: OvO gate có thật sự hoạt động không?
> Kiểm chứng nghi ngờ ở mục 0. Nếu đúng thì đây tự nó đã là một phát hiện đáng viết.

- [x] Script `experiments/diagnostics/check_attention_collapse.py` — probe trực tiếp vào
      `clf.dyam` thay vì đọc cột `attn_*` của `summary_df` (với OvO, `attentions` trả về là
      mixing_matrix 3D outer-product, không phải attention weight)
- [x] Đo `std` của attention weight giữa các bệnh nhân, từng modality, cả 2 model
- [x] Đo `std` của `attn_scores_raw` trước/sau khi chia `l_feature_factor`
- [x] Ghi `results/diagnostics.json`
- [x] **Cổng quyết định** — tiêu chí gốc `std(attn_ovo) < 0.02` **KHÔNG** kích hoạt
      (`sd = 0.133`). Tiêu chí này sai; đã thay bằng phép đo đúng (xem dưới)

#### Kết quả — giả thuyết đúng về cơ chế, sai về cách đo

Số feature sau L1 filter: `rad_pc=31, rad_pl=11, rad_ln=26, glcm=150, gen=11, pdl1=1`.

| Đại lượng | Giá trị |
|---|---|
| `sd(sigmoid(sᵢ - mean others))` | **0.034 – 0.070** (rất phẳng, quanh 0.5) |
| `sd(attention weight)` OvO | 0.133 |
| `sd(attention weight)` Original | 0.142 |

Gate **đúng là gần như phẳng** như dự đoán — `sd(sigmoid) ≈ 0.036`. Nhưng `sd(attn)` vẫn lớn,
nên tiêu chí `sd < 0.02` không bắt được. Lý do: biến thiên đó **không đến từ học**, mà đến từ
**mask** — mỗi bệnh nhân thiếu modality khác nhau.

Phép đo đúng là so với baseline "chia đều cho các modality có sẵn" (`aᵢ = maskᵢ / Σmask`):

| Model | Lệch trung bình so với mask-uniform | Lệch lớn nhất |
|---|---|---|
| **OvO** | **0.0121** | 0.0634 |
| Original | 0.0437 | 0.2005 |

**Kết luận: OvO attention về mặt chức năng là trơ.** Nó tái tạo gần như chính xác phép chia đều
cho các modality có sẵn — sai lệch trung bình 0.012 trên thang trọng số [0,1]. Nó không học được
trọng số riêng cho từng bệnh nhân. Original có học, nhưng cũng chỉ lệch 0.044.

Điều này giải thích trọn vẹn bảng kết quả OvO cũ: OvO ≈ trung bình cộng đều các modality có sẵn,
nên AUC của nó không thể khác Original một cách hệ thống. Toàn bộ dải ±5% là nhiễu CV.

**Hệ quả cho thiết kế:** vấn đề không nằm ở *công thức* attention (hợp tác vs cạnh tranh) mà ở
chỗ **tín hiệu vào gate quá yếu** — logit bị chia cho số feature rồi ép qua sigmoid quanh 0.
Đây chính là điểm nghẽn #3, và củng cố hướng của phương pháp A (LayerNorm + bỏ phép chia).

---

### Step 3 — Khóa baseline (thiết lập sàn nhiễu)
> Mọi so sánh về sau đều đối chiếu với bảng này.

- [x] Chạy `train()` (Original) trên `BM1`–`BM4`, 5 seeds × 10-fold
- [x] Chạy `train_ovo()` trên `BM1`–`BM4`, 5 seeds × 10-fold
- [x] Ghi `results/baseline.json` — bảng `mean ± sd` cho 8 ô
- [x] Đối chiếu sd quan sát được với ước lượng lý thuyết
- [x] Kết luận có/không: OvO có khác Original một cách có ý nghĩa thống kê?

#### Bảng baseline (mean ± sd, 5 seeds × 10-fold, paired bootstrap 2000 lần)

| BM | Config | Original | OvO | Δ (OvO−Org) | p |
|---|---|---|---|---|---|
| BM1 | Rad+IHC-G+Gen+PDL1 | 0.7595 ± 0.0221 | 0.7728 ± 0.0197 | +0.0091 | 0.313 |
| BM2 | Rad+IHC-G+Gen+PDL1+Labs | 0.7638 ± 0.0179 | 0.7632 ± 0.0097 | −0.0044 | 0.536 |
| BM3 | Rad+Gen | 0.7095 ± 0.0177 | 0.7132 ± 0.0150 | +0.0004 | 0.980 |
| BM4 | PDL1+Gen | 0.7072 ± 0.0088 | 0.7183 ± 0.0115 | +0.0107 | 0.172 |

**Kết luận 1 — OvO không khác Original.** Cả 4 config đều có |Δ| ≤ 0.011 và p ≥ 0.17.
Không có bằng chứng OvO cải thiện gì. Khớp chính xác với Step 2: một cơ chế tái tạo phép chia
đều theo mask thì không thể khác Original một cách hệ thống.

**Kết luận 2 — các con số cũ lạc quan do may mắn phân hoạch.** So với bảng 1-seed cũ:

| BM | Original cũ → mới | OvO cũ → mới |
|---|---|---|
| BM1 | 0.7839 → **0.7595** (−0.024) | 0.8003 → **0.7728** (−0.028) |
| BM2 | 0.7879 → **0.7638** (−0.024) | 0.7834 → **0.7632** (−0.020) |
| BM3 | 0.7384 → **0.7095** (−0.029) | 0.7548 → **0.7132** (−0.042) |
| BM4 | 0.6931 → **0.7072** (+0.014) | 0.7157 → **0.7183** (+0.003) |

Con số headline **AUC = 0.8003** là sản phẩm của một phân hoạch CV thuận lợi; trung bình trên 5
phân hoạch là **0.7728**. Mọi phát biểu trong paper dựa trên 0.80 cần viết lại.

**Kết luận 3 — sd do phân hoạch (0.009–0.022) đã lớn hơn mọi Δ quan sát được.** Lưu ý sd này
*nhỏ hơn* SE Hanley–McNeil (≈0.032) vì nó chỉ đo phương sai phân hoạch trên một cohort cố định,
không gồm phương sai lấy mẫu cohort. Tức ngay cả nguồn nhiễu nhỏ hơn cũng đã nuốt trọn tín hiệu.

---

### Step 4 — Phương pháp A: Gated Additive Log-Odds
> Sửa cả 3 điểm nghẽn cùng lúc. Thay đổi nhỏ nhất, kỳ vọng lợi ích lớn nhất.

```
rᵢ = Wᵢ · LayerNorm(xᵢ)              # bỏ tanh, bỏ chia n_features
gᵢ = sigmoid(Vᵢ · LayerNorm(xᵢ))     # ∈ [0,1], KHÔNG normalize
total = Σ maskᵢ · gᵢ · rᵢ
```

Gate độc lập → modality yếu tự tắt chứ không cướp trọng số của modality mạnh; bỏ tanh trả lại
dải động. Vẫn giữ tính diễn giải: `gᵢ` vẫn là "độ quan trọng của modality i cho bệnh nhân này",
vẫn vẽ được heatmap như hiện tại.

- [x] **4a.** `experiments/A_gated_logodds/model_gated.py` — đã viết, **đã xóa ở 4e**
- [x] **4b.** Smoke test `BM1` 1 seed: AUC = 0.6897 (dưới ngưỡng 0.70 tự đặt).
      Probe lưới 12 cấu hình → tốt nhất `epochs=125, alpha=0.01, lr=0.001` cho 0.7439,
      vẫn dưới baseline. Dùng cấu hình này cho 4c (lợi thế tuning mà baseline không có).
- [x] **4c.** Đánh giá đầy đủ `BM1`–`BM4` × 5 seeds → `results/method_A.json`
- [x] **4d.** Paired bootstrap vs Original và vs OvO
- [x] **4e.** Áp tiêu chí mục 1.2 → **XÓA** `experiments/A_gated_logodds/`

#### Kết quả cuối (sau khi sửa 2 bug implementation)

| BM | Original | OvO | Gated A | Δ(A−Org) | p |
|---|---|---|---|---|---|
| BM1 | 0.7595 ± 0.0221 | 0.7728 ± 0.0197 | 0.7330 ± 0.0173 | −0.0286 | 0.305 |
| BM2 | 0.7638 ± 0.0179 | 0.7632 ± 0.0097 | 0.7288 ± 0.0260 | −0.0318 | 0.226 |
| BM3 | 0.7095 ± 0.0177 | 0.7132 ± 0.0150 | 0.6504 ± 0.0226 | −0.0575 | 0.070 |
| BM4 | 0.7072 ± 0.0088 | 0.7183 ± 0.0115 | **0.7317 ± 0.0054** | **+0.0205** | 0.397 |

Tiêu chí: Δ ≥ +0.015 trên **1/4** config (cần ≥3/4); p = 0.305 trên BM1 (cần < 0.10) → **XÓA**.

#### Hai bug trong implementation của tôi (đã sửa trước khi chốt)

1. **`get_l2_weight_sum()` phạt cả gain của LayerNorm.** Bản gốc lọc `"weight" in name` là đủ vì
   nó không có LayerNorm; ở A thì gain LayerNorm cũng tên `weight` (khởi tạo = 1, norm ≈ √150
   với glcm), nên L2 kéo gain về 0 và triệt tiêu đúng phép chuẩn hoá vừa thêm.
   Sửa xong AUC gần như không đổi (0.6921 → 0.6897) → **không phải nguyên nhân**.
2. **`nn.LayerNorm` xoá sạch modality 1 feature.** LayerNorm chuẩn hoá *trên chiều feature*,
   nên với n=1 thì `mean` = chính giá trị đó và đầu ra **luôn bằng 0**. Đã xác nhận:
   `LayerNorm(1)([0.5, 1.5, −2.0, 3.0]) → [0, 0, 0, 0]`. `cnl_pdl1_score` đúng loại này, nên
   PD-L1 bị xoá trong BM1/BM2/BM4. Sửa bằng `Identity` cho modality 1 feature.
   Tác động: BM4 **0.6027 → 0.7317**; BM1/BM2 gần như không đổi (−0.002, −0.003).

#### Vì sao A thất bại — và manh mối cho Step 5

Mẫu hình rõ: **A thua ở mọi config có radiomics, thắng ở config không có.**

| Config | Có radiomics? | Δ(A−Org) |
|---|---|---|
| BM3 Rad+Gen | có (3 modality rad) | −0.0575 |
| BM2 | có | −0.0318 |
| BM1 | có | −0.0286 |
| BM4 PDL1+Gen | **không** | **+0.0205** |

Giả thuyết: bỏ `tanh` + bỏ `Σa=1` là gỡ mất **regularization ngầm**. Với n=247 và 62 ca thiểu số,
các ràng buộc đó không phải khuyết điểm mà là thứ giữ mô hình khỏi overfit — và thiệt hại nặng
nhất đúng ở radiomics, nơi nhiều feature tương quan cao. BM4 chỉ có 12 feature tổng cộng nên
không đủ chỗ để overfit, và ở đó A lại tốt nhất.

**Hệ quả cho phương pháp B:** giữ nguyên chuẩn hoá `Σa = 1` và giữ `tanh`, **chỉ** thay cách
tính trọng số bằng precision. Tức sửa điểm nghẽn #3 (tín hiệu gate yếu — đã đo được ở Step 2)
mà KHÔNG đụng #1/#2 (vốn hoá ra là tính năng, không phải lỗi).

> Lưu ý diễn giải: mọi p ≥ 0.22 trừ BM3 (0.070). A **không thua có ý nghĩa thống kê** —
> dữ liệu chỉ đơn giản không phân biệt được. Tiêu chí giữ/xóa đòi hỏi bằng chứng *ủng hộ*
> phương pháp mới, và bằng chứng đó không có.

---

### Step 5 — Phương pháp B: Uncertainty-Weighted Fusion
> Hướng có câu chuyện mạnh nhất cho paper.

```
mỗi head xuất (μᵢ, log σᵢ²)
aᵢ ∝ maskᵢ / σᵢ²                      # precision weighting
total = Σ aᵢ · μᵢ / Σ aᵢ
```

Xử lý missing modality tự nhiên (thiếu = precision 0). Câu chuyện: *"mô hình học được khi nào
nên tin radiomics và khi nào nên tin genomics"* — khác biệt rõ rệt so với OvO.

> **Thiết kế đã điều chỉnh sau thất bại của A:** giữ nguyên `tanh` và `Σaᵢ = 1`, CHỈ thay cách
> tính `aᵢ` bằng precision. Tức sửa đúng điểm nghẽn #3 mà không đụng #1/#2.

- [x] **5a.** `model_uncertainty.py` — đã viết, **đã xóa ở 5e**
- [x] **5b.** Smoke test `BM1`: AUC = 0.7250, nhưng **sửa được đúng vấn đề chẩn đoán ở Step 2** —
      lệch so với mask-uniform = **0.1397** (OvO 0.0121, Original 0.0437), gấp 11× OvO.
      Attention thật sự học theo bệnh nhân. Nhưng AUC lại tệ hơn.
- [x] **5c.** Đánh giá đầy đủ `BM1`–`BM4` × 5 seeds → `results/method_B.json`
- [x] **5d.** Paired bootstrap vs Original và vs OvO
- [x] **5e.** Áp tiêu chí mục 1.2 → **XÓA** `experiments/B_uncertainty_fusion/`

#### Probe `logvar_clamp` — xu hướng đơn điệu, và nó nói lên tất cả

`logvar_clamp` chặn `log σ²`, tức kiểm soát mức cực đoan của trọng số. BM1, 1 seed:

| clamp | 4.0 | 2.0 | 1.0 | 0.5 | 0.25 |
|---|---|---|---|---|---|
| AUC | 0.7250 | 0.7475 | 0.7577 | 0.7719 | 0.8033 |

**Attention càng ít tác dụng thì mô hình càng tốt.** Giới hạn `clamp = 0` là bỏ hẳn attention
(precision hằng số → `aᵢ = maskᵢ/Σmask`), nên nó được đưa vào đánh giá đầy đủ như một biến thể
riêng tên `uniform_avg`.

#### Kết quả cuối (mean ± sd, 5 seeds × 10-fold)

| BM | Original | OvO | unc_B (clamp 0.25) | uniform_avg (KHÔNG attention) | Δ(B−Org) | p |
|---|---|---|---|---|---|---|
| BM1 | 0.7595 ± 0.0221 | 0.7728 ± 0.0197 | **0.7792 ± 0.0255** | 0.7746 ± 0.0186 | +0.0180 | 0.116 |
| BM2 | 0.7638 ± 0.0179 | 0.7632 ± 0.0097 | **0.7718 ± 0.0073** | 0.7665 ± 0.0110 | +0.0071 | 0.517 |
| BM3 | 0.7095 ± 0.0177 | 0.7132 ± 0.0150 | 0.7101 ± 0.0138 | 0.7110 ± 0.0148 | −0.0001 | 0.987 |
| BM4 | 0.7072 ± 0.0088 | 0.7183 ± 0.0115 | **0.7262 ± 0.0110** | 0.7191 ± 0.0084 | +0.0170 | 0.142 |

Tiêu chí: Δ ≥ +0.015 trên **2/4** config (cần ≥3/4); p = 0.116 trên BM1 (cần < 0.10) → **XÓA**.

#### Phát hiện quan trọng nhất: có attention ≈ không có attention

| BM | unc_B (có attention) | uniform_avg (không) | Chênh |
|---|---|---|---|
| BM1 | 0.7792 | 0.7746 | +0.0046 |
| BM2 | 0.7718 | 0.7665 | +0.0053 |
| BM3 | 0.7101 | 0.7110 | −0.0010 |
| BM4 | 0.7262 | 0.7191 | +0.0070 |

Chênh lệch ≤ 0.007 trên cả 4 config — **không phân biệt được**. B sửa được vấn đề trơ (attention
lệch 0.1397 so với chia đều, gấp 11× OvO) nhưng việc đó **không chuyển thành hiệu năng**.

Đây là bằng chứng mạnh hơn Step 2: attention không chỉ trơ trong bản hiện tại, mà kể cả khi làm
cho nó hoạt động thật sự thì cũng **không thu được gì**. Trên cohort này (n=247, 62 ca thiểu số),
phép gộp bằng chứng đơn giản nhất — trung bình `tanh(risk)` trên các modality có sẵn — là đủ.

> Cảnh báo về chính số của tôi: probe 1-seed cho `uniform_avg` = 0.7830 và `clamp 0.25` = 0.8033
> trên BM1; sang 5 seed tụt về 0.7746 và 0.7792. Đúng hiệu ứng chọn seed đã tạo ra 0.8003 trong
> tài liệu cũ. Chỉ tin bảng 5 seed.

**Đã giữ lại:** `experiments/baselines/model_uniform_avg.py` — chính file đã sinh ra các số
`uniform_avg`, đổi mặc định `logvar_clamp = 0`. Đã kiểm chứng tái lập khớp đến 1e-9
(BM1 seed 42: 0.792502).

---

### Step 6 — Baseline trung thực (cho paper)
> Nếu attention không vượt late-fusion LR thì nó chưa xứng đáng tồn tại trong bài.

- [x] LR trên concat toàn bộ modality → `lr_concat` (`experiments/baselines/model_lr.py`)
- [x] Late fusion: trung bình logit z-score của các LR riêng từng modality → `lr_late`
- [x] Ghi `results/baseline_lr.json`
- [x] So sánh: model attention có vượt late-fusion LR một cách có ý nghĩa không?

> Không dùng `train_LR` của `lung_helpers` vì chữ ký khác (nhận 1 DataFrame) và không kiểm soát
> được L1 filter theo fold. Viết mới để dùng **đúng** cấu trúc fold + L1 filter như mô hình neural.

#### Kết quả (mean ± sd, 5 seeds × 10-fold)

| BM | Original | OvO | uniform_avg | lr_concat | lr_late |
|---|---|---|---|---|---|
| BM1 | 0.7595 ± 0.0221 | 0.7728 ± 0.0197 | **0.7746 ± 0.0186** | 0.7276 ± 0.0179 | 0.7615 ± 0.0102 |
| BM2 | 0.7638 ± 0.0179 | 0.7632 ± 0.0097 | **0.7665 ± 0.0110** | 0.7112 ± 0.0195 | 0.7556 ± 0.0054 |
| BM3 | 0.7095 ± 0.0177 | **0.7132 ± 0.0150** | 0.7110 ± 0.0148 | 0.6908 ± 0.0087 | 0.6968 ± 0.0083 |
| BM4 | 0.7072 ± 0.0088 | 0.7183 ± 0.0115 | 0.7191 ± 0.0084 | 0.7472 ± 0.0110 | **0.7515 ± 0.0084** |

#### `lr_late` vs `uniform_avg` — cô lập đóng góp của head neural

Hai mô hình này gộp bằng chứng **giống hệt nhau** (trung bình trên modality có sẵn), chỉ khác
head: neural `tanh(W·x)` vs hồi quy logistic. Nên hiệu số cô lập đúng đóng góp của head neural.

| BM | Δ(lr_late − uniform_avg) | 95% CI | p |
|---|---|---|---|
| BM1 | −0.0167 | [−0.0584, +0.0217] | 0.398 |
| BM2 | −0.0121 | [−0.0520, +0.0274] | 0.550 |
| BM3 | −0.0234 | [−0.0694, +0.0223] | 0.314 |
| BM4 | +0.0266 | [−0.0022, +0.0590] | 0.074 |

**Không config nào có ý nghĩa thống kê.** Head neural nhỉnh hơn LR ~0.012–0.023 trên các config
có radiomics, thua 0.027 trên config không có. Tất cả nằm trong nhiễu.

#### `lr_late` vs `Original` — kết quả có ý nghĩa duy nhất, và nó nghiêng về LR

| BM | Δ(lr_late − original) | 95% CI | p |
|---|---|---|---|
| BM1 | −0.0070 | [−0.0581, +0.0431] | 0.755 |
| BM2 | −0.0135 | [−0.0587, +0.0298] | 0.542 |
| BM3 | −0.0238 | [−0.0728, +0.0246] | 0.334 |
| BM4 | **+0.0393** | [+0.0013, +0.0809] | **0.045** |

Trên BM4 (`PDL1+Gen`, 12 feature), **hồi quy logistic thường vượt mô hình attention có ý nghĩa
thống kê**. Đây là kết quả p < 0.05 duy nhất trong toàn bộ dự án — và nó chống lại kiến trúc
hiện tại, không ủng hộ.

`lr_concat` thua rõ trên mọi config có radiomics (−0.03 đến −0.05): ghép hàng trăm feature
tương quan vào một LR duy nhất là cách gộp tệ. Nhưng nó lại thắng Original trên BM4 (+0.040).

**Kết luận Step 6:** mô hình neural đa modality **không vượt được** late-fusion LR trên cohort
này. Chênh lệch ≤ 0.027, không config nào đạt p < 0.05 theo hướng ủng hộ neural.

---

### Step 7 — Ablation của phương pháp thắng cuộc
> Chỉ chạy nếu Step 4 hoặc Step 5 có phương pháp sống sót. Đây là phần đóng góp khoa học:
> chỉ ra **chính xác** yếu tố nào tạo ra cải thiện.

> **ĐỔI HƯỚNG:** cả A và B đều bị loại nên không còn "phương pháp thắng cuộc" để ablation.
> Câu hỏi giá trị hơn: trong chính kiến trúc đang dùng, thành phần nào đóng góp? Dùng các cờ
> có sẵn (`attention_gate_enabled`, `cross_modality_enabled`) nên không cần code model mới.

- [x] Ablation 1: attention học được vs chia đều (`original` vs `uniform_avg`)
- [x] Ablation 2: bỏ chuẩn hoá (`attention_gate_enabled=False` → tổng thay vì trung bình)
- [x] Ablation 3: `cross_modality_enabled=True` (N² lớp attention thay vì N)
- [x] Ghi `results/ablation.json` + bảng
- [x] Kết luận: yếu tố nào thực sự quan trọng?

#### Bảng ablation (mean ± sd, 5 seeds × 10-fold)

| BM | original (gate học) | gate_off (TỔNG) | uniform_avg (TRUNG BÌNH) | cross_modality |
|---|---|---|---|---|
| BM1 | 0.7595 ± 0.0221 | 0.7339 ± 0.0267 | **0.7746 ± 0.0186** | 0.7677 ± 0.0219 |
| BM2 | 0.7638 ± 0.0179 | 0.7253 ± 0.0164 | 0.7665 ± 0.0110 | **0.7683 ± 0.0165** |
| BM3 | 0.7095 ± 0.0177 | 0.6880 ± 0.0122 | **0.7110 ± 0.0148** | 0.7101 ± 0.0164 |
| BM4 | 0.7072 ± 0.0088 | **0.7305 ± 0.0066** | 0.7191 ± 0.0084 | 0.7107 ± 0.0071 |

#### Kết luận 1 — attention học được KHÔNG đóng góp gì (null result chặt)

| BM | Δ(uniform_avg − original) | 95% CI | p |
|---|---|---|---|
| BM1 | +0.0098 | [−0.0117, +0.0322] | 0.364 |
| BM2 | −0.0014 | [−0.0186, +0.0154] | 0.907 |
| BM3 | −0.0004 | [−0.0126, +0.0114] | 0.959 |
| BM4 | +0.0127 | [−0.0048, +0.0301] | 0.161 |

Điểm mấu chốt: **CI ở đây rất hẹp** (±0.012 đến ±0.032), khác hẳn các so sánh trước vốn có CI
±0.08. Nên đây không phải "thiếu lực thống kê nên không kết luận được" mà là **loại trừ được
mọi hiệu ứng lớn hơn ~±0.02**. Thay toàn bộ attention học được bằng hằng số `1/Σmask` thì
**không có gì thay đổi**.

#### Kết luận 2 — giá trị của gate nằm ở CHUẨN HOÁ, không ở trọng số học được

| BM | Δ(gate_off − original) | 95% CI | p |
|---|---|---|---|
| BM1 | −0.0258 | [−0.0618, +0.0100] | 0.161 |
| BM2 | −0.0340 | [−0.0676, +0.0002] | 0.053 |
| BM3 | −0.0202 | [−0.0574, +0.0175] | 0.279 |
| BM4 | **+0.0230** | [+0.0018, +0.0471] | **0.035** |

Tắt gate = `Σ tanh(rᵢ)` (TỔNG, không chia). Trên config nhiều modality thì tệ đi rõ; trên BM4
(2 modality) lại tốt lên có ý nghĩa. Ghép với Kết luận 1:

> **Toàn bộ giá trị của cơ chế attention đến từ việc nó chia cho số modality có sẵn, chứ không
> phải từ trọng số nó học được.** Thay trọng số học bằng hằng số → không đổi. Bỏ phép chia →
> hỏng. Đây là lời giải thích cơ học cho mọi kết quả từ Step 2 tới giờ.

Ý nghĩa lâm sàng: bệnh nhân thiếu modality không được cộng dồn ít điểm hơn một cách giả tạo.
Đó là một phép hiệu chỉnh số học, không cần mạng nơ-ron để học.

#### Kết luận 3 — `cross_modality_enabled` hoàn toàn trơ, mà tốn 6× tính toán

| BM | Δ(cross − original) | p |
|---|---|---|
| BM1 | +0.0034 | 0.678 |
| BM2 | +0.0031 | 0.685 |
| BM3 | +0.0003 | 0.948 |
| BM4 | +0.0024 | 0.528 |

Tất cả ≤ +0.0034, p ≥ 0.53. Nhưng nó dùng **N² lớp linear thay vì N** (6 modality → 36 thay vì
6), là lý do lần chạy này lâu gấp nhiều lần. **Khuyến nghị: luôn để `False`.**

---

### Step 8 — Tổng hợp
- [x] Bảng kết quả cuối: Original | OvO | A | B | LR-concat | LR-late, trên `BM1`–`BM4` → `ket-qua.md`
- [x] Hình: forest plot ΔAUC với CI → `forest-delta-auc.svg` (24 so sánh; chỉ 2 loại trừ 0, cả hai nghiêng về LR)
- [x] Xóa toàn bộ folder phương pháp không đạt tiêu chí (`A_gated_logodds/`, `B_uncertainty_fusion/` — đã xác nhận không còn)
- [x] Viết `document/2026-07-20_attention-redesign/ket-qua.md`
- [x] Cập nhật Nhật ký bên dưới

---

## 3. Nhật ký kết quả

> Ghi lại cả kết quả âm — để không lặp lại thí nghiệm đã thất bại.

| Ngày | Step | Kết quả | Quyết định |
|---|---|---|---|
| 2026-07-21 | 8 | **Tổng hợp xong.** Không phương pháp nào đạt tiêu chí GIỮ. Bảng cuối 6 phương pháp × 4 BM + forest plot ΔAUC (24 so sánh, chỉ 2 loại trừ 0, cả hai nghiêng về LR ở BM4). Ba kết luận: (1) OvO ≈ Original, 0.80 cũ là ảo giác seed; (2) giá trị attention nằm ở chuẩn hoá `1/Σmask`, không ở trọng số học (CI hẹp ±0.02); (3) làm attention hoạt động thật cũng không cải thiện, neural không vượt LR. Viết `ket-qua.md` + `forest-delta-auc.svg` | **HOÀN TẤT PLAN.** A/B đã xóa, giữ `model_uniform_avg.py`. Dừng loop |
| 2026-07-20 | 0 | Xác định 3 điểm nghẽn kiến trúc; kết quả OvO cũ nằm trong biên nhiễu | Tiến hành plan |
| 2026-07-20 | 1 | Harness chạy được. n=247, label `{1:185, 0:62}`, 11 modalities. SE(AUC) ≈ 0.032 | Xong, sang Step 2 |
| 2026-07-20 | 2 | **OvO attention trơ về mặt chức năng**: lệch trung bình chỉ 0.012 so với `maskᵢ/Σmask`. `sd(sigmoid) ≈ 0.036`. Original lệch 0.044 | Xong. Tiêu chí `sd<0.02` ban đầu sai → đã thay bằng so sánh với mask-uniform |
| 2026-07-20 | 7 | **Giá trị của attention nằm ở phép CHUẨN HOÁ, không ở trọng số học được.** `uniform_avg` vs `original`: \|Δ\| ≤ 0.013, p ≥ 0.16, **CI hẹp ±0.02** → null result chặt. Bỏ chuẩn hoá (`gate_off`) thì hỏng (−0.02 đến −0.034). `cross_modality` trơ hoàn toàn (≤+0.0034, p≥0.53) mà tốn N² lớp | Xong, sang Step 8 |
| 2026-07-20 | 6 | **Neural KHÔNG vượt được LR.** `lr_late` vs `uniform_avg`: Δ ≤ 0.027, p ≥ 0.074 trên cả 4 config. Trên BM4, `lr_late` **vượt Original có ý nghĩa** (+0.0393, p=0.045) — kết quả p<0.05 duy nhất của dự án, và nó chống lại kiến trúc hiện tại | Xong, sang Step 7 |
| 2026-07-20 | 5 | **Phương pháp B (precision fusion) THẤT BẠI** — 2/4 config, p=0.116. Nhưng B *sửa được* vấn đề trơ (lệch 0.1397 vs OvO 0.0121) mà **không cải thiện AUC**. `uniform_avg` (không attention) ngang B trong phạm vi ≤0.007 trên cả 4 config. Đã xóa `experiments/B_uncertainty_fusion/` | Giữ `experiments/baselines/model_uniform_avg.py` (đã kiểm chứng tái lập). Kết luận: attention không mang lại gì trên cohort này |
| 2026-07-20 | 4 | **Phương pháp A (gated log-odds) THẤT BẠI** — Δ ≥ +0.015 chỉ 1/4 config, p=0.305 trên BM1. Thắng duy nhất ở BM4 (+0.0205, config không có radiomics). Đã xóa `experiments/A_gated_logodds/` | Kết quả số giữ trong `results/method_A.json` + `results/scores/gated_A__*.csv`. Bài học: `tanh` + `Σa=1` là regularization ngầm, đừng gỡ |
| 2026-07-20 | 3 | Phát hiện 3 bug (`seed` inert, thiếu `.to(device)` trong OvO predict_proba, `summary_df` OvO dtype object). Sau khi sửa: **OvO không khác Original trên cả 4 config**, \|Δ\| ≤ 0.011, p ≥ 0.17. Con số cũ AUC 0.8003 → thực tế 0.7728 ± 0.0197 | Xong. Baseline đã khóa, sang Step 4 |

---

## 4. Ghi chú kỹ thuật

- ~~`train()` và `train_ovo()` đã nhận tham số `seed` → chạy multi-seed không cần sửa gì.~~
  **SAI — xem bug #1 dưới đây.**

### Bug phát hiện khi chạy Step 3

**#1 — Tham số `seed` của `train()`/`train_ovo()` là VÔ HIỆU.** Cả hai gọi `set_global_seed(seed)`
nhưng ngay sau đó dùng `KFold(n_splits=folds, random_state=0, shuffle=True)` — hardcode `0`
(`lung_helpers.py:2513`, `:2658`). Thêm nữa `AttentionMatrix.__init__` hardcode
`torch.manual_seed(42)` (`:979`), ghi đè seed toàn cục. Hệ quả: **mọi seed cho ra kết quả giống
hệt nhau** — đã kiểm chứng, 5 seed trên BM1 đều cho AUC = 0.7976.

*Cách xử lý:* không sửa `train()` (đổi `random_state` sẽ làm lệch mọi kết quả cũ). Thay vào đó
`run_repeated_cv` hoán vị thứ tự hàng của `outcomes` theo seed — `train()` gọi
`kf.split(outcomes.index)` rồi lấy `outcomes.index[train]`, còn dữ liệu truy cập bằng `.loc`
nên vẫn khớp đúng bệnh nhân. Đã kiểm chứng: BM4 Original cho 0.6917 / 0.7129 / 0.7087.
*Giới hạn:* khởi tạo model vẫn cố định ở 42, nên `sd` đo được là phương sai do **phân hoạch CV**
(thành phần trội), không gồm phương sai do khởi tạo.

**#2 — `MultiModalDynamicModelOvO.predict_proba` thiếu `.to(self.device)` trên mask**
(`lung_helpers.py:1713`). Bản Original (`:1509`) có. Trên máy có CUDA, `train_ovo` crash ngay:
`Expected all tensors to be on the same device, but found cuda:0 and cpu`. **Đã sửa.**
Suy ra mọi kết quả OvO trước đây đều chạy trên CPU — không ảnh hưởng tính đúng đắn của chúng.

**#3 — `summary_df` của OvO bị ép dtype `object`.** `train_ovo` lưu `attentions[idx]` vốn là
mixing_matrix 2D vào các cột `attn_*`, làm toàn bộ DataFrame thành `object` →
`roc_auc_score` báo `unknown format`. Đã xử lý bằng `.astype(float)` trong harness.
Hệ quả rộng hơn: **các cột `attn_*` trong `summary_df` của OvO không dùng được để diễn giải** —
chúng là hàng của ma trận outer-product, không phải attention weight. Đây là lý do Step 2 phải
probe thẳng vào `clf.dyam`.
- `train_gmu()` (`:2785`) **chưa** có tham số `seed` — nếu cần đưa GMU vào so sánh thì phải bổ sung.
- `auc_roc_ci()` (`:290`) dùng DeLong, nhưng DeLong CI trên score gộp từ CV là **không** đúng
  về mặt lý thuyết (các fold không độc lập) → dùng paired bootstrap trên seed thay thế.
- Model params chuẩn (từ `compare_ovo_attention_full.py:141`):
  `{'epochs': 125, 'lr': 0.01, 'alpha': 0.001, 'beta': 0.0, 'cross_modality_enabled': False}`
- Modality NLP luôn cần `no_scale=[idx]` — embeddings đã cosine-normalized, RobustScaler sẽ phá hỏng.
- **Lỗi phát hiện trong `train_ovo_softmax` (`lung_helpers.py:2759`):** unpack
  `risks, attentions, shares, _ = clf.get_summary_scores(...)` trong khi hàm trả về
  `(output, risk_scores, mixing_matrix, attention_share)` → lệch một vị trí, `risks` thực chất
  nhận `output`. `train_ovo` (`:2693`) và `train` (`:2619`) unpack đúng. Không ảnh hưởng plan này
  (không dùng biến thể softmax) nhưng mọi kết quả cũ từ `train_ovo_softmax` đều đáng ngờ.
- Chạy script phải đặt `PYTHONIOENCODING=utf-8`, nếu không stdout tiếng Việt lỗi cp1252 khi redirect.
- Cache dữ liệu pickle dạng dict, không pickle dataclass (class bị gắn vào `__main__`).
