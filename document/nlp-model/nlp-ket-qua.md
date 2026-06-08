# Kết quả Đánh giá NLP Clinical Embedding

> KFold 10-fold, Discovery cohort n=247, `all-MiniLM-L6-v2`

---

## 1. Kết luận nhanh

| Câu hỏi | Kết quả |
|---------|---------|
| NLP có AUC cao hơn Labs/NoClin không? | **Có** (+0.017–0.030), nhưng CI chồng nhau |
| Có ý nghĩa thống kê theo DeLong CI? | **Không** — tất cả CI đều chồng nhau |
| KM log-rank có ý nghĩa không? | **Có** — tất cả p < 0.005 ✓ |
| Model y tế (BioClinBERT) có tốt hơn MiniLM? | **Không** — kém hơn 0.017–0.018 AUC |
| NLP đứng một mình có tốt không? | **Không** — NLP-only 0.539 < Labs-only 0.594 |

> **Kết luận:** Sau 7 biến thể thử nghiệm, NLP embedding cho xu hướng cải thiện AUC nhất quán nhưng **không đạt ý nghĩa thống kê** do n=247 quá nhỏ. Không khuyến nghị dùng trong mô hình chính. Trình bày như **negative result** trong supplementary.

---

## 2. Ký hiệu modality

| Ký hiệu | Dữ liệu thực tế |
|---------|-----------------|
| **Rad** | Radiomics CT (PC + PL + LN lesions) |
| **IHC-A** | Pathology: PD-L1 IHC intensity |
| **IHC-G** | Pathology: GLCM texture từ IHC |
| **Gen** | Genomics (EGFR, KRAS, STK11, TMB, ...) |
| **PDL1** | PD-L1 TPS score |
| **Labs** | 13 features số lâm sàng |
| **NLP** | 384-dim sentence embedding (MiniLM-L6-v2) |
| **NLP-PCA16** | NLP → PCA → 16-dim (~80% variance) |

---

## 3. AUC — 7 biến thể

*(KFold 10-fold, n=247, Discovery cohort)*

### IHC-A (Pathology: PDL1 IHC)

| Model | AUC | 95% CI (DeLong) | ΔAUC vs No Clinical |
|-------|-----|-----------------|---------------------|
| DyAM Rad+IHC-A+Gen+PDL1 (No Clin) | 0.764 | [0.695–0.833] | *(ref)* |
| + Labs | 0.768 | [0.700–0.837] | +0.004 |
| + NLP raw (384d) | 0.781 | [0.717–0.845] | +0.017 |
| **+ NLP-PCA16 (16d)** | **0.784** | **[0.719–0.848]** | **+0.020** |
| + Combined (Labs+NLP) | 0.767 | [0.700–0.835] | +0.003 |
| + Labs+NLP-PCA16 (x2) | 0.753 | [0.682–0.823] | −0.011 |
| + BioClinBERT-PCA16 | 0.767 | [0.701–0.833] | +0.003 |

### IHC-G (Pathology: GLCM texture)

| Model | AUC | 95% CI (DeLong) | ΔAUC vs No Clinical |
|-------|-----|-----------------|---------------------|
| DyAM Rad+IHC-G+Gen+PDL1 (No Clin) | 0.784 | [0.717–0.850] | *(ref)* |
| + Labs | 0.788 | [0.723–0.853] | +0.004 |
| **+ NLP raw (384d)** | **0.813** | **[0.753–0.874]** | **+0.030** |
| + NLP-PCA16 (16d) | 0.812 | [0.751–0.873] | +0.028 |

---

## 4. Kiểm định thống kê

### 4.1. DeLong 95% CI — Chuẩn bài báo gốc

**Nền tảng toán học:** AUC ≡ P(score_pos > score_neg). DeLong tính phương sai của AUC qua structural components:

```
Với mỗi bệnh nhân dương i:  V10_i = mean ψ(score_i, score_j) trên tất cả j âm
Với mỗi bệnh nhân âm j:    V01_j = mean ψ(score_i, score_j) trên tất cả i dương
  ψ(x, y) = 1 nếu x > y, 0.5 nếu x = y, 0 nếu x < y

Var(AUC) = [Σ(V10_i − AUC)² / (n₊−1)] / n₋
         + [Σ(V01_j − AUC)² / (n₋−1)] / n₊

CI 95% = [AUC − 1.96√Var(AUC),  AUC + 1.96√Var(AUC)]
```

**Tiêu chí:** Hai model có ý nghĩa thống kê khi **CI không chồng nhau**.

**Kết quả:**
```
IHC-G — No Clinical:  [0.717 ════════════ 0.850]
IHC-G — + NLP raw:         [0.753 ════════════ 0.874]
                                  ↑ vẫn chồng tại 0.753–0.850
→ Không ý nghĩa thống kê dù ΔAUC = +0.030
```

Tất cả CI chồng nhau đáng kể → **không có biến thể nào đạt ý nghĩa theo DeLong CI.**

### 4.2. KM Log-rank — Phân tầng sống còn

**Công thức log-rank:**
```
Tại mỗi thời điểm t có sự kiện:
  E_1t = (n_1t / (n_1t + n_2t)) × d_t     (số sự kiện kỳ vọng)
  U = Σ_t (O_1t − E_1t)
  χ² = U² / V  ~  χ²(df=1)
```

Tiêu chí oncology: χ² > 7.88 → p < 0.005

| Model | χ² stat | p-value |
|-------|---------|---------|
| No Clinical (IHC-A) | 28.17 | < 0.005 |
| NLP-PCA16 (IHC-A) | 23.74 | < 0.005 |
| No Clinical (IHC-G) | 20.07 | < 0.005 |
| **NLP-PCA16 (IHC-G)** | **28.92** | **< 0.005** ★ cao nhất |

Tất cả model đạt **p < 0.005**. NLP cải thiện phân tầng với IHC-G (20.07 → 28.92).

**Khác với DeLong CI:** Log-rank đánh giá phân tầng sống còn, không so sánh 2 model. Vì vậy tất cả model đều có thể đạt p < 0.005 dù CI DeLong chồng nhau.

### 4.3. Bootstrap p-value (bổ sung)

*(5000 lần bootstrap, so với Labs làm reference)*

| Biến thể | IHC | ΔAUC | p-value | Bootstrap CI |
|----------|-----|------|---------|--------------|
| MiniLM raw | IHC-G | +0.026 | 0.495 | [+0.001, +0.051] |
| MiniLM-PCA16 | IHC-G | +0.024 | 0.508 | [0.000, +0.049] |
| BioClinBERT-PCA16 | IHC-A | −0.001 | 0.927 | [−0.031, +0.029] |

Tất cả p >> 0.05.

---

## 5. Tại sao NLP không vượt ngưỡng ý nghĩa?

### Nguyên nhân chính: n/d ratio quá nhỏ

| Modality | Dim (d) | n/d ratio | Đánh giá |
|----------|---------|-----------|----------|
| Labs | 13 | 247/13 ≈ **19.0** | Tốt nhất |
| NLP-PCA16 | 16 | 247/16 ≈ 15.4 | Chấp nhận được |
| NLP raw | 384 | 247/384 ≈ **0.64** | Nguy hiểm |
| BioClinBERT raw | 768 | 247/768 ≈ **0.32** | Tệ nhất |

**Tại sao CI rộng với n=247:**
```
n = 247 → CI ≈ ±0.07   → chênh lệch 0.02 bị "chìm" trong nhiễu
n = 2000 → CI ≈ ±0.02  → chênh lệch 0.02 sẽ có ý nghĩa
```

Chỉ 3–4 bệnh nhân phân loại khác đi là AUC thay đổi ±0.02 — lớn hơn tín hiệu NLP thực sự.

### Nguyên nhân thứ cấp — BioClinBERT kém hơn MiniLM

- BioClinBERT được train cho masked language modeling, không phải sentence similarity
- Mean pooling hidden states là heuristic — không phải cách model được thiết kế
- 768d → 16d (PCA) = 92.8% variance collapse → embedding space không phân biệt tốt

### Nguyên nhân thứ cấp — Combined (Labs+NLP) không giúp ích

8 modalities với n=247 → attention không học ổn định giữa các folds → variance cao → AUC giảm.

---

## 6. AUC 0.788 trong bức tranh tổng thể

| Bước | Model | AUC |
|------|-------|-----|
| Baseline thấp nhất | LR Clinical (13 features) | 0.570 |
| Đơn modality | LR Rad-LN | 0.681 |
| Đơn modality | LR PDL1-TPS / LR Gen | 0.729 |
| **DyAM đa modality** | **DyAM Rad+IHC-G+Gen+PDL1** | **0.784** |
| **DyAM + Labs** | **DyAM Rad+IHC-G+Gen+PDL1+Labs** | **0.788** |
| DyAM + NLP | DyAM Rad+IHC-G+Gen+PDL1+NLP raw | 0.813 |
| Trần lý thuyết | Bayes Error Rate (NSCLC ICI) | ~0.85–0.88 |

**AUC 0.788 ở chuẩn oncology:** > 0.70 = tốt; > 0.80 = rất tốt → 0.788 nằm ở ngưỡng "tốt–rất tốt". Còn cách trần lý thuyết ~0.08 nhưng khoảng này **không phân biệt được** với CI ±0.07.

**Kết luận:** Với n=247 và 4 modality hiện có, AUC ~0.79–0.81 là gần trần thực tế. Thay vì chase AUC, hướng có giá trị hơn là đổi bài toán sang **survival prediction** (C-index).

---

## 7. Khuyến nghị

**Cho luận văn:**
- Dùng **No Clinical / Labs** làm baseline — đã đủ mạnh
- Trình bày NLP như **negative result** trong supplementary
- Diễn đạt: *"NLP embedding cho xu hướng cải thiện nhất quán (ΔAUC=+0.02–0.03) nhưng không đạt ý nghĩa thống kê theo DeLong CI do hạn chế cohort size (n=247). Tất cả model đạt log-rank p<0.005 trong phân tầng sống còn."*

**Nếu muốn NLP hoạt động:**
- Tăng n > 1000 bệnh nhân
- Fine-tune sentence encoder trực tiếp trên NSCLC/ICI data
- Extract structured features từ NLP (tên mutation, tiền sử điều trị) thay vì dense embeddings

---

*Tổng hợp từ: `nlp-evaluation.md`, `auc-tong-hop-nlp.md`*
