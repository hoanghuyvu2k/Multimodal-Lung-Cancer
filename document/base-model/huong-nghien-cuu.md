# Hướng Nghiên Cứu và Phát triển

> Tổng hợp các hướng cải tiến kiến trúc (từ bài báo Nature) và các hướng nghiên cứu tiếp theo cho luận văn.

---

## Phần 1 — Cải tiến Kiến trúc (từ Nghiên cứu Mới nhất)

*Cảm hứng từ: "Multimodal deep learning for cancer prognosis prediction with clinical information prompts integration" — Nature*

### Hướng 1A: "Ngôn ngữ hóa" dữ liệu Lâm sàng

**Vấn đề:** Clinical hiện dùng `RobustScaler` trên 13 con số thô (tuổi 65 → "-0.12") — mất ngữ cảnh y khoa. Clinical bị lấn át bởi Radiomics/Genomics chiều cao hơn.

**Giải pháp:** Language Prompting:
1. Chuyển 13 chỉ số thành câu văn tiếng Anh: *"Patient is 65-year-old, ECOG 1, albumin 4.2..."*
2. Encode qua sentence transformer (MiniLM, BioBERT) → vector 384 chiều
3. Tích hợp vào pipeline như một modality độc lập với `no_scale`

→ **Đã thực hiện:** Xem `nlp-model/` cho kết quả chi tiết.

### Hướng 1B: Feature-Level Cross-Attention

**Vấn đề:** `AttentionMatrix` hoạt động ở mức modality-level (tạo ma trận N×N giữa các modality). Chưa có tương tác ở mức feature-level.

**Giải pháp:** Bidirectional Cross-Attention cấp độ đặc trưng — cho phép features của Clinical kết nối trực tiếp với features của Radiomics:
- *"Bệnh nhân hút thuốc nặng"* → tự động soi kỹ vào texture tổn thương trên CT
- Học được logic liên quan giữa bệnh án và hình ảnh

### Hướng 1C: Cosine Similarity Loss

**Vấn đề:** Deep learning có xu hướng "lười" — Gen và Clinical cùng focus vào cùng 1 điểm rõ trên khối u → dư thừa thông tin.

**Giải pháp:** Thêm Cosine Similarity Loss: phạt nếu attention matrix của Gen và Clinical quá giống nhau. Buộc model tìm thông tin bổ sung (complementary) từ các modality.

### Hướng 1D: Foundation Models (Scale-up)

**Vấn đề:** Features hiện tại là tĩnh (extracted) — dễ overfitting với n nhỏ.

**Giải pháp khi có điều kiện:**
- **Genomics:** `scFoundation` — encode chuỗi gen thô thành embeddings
- **Radiomics:** `UNI`, `RetCCL` — encode toàn bộ ảnh CT thay vì features thủ công
- Kế thừa tinh hoa từ hàng triệu ảnh thay vì tự train vài nghìn ảnh

---

## Phần 2 — Hướng Nghiên Cứu Tiếp Theo (Tổng quan)

| Hướng | Độ ưu tiên | Khả thi ngay? | Tác động kỳ vọng |
|-------|-----------|--------------|-----------------|
| 1. Survival Prediction (Cox loss) | ★★★ Cao nhất | Có — data đã có | C-index mới, câu hỏi lâm sàng mạnh hơn |
| 2. OvO Attention — phân tích sâu | ★★ Trung bình | Có — code đã có | Hiểu rõ khi nào OvO tốt hơn |
| 3. External Validation | ★★ Trung bình | Có — cohort có sẵn | Độ tin cậy tổng quát hóa |
| 4. NLP Embedding (n lớn) | ★ Thấp | Không — cần n > 1000 | Xác nhận trend đã quan sát |
| 5. Interpretability / Biomarker Discovery | ★★ Trung bình | Có | Ý nghĩa lâm sàng của attention |

### Timeline khuyến nghị cho luận văn (1 người, ~2 tháng)

```
Tuần 1:     External Validation (nhanh, chắc chắn có kết quả)
Tuần 2–3:   OvO phân tích sâu (đã có code, cần phân tích thêm)
Tuần 4–9:   Survival Prediction (đóng góp chính — câu hỏi mới)
Tuần 10–11: Interpretability / attention cluster analysis
Tuần 12:    Tổng hợp, viết kết luận
```

### Cấu trúc luận văn đề xuất

```
Chương 1: Background — NSCLC, ICI, Multimodal learning, Survival analysis
Chương 2: DyAM gốc — Cooperative attention, kết quả binary (AUC=0.788)
Chương 3: OvO Attention — Competitive vs Cooperative, khi nào OvO tốt hơn
Chương 4: Survival Prediction — Cox loss, C-index (đóng góp chính)
Chương 5: External Validation + Interpretability
Chương 6: NLP (supplementary/negative result) + Kết luận
```

---

## Phần 3 — Survival Prediction (Chi tiết)

### 3.1. Vấn đề của bài toán Binary hiện tại

```
Bệnh nhân A: tiến triển sau 1 tháng  → label = 1
Bệnh nhân B: tiến triển sau 12 tháng → label = 1
→ Model coi hai người như nhau — mất thông tin thời gian
```

**Câu hỏi bác sĩ thực tế:** *"Bệnh nhân sẽ benefit BAO LÂU?"* — binary classification không trả lời được.

### 3.2. Bài toán mới: Survival Prediction

```
Input:  Multimodal data (Rad + IHC + Gen + PDL1) — giống hệt hiện tại
Output: Risk score liên tục → dự đoán PFS
Loss:   Cox Partial Likelihood Loss  (thay BCEWithLogitsLoss)
Metric: C-index (thay AUC-ROC)
```

### 3.3. Dữ liệu đã có sẵn

| Cột | Mô tả | Thống kê |
|-----|-------|---------|
| `pfs` | Progression-Free Survival (tháng) | n=366, median=2.65, mean=6.67, max=59.7 |
| `pfs_censor` | 1=tiến triển xảy ra, 0=censored | 308 events (84%) / 366 tổng |

**Không cần thu thập thêm dữ liệu** — `df_clinical['pfs']` đã có trong notebook (dùng ở cell 53, 60).

### 3.4. C-index — Metric của Survival

```
C-index = P(risk_A > risk_B | PFS_A < PFS_B)
```

| Giá trị | Ý nghĩa |
|---------|---------|
| 0.5 | Random |
| 0.6–0.7 | Trung bình |
| > 0.7 | Tốt trong oncology |

C-index xử lý **censoring** đúng hơn AUC. Với dataset này (16% censored), C-index là metric phù hợp hơn.

### 3.5. Cox Partial Likelihood Loss

```python
def cox_partial_loss(risk_scores, durations, events):
    order      = torch.argsort(durations, descending=True)
    risk       = risk_scores[order]
    event      = events[order]
    log_cumsum = torch.logcumsumexp(risk, dim=0)
    return -torch.mean((risk - log_cumsum) * event)
```

**Trực giác:** Với mỗi bệnh nhân bị tiến triển tại thời điểm t, phạt nếu model không xếp bệnh nhân đó có risk cao hơn tất cả bệnh nhân còn sống tại t.

### 3.6. Thay đổi cần thực hiện

Thêm vào `lung_helpers.py` (không sửa `train()` và `train_ovo()` gốc):

```python
# 1. Loss function
def cox_partial_loss(risk_scores, durations, events): ...

# 2. C-index metric
def concordance_index(durations, risk_scores, events):
    from lifelines.utils import concordance_index as ci
    return ci(durations, -risk_scores, events)

# 3. Hàm train mới — độc lập
def train_survival(modality_list_in, modality_mask, df_survival,
                   l1_dfs_filter, model_params, folds=10): ...
```

### 3.7. Thiết kế thử nghiệm

| So sánh | Câu hỏi |
|---------|---------|
| DyAM-Binary vs DyAM-Survival | Cox loss có tận dụng thông tin thời gian tốt hơn? |
| Original attention vs OvO với Cox loss | Cơ chế nào phù hợp cho survival? |
| Rad-only → Rad+Gen → Rad+IHC+Gen+PDL1 | Thêm modality cải thiện C-index như thế nào? |

### 3.8. So sánh ý nghĩa lâm sàng

| Câu hỏi | Binary DyAM | Survival DyAM |
|---------|------------|---------------|
| Bệnh nhân có đáp ứng không? | ✓ | ✓ |
| Bệnh nhân sẽ benefit bao lâu? | ✗ | **✓** |
| Phân nhóm risk liên tục? | Chỉ nhị phân | **✓** |

### 3.9. Tài liệu tham khảo

- **DeepSurv** (Katzman et al. 2018): Neural network với Cox loss — [arxiv.org/abs/1606.00931](https://arxiv.org/abs/1606.00931)
- **lifelines**: `lifelines.utils.concordance_index`

---

## Phần 4 — External Validation

### Cohort có sẵn trong notebook

```python
# Trong Figures-Finalized.ipynb:
df_cohort_valid = df_cohort[df_cohort['cohort'] == 'validation']
```

| Cohort | n | Mục đích |
|--------|---|---------|
| Discovery | 247 | Train + KFold CV (đã dùng) |
| Validation | ~120 | External validation — chưa khai thác |

**Việc cần làm:**
1. Chạy model tốt nhất (DyAM Rad+IHC-G+Gen+PDL1) trên validation cohort
2. So sánh AUC discovery vs validation — kiểm tra overfitting
3. KM log-rank trên validation cohort
4. Nếu AUC validation << discovery: điều tra nguyên nhân (population shift, data quality)

---

## Phần 5 — Interpretability / Biomarker Discovery

### Phân tích attention weights

`summary_df` từ `train()` đã chứa `share_rad`, `share_ihc`, `share_gen`, `share_pdl1` per bệnh nhân.

```python
# Cluster bệnh nhân theo attention profile
from sklearn.cluster import KMeans
attention_profiles = summary_df[['share_rad', 'share_ihc', 'share_gen', 'share_pdl1']]
clusters = KMeans(n_clusters=3).fit_predict(attention_profiles)

# So sánh đặc điểm lâm sàng giữa các cluster
# KM curve theo cluster → cluster nào có prognosis tốt hơn?
```

**Kết quả kỳ vọng:**
- **Cluster "Genomics-dominant"**: bệnh nhân có driver mutation rõ (EGFR+, ALK+)
- **Cluster "Radiomics-dominant"**: bệnh nhân wild-type, heterogeneous tumor
- Ý nghĩa lâm sàng: gợi ý bệnh nhân nào cần xét nghiệm nào nhất

---

*Tổng hợp từ: `cai-tien-mo-hinh.md`, `huong-nghien-cuu-tiep-theo.md`, `huong-nghien-cuu-survival.md`*
