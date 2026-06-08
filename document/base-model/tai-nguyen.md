# Tài nguyên Học tập và Công cụ

---

## Phần 1 — Kaggle Notebooks về Attention và Multi-Modal Learning

### 1.1. Lộ trình học đề xuất

| Bước | Chủ đề | Thời gian | Từ khóa tìm kiếm trên Kaggle |
|------|--------|----------|------------------------------|
| 1 | Attention cơ bản | 1–2 tuần | `attention mechanism explained kaggle` |
| 2 | Attention cho Tabular | 1 tuần | `tabular neural attention kaggle` |
| 3 | Multi-Modal đơn giản | 2 tuần | `petfinder simple multimodal baseline` |
| 4 | Attention-based Fusion | 2 tuần | `attention fusion multimodal kaggle` |
| 5 | Missing Modality Handling | 1 tuần | `missing data multimodal kaggle` |
| 6 | Áp dụng vào dự án | 2–3 tuần | — |

### 1.2. Notebooks đề xuất theo chủ đề

**Attention Mechanism:**
- `"Attention Is All You Need - Pytorch Implementation"` — self-attention, multi-head, positional encoding
- `"Simple Attention Mechanism in PyTorch"` — attention cơ bản với ví dụ đơn giản
- `"TabNet: Attentive Interpretable Tabular Learning"` — attention cho dữ liệu bảng

**Multi-Modal:**
- `"PetFinder.my - Simple Multimodal Baseline"` — image + tabular, code đơn giản, dễ hiểu
- `"RSNA Pneumonia Detection - Multimodal Approach"` — y tế, image + metadata
- `"Cross-Modal Attention for Vision-Language Tasks"` — bidirectional attention

**Missing Modality:**
- `"Handling Missing Data in Multi-Modal Learning"` — masking, zero padding

**Code mẫu masking:**
```python
mask = torch.tensor([[1, 1, 0], [1, 0, 1]])  # [batch, n_modalities]
masked_features = features * mask.unsqueeze(-1)
```

**Code mẫu attention weights:**
```python
attention_scores = torch.matmul(query, key.transpose(-2, -1))
attention_weights = F.softmax(attention_scores, dim=-1)
output = torch.matmul(attention_weights, value)
```

### 1.3. Papers tham khảo

- **"Attention Is All You Need"** (Transformer) — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- **"DeepSurv"** (Cox loss với neural network) — [arXiv:1606.00931](https://arxiv.org/abs/1606.00931)
- **"Multimodal Machine Learning: A Survey"** — tìm trên arXiv

### 1.4. Cách tìm Notebooks trên Kaggle

1. Vào [kaggle.com](https://www.kaggle.com) → **Code**
2. Tìm: `attention mechanism pytorch`, `multimodal learning`, `feature fusion`, `missing modality`
3. Hoặc vào Competitions → filter tag `multimodal`, `tabular data`, `medical imaging`

### 1.5. Checklist kiến thức

- [ ] Giải thích được attention mechanism là gì
- [ ] Implement attention từ đầu bằng PyTorch
- [ ] Biết cách kết hợp nhiều nguồn dữ liệu
- [ ] Xử lý được missing modalities với masking
- [ ] So sánh được các phương pháp fusion khác nhau

---

## Phần 2 — Chuyển đổi Markdown sang PowerPoint

### Cách 1: Python script (khuyến nghị)

```bash
pip install python-pptx
cd code
python convert_to_pptx.py
```

Output: `MultiModalDynamicModel_Presentation.pptx` trong thư mục `code/`.

### Cách 2: Marp Online (dễ nhất)

1. Vào [marp.app](https://marp.app/)
2. Mở file `.md`
3. Export → PowerPoint (.pptx)

### Cách 3: Marp CLI

```bash
npm install -g @marp-team/marp-cli
cd code
marp MultiModalDynamicModel_Presentation.md --pptx
```

### Lưu ý

- Python script tự động format code blocks, bullet points, tables
- Marp giữ nguyên format markdown tốt hơn
- Cần Python 3.6+ cho script Python

---

*Tổng hợp từ: `tai-nguyen-hoc-tap.md`, `chuyen-doi-pptx.md`*
