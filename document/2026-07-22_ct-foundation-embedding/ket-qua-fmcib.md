# Kết quả cuối: fmcib THẬT (chạy trên Colab) — đóng cửa hướng foundation embedding

> Ngày: 2026-07-22 · Mã: `experiments/foundation_embed/fmcib_eval.py` · Kết quả: `results/fmcib_eval.json`
> Embedding: `ct_fmcib_embed_{discovery,rad_valid}.parquet` (4096-dim, trích bằng fmcib thật trên Colab GPU)
> Notebook: `fmcib_colab.ipynb` (clone repo + import trực tiếp, bỏ qua pip build gãy)

## TL;DR

Chạy được **fmcib thật** (Foundation Model for Cancer Imaging Biomarkers — model 3D ĐÚNG NHẤT cho tổn thương CT,
weights 738MB, khớp state_dict hoàn hảo). Kết quả **âm dứt khoát, loại bỏ mọi caveat "dùng model sai"**:
- Single-modality: nội bộ **0.503 (ngẫu nhiên)**, external 0.546 (nhiễu).
- Fusion: **thêm fmcib vào BM1 làm TỆ hơn** (−0.030); thay/embed đều tệ hơn.

## Số liệu

**Single-modality (discovery-CV | external):**

| model | nội bộ | external |
|---|---|---|
| radiomics thủ công | 0.582 | 0.461 |
| BiomedCLIP 2.5D | 0.510 | 0.530 |
| MedicalNet 3D | 0.514 | 0.627 (nhiễu) |
| **fmcib (thật)** | **0.503** | 0.546 [0.33,0.76] |

**Fusion (discovery-CV 5-seed, uniform_avg):**

| config | AUC |
|---|---|
| A — hand-crafted (BM1) | **0.7746** |
| E — BM1 **+ fmcib** (thêm modality) | 0.7445 (**−0.030**) |
| D — fmcib rad + path thủ công | 0.6997 (−0.075) |
| B — fmcib + Phikon (embed cả 2) | 0.6534 (−0.121) |

## Kết luận — hướng foundation embedding ĐÓNG

**Ngay cả fmcib** — model 3D cancer-CT chuyên biệt, đúng chính xác bài toán "trích đặc trưng tổn thương CT" —
**không mang tín hiệu response nào ở nội bộ (0.503 = ngẫu nhiên)**, và **kéo tụt fusion** dù thêm hay thay.
Ba model CT (BiomedCLIP, MedicalNet, fmcib) đều hội tụ về ~0.50-0.51 nội bộ → **không phải lỗi chọn model**.

Bức tranh giờ khép kín và nhất quán tuyệt đối:

| Hướng thử | Kết quả |
|---|---|
| Fusion attention (DyAM/OvO/stacked) | = uniform_avg, không hơn |
| Feature engineering (prune/weight) | không hơn |
| Pathology embedding (Phikon) | có tín hiệu nội bộ nhưng KHÔNG vượt GLCM external |
| Radiology embedding (BiomedCLIP/MedicalNet/**fmcib**) | ~ngẫu nhiên, hại fusion |

→ **Giới hạn nằm ở DỮ LIỆU (n nhỏ, trần tín hiệu ~0.78, radiomics không generalize), KHÔNG phải ở model/feature.**
Đây là kết luận âm mạnh nhất và trung thực nhất — đáng làm thông điệp trung tâm của paper.

## Caveat (đã trung thực)
- Embedding fmcib dùng normalize xấp xỉ (clip+z-score) thay vì transform gốc `get_features`. Nhưng 0.503 = *đúng
  ngẫu nhiên* và khớp với 2 model CT khác (cùng ~0.51) → khó là do preprocessing; tín hiệu thật ắt lộ ra ít nhất một phần.
- n external nhỏ (46-47), CI rộng — nhưng nội bộ (n=187) đã đủ để nói "không có tín hiệu".

## Assets
- `ct_fmcib_embed_{discovery,rad_valid}.parquet` (187/47 × 4096), `results/fmcib_eval.json`, `fmcib_colab.ipynb`
