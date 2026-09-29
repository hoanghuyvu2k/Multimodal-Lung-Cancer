# Plan Phase 3: CT foundation model CHUYÊN DỤNG (fmcib) thay BiomedCLIP

> Ngày tạo: 2026-07-22
> Vì sao: Phase 2 âm nhưng **BiomedCLIP là model sai** (vision-language tổng quát, 2.5D). Phase 3 dùng model
> ĐÚNG chỗ: **fmcib** (Foundation Model for Cancer Imaging Biomarkers, Pai 2024) — 3D ResNet50 SimCLR pretrain
> trên ~11k lesion CT, thiết kế riêng để trích embedding lesion-centered. Đây là phép thử CÔNG BẰNG cho giả thuyết
> "CT foundation embedding phá được rào generalization của radiomics (~0.46–0.60)".

## 0. Bối cảnh & giả thuyết

- Radiomics thủ công: external ~0.46–0.60, gãy generalization → dư địa lớn nhất.
- Phase 2 BiomedCLIP: 0.51 nội bộ (ngẫu nhiên) → model tổng quát không mã hoá texture CT. KHÔNG phải kết luận về CT-FM.
- **fmcib khác hẳn:** 3D, pretrain trên CT lesion, input = patch 50mm³ quanh tâm lesion → embedding 4096-dim.

**Giả thuyết:** fmcib-embed **> radiomics** (nội bộ) và nhất là **phá rào external ~0.46–0.60**. Đối chứng sạch:
giữ cohort/eval/mask, chỉ đổi bộ trích feature. Dùng lại mapping + mask + series-match của Phase 2.

### Dữ liệu / môi trường
- Dùng lại `ct_slide_map.csv` (190 disc + 47 rad_valid), mask `.mha`, series-match-by-size (Phase 2 đã giải).
- GPU RTX 3060. Cần cài **fmcib** (`foundation-cancer-image-biomarker`) + monai. Rủi ro: dep nặng/xung đột → Step 0 verify,
  fallback **MONAI/MedicalNet 3D ResNet** nếu fmcib không load.

### fmcib pipeline (điểm khác Phase 2)
- Input = **patch 3D 50×50×50 @ 1mm** center tại **tâm lesion** (centroid từ mask, world coords), normalize theo fmcib.
- Model → vector **4096-dim** (không mean-pool slice; là 3D thật).

### Tiêu chí (điểm chốt)
- **CHÍNH:** external rad_valid. fmcib **> radiomics (~0.46–0.60)** và **> BiomedCLIP (~0.5)** → giả thuyết đúng.
- **PHỤ:** discovery-CV có tín hiệu thật (>> 0.51 của BiomedCLIP; ideally ≥ radiomics 0.58).
- Nếu fmcib cũng ~0.5 external → bằng chứng mạnh: **giới hạn ở dữ liệu, không phải feature** (kết luận âm đáng viết).

## 1. Nguyên tắc (kế thừa)
- Mã `experiments/foundation_embed/ct3d_*`. KHÔNG sửa lung_helpers. Kết quả `results/fm_ct3d.json`, cache `.npy`.
- Mỗi Step một lượt loop; job GPU background + smoke trước batch. utf-8, path tuyệt đối.

## 2. Các bước

### Step 0 — Deps + verify fmcib
- [x] fmcib package **build FAIL trên Windows** (setuptools≥81 bỏ pkg_resources; --no-build-isolation vẫn fail) và
  weights không có ở HF `surajpaib/fmcib` (404). → **Chuyển FALLBACK: MONAI MedicalNet resnet50 3D pretrained** (23 bộ 3D y khoa, feat 2048).
- [x] Verify: MONAI resnet50(spatial_dims=3, pretrained=True) load OK, forward [1,1,64³] → [1,2048]. monai 1.6.0.
- [x] **Lưu ý trung thực:** đây là MedicalNet (3D y khoa tổng quát) chứ KHÔNG phải fmcib (cancer-CT chuyên biệt);
  vẫn là phép thử 3D-foundation công bằng hơn BiomedCLIP, nhưng yếu hơn fmcib. fmcib thật cần chạy ở Linux/Colab.

### Step 1 — Module + smoke
- [x] `ct3d_embedding.py`: volume+mask → centroid lesion → patch 64³ voxel → z-score → MedicalNet resnet50 → 2048-dim. Cache
- [x] Smoke 5 volume OK: 2048-dim, ~1–1.7s/volume (batch nhanh)

### Step 2 — Batch discovery
- [x] `ct3d_batch.py discovery`: 188/190 → `ct3d_fm_embed_discovery.parquet` (188×2048)

### Step 3 — Batch rad_valid
- [x] `ct3d_batch.py rad_valid`: 47/47 → `ct3d_fm_embed_rad_valid.parquet` (47×2048)

### Step 4 — Discovery-CV
- [x] MedicalNet 3D **0.514±0.027 (≈ngẫu nhiên)** < radiomics 0.582. Không bắt được tín hiệu nội bộ. Ghi `fm_ct3d.json`

### Step 5 — External (ĐIỂM CHỐT)
- [x] MedicalNet 3D external 0.627 [0.44,0.81] vs radiomics 0.461 vs BiomedCLIP 0.530. Nhìn cao nhất **NHƯNG NHIỄU**
  (nội bộ ngẫu nhiên → external không interpretable; CI phủ dưới 0.5). **Không phá rào một cách đáng tin.**

### Step 6 — Tổng hợp
- [x] Hình `fig-ct3-models.svg`; **quyết định: Phase 3 âm (fallback MedicalNet không có tín hiệu; fmcib thật chưa test được)**;
  viết `ket-qua-phase3.md`. **PLAN HOÀN TẤT — dừng loop.**

## 3. Nhật ký
| Ngày | Step | Kết quả | Quyết định |
|---|---|---|---|
| 2026-07-22 | 4-6 | MedicalNet 3D nội bộ 0.514 (≈ngẫu nhiên) < radiomics 0.582; external 0.627 nhưng NHIỄU (CI [0.44,0.81], nội bộ random). Hình + ket-qua-phase3 xong | **Âm. fmcib thật chưa chạy được (Windows). Dừng loop** |
| 2026-07-22 | 0-3 | fmcib build fail → fallback MedicalNet 3D (2048-dim). Module + batch OK (188/47 volume) | Sang eval |

## 4. Ghi chú / rủi ro
- fmcib có thể cần seed-point CSV + wrapper riêng; nếu API rườm rà, gọi trực tiếp backbone (load state_dict 3D ResNet50) + preprocess tay.
- Patch có thể lệch nếu spacing z lớn (một số CT lát dày ~5mm); resample về 1mm theo fmcib.
- Fallback MedicalNet/MONAI nếu weights fmcib khó tải.
