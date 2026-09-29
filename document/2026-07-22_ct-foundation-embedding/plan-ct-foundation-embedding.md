# Plan: CT foundation embedding thay radiomics thủ công (Phase 2 — Radiology)

> Ngày tạo: 2026-07-22
> Ý tưởng (tiếp Phase 1): nút thắt là feature. **Radiomics CT external chỉ ~0.46–0.60** (gãy generalization) →
> dư địa LỚN NHẤT. Thay radiomics thủ công bằng embedding foundation model trên CT gốc, đo generalization external.

## 0. Bối cảnh & giả thuyết

Từ trước:
- Radiomics thủ công: nội bộ ~0.65–0.71 nhưng **external ~0.46 (của tôi) / ~0.60 (bài báo), rất bất ổn** → KHÔNG generalize.
- Pathology GLCM đã tốt (external 0.765); Phikon mean-pool không vượt được (Phase 1). Nên chuyển sang chỗ *thật sự gãy*.

**Giả thuyết:** embedding CT foundation model — bất biến scanner hơn radiomics — **có phá được rào external ~0.46–0.60**
của radiomics không? Đối chứng sạch: giữ cohort/eval, chỉ đổi bộ trích feature (radiomics thủ công → CT-embed).
**Đây là chỗ có upside rõ nhất** (pathology thủ công vốn đã tốt, radiomics thì không).

### Dữ liệu (đã xác nhận)
- `../datasets/radiology/LUNG_18-193/volumes/{acc}/SCANS/{k}/*_volumetric_image.nii` (237 volume, NHIỀU series/volume).
- `../datasets/radiology/LUNG_18-193/segmentations/{acc}_{nh|amp}.mha` (238 mask tổn thương).
- Map: `did_acc`/`radiology_accession_number` → 237/237 khớp. Cohort: discovery (~187 có radiomics) vs rad_valid (50).
- Môi trường: GPU RTX 3060. Cần cài thêm **SimpleITK** (.mha/.nii) + **open_clip_torch** (BiomedCLIP). nibabel đã có.

### Model
- **BiomedCLIP** (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`, open_clip, **KHÔNG gated**) — image encoder
  cho lát cắt CT (2.5D). 512-dim. Cân nhắc CT-FM 3D chuyên dụng sau nếu load được dễ.
- Gộp lát → bệnh nhân: **mean-pool** trước (như Phase 1); ABMIL sau nếu có tín hiệu.

### Xử lý kỹ thuật cần giải (Step 0/1)
- **Chọn series khớp mask:** mỗi volume nhiều SCANS; chọn series có geometry (size/spacing/origin) trùng `.mha`
  (SimpleITK), hoặc resample mask về ảnh.
- **Window HU:** tổn thương phổi = mô mềm trong nền phổi. Default **soft-tissue (WL 40, WW 400)**; thử lung window nếu cần.
- Crop ROI quanh lesion (bbox + margin), lấy các lát axial cắt qua lesion, resize 224.

### Tiêu chí thành công (điểm chốt)
- **CHÍNH:** external rad_valid. CT-embed **> radiomics external (~0.46–0.60)** → giả thuyết đúng, đóng góp lớn.
- **PHỤ:** discovery-CV không thua radiomics quá nhiều.
- Nếu CT-embed cũng ~0.5 external → bằng chứng mạnh rằng tín hiệu radiomics *thực sự* không transfer (không phải lỗi feature).

## 1. Nguyên tắc thực thi (kế thừa)
- Mã tại `experiments/foundation_embed/` (thêm file `ct_*`). KHÔNG sửa `lung_helpers.py`.
- Kết quả ra `results/fm_ct*.json`, embedding cache parquet + `.npy`. Mỗi Step một lượt loop; job GPU background + smoke trước batch.
- `PYTHONIOENCODING=utf-8`, đường dẫn tuyệt đối.

## 2. Các bước

### Step 0 — Khung + deps + mapping + geometry
- [x] Cài SimpleITK + open_clip_torch. **BiomedCLIP OK** (CUDA, 512-dim)
- [x] Map: **190 discovery** (qua dmp_pt_id) + **47 rad_valid** (acc = key). Lưu `ct_slide_map.csv`
- [x] **Series matching OK**: chọn đúng series `.nii` khớp size mask `.mha` (vd 190323: 7 series → series 2, mask nhị phân, 12598 voxel). Ghi `results/ct_setup.json`

### Step 1 — Module trích embedding + smoke
- [x] `ct_fm_embedding.py`: volume+mask → series khớp → window soft-tissue → lát axial qua lesion (crop bbox) → BiomedCLIP → mean-pool → 512-dim. Cache .npy
- [x] Smoke 5 volume OK: 3–23 lát/volume, 512-dim, **0.7–1.5s/volume** (batch 237 ≈ ~5–10 phút)

### Step 2 — Batch discovery
- [x] `ct_step2_batch.py discovery`: **188/190 OK** (2 mask hỏng bỏ qua) → `ct_fm_embed_discovery.parquet` (188×512), ~3 phút

### Step 3 — Batch rad_valid
- [x] `ct_step2_batch.py rad_valid`: **47/47 OK** → `ct_fm_embed_rad_valid.parquet` (47×512), ~55s

### Step 4 — Discovery-CV (CT-embed vs radiomics)
- [x] 5-seed CV LR: **CT-embed 0.510±0.019 (≈ngẫu nhiên) < radiomics 0.582±0.024**. Ghi `results/fm_ct.json`

### Step 5 — External validation (ĐIỂM CHỐT)
- [x] External (cùng n=46): CT-embed 0.519 [0.35,0.72] vs radiomics 0.461 [0.23,0.68]. Nominal +0.058 nhưng **vô nghĩa**
  vì nội bộ CT-embed đã ngẫu nhiên (0.51) → model chưa học được gì.
- [x] Trả lời: **KHÔNG** — CT-embed không phá được rào, và bản thân nó cũng không có tín hiệu (BiomedCLIP sai model).

### Step 6 — Tổng hợp
- [x] Hình `fig-ct-vs-radiomics.svg`; **quyết định: Phase 2 âm rõ**; viết `ket-qua.md`. **PLAN HOÀN TẤT — dừng loop.**

## 3. Nhật ký kết quả
| Ngày | Step | Kết quả | Quyết định |
|---|---|---|---|
| 2026-07-22 | 4-6 | CT-embed nội bộ 0.510 (≈ngẫu nhiên) < radiomics 0.582; external cả hai ~0.5. BiomedCLIP không có tín hiệu cho task. Hình + ket-qua xong | **ÂM rõ. Phase 2 xong.** BiomedCLIP sai model; Phase 3 cần CT-FM 3D. Dừng loop |
| 2026-07-22 | 1-3 | Module CT OK (smoke 0.7-1.5s/vol). Batch discovery 188/190, rad_valid 47/47 → parquet 512-dim | Sang Step 4-5 |
| 2026-07-22 | 0 | Deps OK. BiomedCLIP OK (512-dim). Series-match-by-size hoạt động. Mapping 190 disc + 47 rad_valid | Khung sẵn sàng, sang Step 1 |

## 4. Ghi chú / rủi ro
- Nhiều series/volume: rủi ro chọn nhầm → khớp theo size mask là an toàn nhất.
- `.mha` nhãn nh/amp: 1 mask/volume, dùng trực tiếp; nếu mask nhiều nhãn (nhiều lesion) → gộp hoặc lấy lesion lớn nhất.
- BiomedCLIP 2.5D là bản khả thi; nếu external vẫn ~0.5 thì cân nhắc CT-FM 3D chuyên dụng trước khi kết luận.
- Chống rò rỉ: mean-pool không học; window/normalize cố định; RobustScaler/no_scale như Phase 1.
