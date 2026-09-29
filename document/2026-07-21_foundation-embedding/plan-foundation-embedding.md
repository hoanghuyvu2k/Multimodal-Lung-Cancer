# Plan: Foundation-model embedding thay feature thủ công (Pathology, Phase 1)

> Ngày tạo: 2026-07-21
> Ý tưởng: nút thắt là **feature**, không phải fusion. Thay GLCM/radiomics thủ công bằng embedding từ
> foundation model chạy trên ẢNH GỐC, và đo **generalization ngoài mẫu** (đòn bẩy thật).
> Phase 1 = pathology (WSI → Phikon). Phase 2 = radiology (CT → CT-FM), lên plan riêng sau khi có kết quả P1.

## 0. Bối cảnh & giả thuyết

Đã khóa từ các dự án trước:
- AUC nội bộ kịch trần ~0.78; DyAM/OvO/uniform ngang nhau → fusion không phải đòn bẩy.
- Feature thủ công: **pathology generalize** (ext GLCM ~0.744–0.767), **radiomics KHÔNG** (ext ~0.43–0.66, bất ổn).

**Giả thuyết test:** embedding pathology foundation model (Phikon) — giàu tín hiệu + bất biến scanner hơn GLCM —
**có vượt GLCM external (~0.744) không**, và có nâng combo đa nguồn không. Thí nghiệm đối chứng sạch: giữ nguyên
cohort/fusion/eval, **chỉ đổi bộ trích feature** (GLCM thủ công → Phikon embedding).

### Dữ liệu ảnh gốc (đã xác nhận)
- `../datasets/pathology/LUNG_18-193/slides/*.svs` (235 WSI) + `halo/*.annotations` (vùng u).
- (Phase 2) `../datasets/radiology/LUNG_18-193/volumes/{id}/SCANS/1/*.nii` (238 CT) + `segmentations/*.mha`.
- Môi trường: **GPU RTX 3060 (CUDA OK)**. huggingface_hub có; thiếu openslide/timm/transformers-vision/nibabel/cv2/skimage/h5py.

### Model
- **Phikon** (`owkin/phikon`, ViT-B, KHÔNG gated) — dùng ngay. Nâng cấp CONCH/UNI (gated) sau nếu user xin được quyền.
- Gộp tile → bệnh nhân: **mean-pool** trước (rẻ, không train, không rò rỉ); ABMIL để stretch sau.

### Tiêu chí thành công (điểm chốt)
- **CHÍNH:** external pathology (train discovery → test path_valid). Phikon-embed **> GLCM 0.744** → giả thuyết đúng.
- **PHỤ:** discovery-CV không thua GLCM; và khi thay vào combo đa nguồn (BM1) không giảm.
- Nếu Phikon KHÔNG hơn GLCM external → vẫn là kết quả thật (GLCM thủ công đã đủ tốt cho pathology) — ghi trung thực.

## 1. Nguyên tắc thực thi (kế thừa)
- Mã tại `experiments/foundation_embed/`. **KHÔNG sửa `lung_helpers.py`.** Tích hợp mirror `clinical_nlp_embedding.py`
  (embedding → modality_dict, **`no_scale`**). Kết quả ra `experiments/results/fm_*.json`, embedding cache ra parquet.
- Kỷ luật token: mỗi Step một lượt loop; job nặng chạy background, đường dẫn tuyệt đối, `PYTHONIOENCODING=utf-8`.
- Job GPU dài → smoke vài slide trước khi batch cả cohort.

## 2. Các bước (Phase 1 — Pathology)

### Step 0 — Khung + deps + mapping
- [x] Cài stack ảnh (INSTALL_OK: openslide-bin, timm, transformers, nibabel, opencv, scikit-image, h5py)
- [x] Verify: **Phikon OK** (CUDA, embed_dim **768**); **openslide OK** (WSI level0 = **20x**, 0.5 mpp → tile thẳng); HALO annotation = XML có vùng **"Tumor"** (polygon)
- [x] Mapping: **163 discovery** (qua dmp_pt_id) + **71 path_valid** (slide_id = key trực tiếp) = 234. Lưu `fm_slide_map.csv`
- [x] Ghi `results/fm_setup.json`

### Step 1 — Module trích embedding + smoke
- [x] `pathology_fm_embedding.py`: parse vùng Tumor (HALO XML) → tile 256²@20x → Phikon CLS → mean-pool → vector 768. Cache .npy
- [x] Smoke 5 slide OK: source=tumor, dim 768, norm ~43–45, **1–10s/slide** (batch 234 ≈ ~15–20 phút)

### Step 2 — Batch trích discovery
- [x] `step2_batch.py discovery`: **163/163 slide OK, 0 fail** → `path_fm_embed_discovery.parquet` (163×768), ~15 phút

### Step 3 — Batch trích external
- [x] `step2_batch.py path_valid`: **71/71 slide OK, 0 fail** → `path_fm_embed_path_valid.parquet` (71×768, index=slide_id)

### Step 4 — So sánh discovery-CV (Phikon vs GLCM)
- [x] `step45_eval.py` (LR, 5-seed 10-fold): **Phikon 0.680±0.018 > GLCM 0.630±0.017** (+0.05). Foundation embedding
  giàu tín hiệu HƠN nội bộ. Ghi `results/fm_pathology.json`

### Step 5 — External validation (ĐIỂM CHỐT)
- [x] Train discovery → test path_valid (cùng 52 bệnh nhân): **Phikon 0.697 [0.53,0.85] vs GLCM 0.765 [0.61,0.89]**,
  Δ=−0.068 (CI chồng nhiều → không có ý nghĩa). **Phikon KHÔNG vượt GLCM external.**
- [x] Trả lời giả thuyết: **với mean-pool + Phikon, KHÔNG cải thiện generalization.** Caveat: mean-pool là cách gộp
  thô nhất (ABMIL thường hơn nhiều); n external nhỏ (52), CI rộng; GLCM là feature PD-L1 IHC *đã* rất khớp task.

### Step 6 — Tổng hợp + đa nguồn
- [x] Hình `fig-phikon-vs-glcm.svg` (discovery-CV vs external, Phikon vs GLCM)
- [x] Quyết định: **Phase 1 (Phikon+mean-pool) không cải thiện external → KHÔNG bác bỏ ý tưởng (mean-pool thô);
  Phase 2 = ABMIL + CONCH, hoặc chuyển sang radiology (dư địa lớn hơn).** Viết `ket-qua.md`
- [~] (Bỏ qua) swap đa nguồn BM1: external đã trả lời câu hỏi cốt lõi, swap nội bộ không đổi kết luận → để Phase 2
- [x] **PLAN PHASE 1 HOÀN TẤT — dừng loop.**

## 3. Nhật ký kết quả
| Ngày | Step | Kết quả | Quyết định |
|---|---|---|---|
| 2026-07-22 | 4-6 | Phikon vs GLCM: nội bộ 0.680 vs 0.630 (Phikon hơn); external 0.697 vs 0.765 (Phikon KHÔNG hơn, CI chồng). Hình + ket-qua.md xong | **Phase 1 xong.** mean-pool không đủ; Phase 2 = ABMIL/CONCH hoặc radiology. Dừng loop |
| 2026-07-22 | 2-3 | Batch OK: discovery 163/163, path_valid 71/71 → parquet 768-dim (đã cache) | Sang Step 4-5 (so sánh) |
| 2026-07-21 | 1 | Module xong. Smoke 5 slide OK (tumor region, 768-dim, 1–10s/slide) | Pipeline chạy tốt, sang Step 2 (batch discovery) |
| 2026-07-21 | 0 | Deps OK. Phikon OK (768-dim, CUDA). openslide OK (level0=20x). HALO có vùng Tumor. Mapping 163 disc + 71 path_valid | Khung sẵn sàng, sang Step 1 (module + smoke) |

## 4. Ghi chú kỹ thuật / rủi ro
- openslide trên Windows: dùng `openslide-bin` (bundle DLL) để tránh cài binary tay.
- `.annotations` HALO thường là XML polygon (micron/pixel) — parse vùng u; nếu vướng → fallback Otsu tissue mask.
- Chọn đúng mag: đọc `openslide` properties (`objective-power`), lấy level gần 20x, resize về 224/256 cho Phikon.
- Chống rò rỉ: mean-pool không học nên không rò rỉ; RobustScaler/`no_scale` như NLP module (embedding đã chuẩn hóa).
- Phase 2 (radiology CT-FM) plan riêng: cần chọn model (Merlin/CT-FM), crop ROI theo `.mha`, nặng hơn.
