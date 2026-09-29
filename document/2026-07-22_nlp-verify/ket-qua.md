# Kết quả: Kiểm chứng NLP-clinical trên 21 tổ hợp bài báo (single-run + 5-seed)

> Ngày: 2026-07-22 · Mã: `experiments/nlp_verify/` · Kết quả: `results/nlp_combos.json`
> Backbone **uniform_avg**, cùng `MODEL_PARAMS` (125 epoch, lr 0.01, alpha 0.001, folds 10, seeds [42,7,123,2024,31337]).
> Bảng đầy đủ: `document/2026-07-21_uniform-dyam-allcombo/bang-so-sanh-paper-combos.md` (section "Bảng NLP-CLINICAL").

## TL;DR — PHÁT HIỆN DƯƠNG ĐẦU TIÊN của cả đợt thực nghiệm

Mã hoá 13 biến lâm sàng thô (`cnl_dem_labs`) thành **câu chữ → embedding MiniLM 384-dim** (`cnl_nlp`, no_scale) rồi
đưa vào fusion. So 3 cách dùng lâm sàng cho MỖI tổ hợp: **base** (không lâm sàng) / **+Labs** (13 biến thô) / **+NLP**.

- **+NLP > +Labs ở 19/21 tổ hợp (5-seed)**, **18/21 (single-run)**.
- **Trong fusion đa nguồn (≥2 modality khác loại): +NLP thắng gần như tuyệt đối**, biên **+0.015 → +0.053** (5-seed),
  tới **+0.064** (single-run). +NLP là cao nhất trong 3 cách ở mọi tổ hợp đa nguồn.
- **BM1 (#18):** +NLP cao nhất cả hai kiểu đo — 5-seed **0.7832** (base 0.7720 · Labs 0.7661); single-run **0.8091**
  (base 0.7917 · Labs 0.7879). NLP là thứ **duy nhất trong dự án đẩy BM1 vượt 0.80**.
- **Đỉnh single-run toàn đợt = 0.8105** (#19, +NLP), **0.8091** (BM1, +NLP).

## So sánh công bằng: +NLP vs +Labs (cùng là lâm sàng)

| Kiểu đo | +NLP > +Labs | Biên trong fusion đa nguồn |
|---|---|---|
| 5-seed | 19/21 | +0.015 → +0.053 |
| single-run | 18/21 | +0.013 → +0.064 |

**Head-to-head sạch nhất — 2 combo có Labs của chính bài báo (#20, #21):** thay raw Labs bằng NLP trên đúng cấu hình
gốc → **+0.0156 / +0.0171 (5-seed)**, **+0.0194 / +0.0212 (single-run)**. NLP cải thiện *ngay tại* config bài báo.

## Vì sao 2 ngoại lệ (PDL1, Rad-LU) NLP không thắng

Đều là cấu hình **một-nguồn-mạnh** (PDL1) hoặc **radiomics-only** (Rad-LU). Trong uniform_avg (trung bình đều
theo mask), thêm BẤT KỲ nguồn lâm sàng yếu nào — thô hay NLP — cũng kéo điểm tổ hợp xuống. Đây là bản chất phép
trung bình 1-nguồn, không phải NLP kém: ở các combo này +Labs cũng < base.

## Ý nghĩa cho paper

Khác hẳn mọi hướng đã thử trước đó (attention DyAM/OvO, stacked fusion, foundation embedding ảnh gốc, TabPFN — đều
ÂM hoặc NGANG baseline), **NLP-clinical là can thiệp DUY NHẤT cho cải thiện dương, bền vững, nhất quán** trên nhiều
tổ hợp và ở cả hai kiểu đánh giá. Đây là đóng góp phương pháp mạnh nhất để đưa vào bài — biểu diễn dữ liệu lâm sàng
bằng ngôn ngữ tự nhiên tốt hơn đưa số thô vào mô hình fusion.

## Ghi chú kỹ thuật

- **base khớp cột `uniform_avg` cũ trong ~0.003** (vd #18 base 0.7720 vs uni cũ 0.7746). Lệch nhỏ do môi trường có
  monkeypatch `PowerTransformer._yeo_johnson_optimize` (scipy≥1.14 raise trên cột radiomics suy biến) → loại vài cột
  near-constant. Không ảnh hưởng so sánh nội bộ base/Labs/NLP (cả 3 chạy cùng môi trường).
- Embedding cache 1 lần: `../datasets/nlp_clinical_embed.parquet` (247×384).
- `no_scale=[vị trí cnl_nlp]` theo CLAUDE.md (embedding đã chuẩn hoá cosine, không RobustScaler).

## Assets
- `experiments/results/nlp_combos.json` — 21 combo × {base, labs, nlp} × {mean, sd, per_seed, single_run}
- `../datasets/nlp_clinical_embed.parquet` — embedding cache
- `experiments/nlp_verify/{nlp_modality,nlp_run,run_nlp}.py`
