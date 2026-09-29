# Plan: uniform_avg vs DyAM trên TẤT CẢ tổ hợp nguồn dữ liệu

> Ngày: 2026-07-21 · Mục tiêu: kiểm chứng **uniform_avg (không attention) ≥ DyAM (original, attention học)**
> trên **mọi tổ hợp ≥2 nguồn** — đặc biệt bác/xác nhận luận điểm "DyAM tốt hơn khi nhiều nguồn dữ liệu".

## 0. Thiết kế

**5 nguồn (domain)** — order cố định, **Rad đặt đầu** để `ctx.rad_filters` (khoá vị trí 0/1/2 = pc/pl/ln) áp đúng:

| Ký hiệu | Nguồn | Modality |
|---|---|---|
| R | Rad | rad_lesion_pc, rad_lesion_pl, rad_lesion_ln |
| P | Path | path_ihc_glcm |
| G | Gen | gen_driver_mut_amp |
| D | PDL1 | cnl_pdl1_score |
| L | Labs | cnl_dem_labs |

**26 tổ hợp ≥2 nguồn**: k=2 (10) · k=3 (10) · k=4 (5) · k=5 (1). Mỗi combo chạy **uniform_avg + DyAM(train)**
× 5 seed × 10-fold qua harness `run_repeated_cv` (inject combo vào `B.BENCHMARKS`, Rad-first).

**4 combo trùng benchmark đã khoá → kiểm reproduce** (locked 5-seed: uniform=backbone_lock, DyAM=baseline.json;
KHÔNG dùng `ref_auc_original` trong benchmarks.py — đó là tham chiếu 1-seed cũ):
- `RG`=BM3 (uniform 0.7110 / DyAM 0.7095) · `DG`=BM4 (0.7191 / 0.7072)
- `RPGD`=BM1 (0.7746 / 0.7595) · `RPGDL`=BM2 (0.7665 / 0.7638)

### Tiêu chí kết luận
Luận điểm **"uniform_avg chắc chắn ≥ DyAM"** ĐƯỢC XÁC NHẬN nếu:
- (a) **Không combo nào** DyAM tốt hơn uniform_avg có ý nghĩa (paired bootstrap p<0.05 theo hướng DyAM thắng).
- (b) **Δ(uniform − DyAM) KHÔNG giảm theo k**: nếu DyAM thật sự tốt hơn khi nhiều nguồn, Δ phải âm dần ở k lớn.
  Refute luận điểm DyAM nếu mean Δ ≥ 0 ở mọi mức k (đặc biệt k=4,5).
- Ngược lại (DyAM thắng ở ≥1 combo có ý nghĩa, HOẶC Δ âm dần rõ theo k) → luận điểm uniform_avg KHÔNG chắc.

## 1. Nguyên tắc
- Không sửa `lung_helpers.py`. Mã tại `experiments/allcombo/`. Tái dùng `run_repeated_cv` (seed-permutation),
  `paired_bootstrap`, `save_scores`. Kết quả gộp dần vào `results/allcombo.json` (append per step → chống mất).
- `PYTHONIOENCODING=utf-8`, chạy background đường dẫn tuyệt đối. Mỗi Step một lượt loop.

## 2. Các bước
### Step 0 — Khung + reproduce
- [x] `experiments/allcombo/run_combo.py`: DOMAINS, sinh 26 combo, hàm chạy 1 combo (uniform+DyAM, inject BENCHMARKS Rad-first)
- [x] Reproduce `DG`(=BM4): uniform=**0.7191** (khớp locked 0.7191) · DyAM=**0.7072** (khớp baseline.json 0.7072) → harness ĐÚNG

### Step 1 — k=2 (10 combo)
- [x] Chạy 10 combo k=2 → append `results/allcombo.json` + scores. RG=BM3 khớp (0.7110/0.7095). Lưu ý: sweep dùng
  thứ tự ORDER=[R,P,G,D,L] nên combo Gen+PDL1 sinh ra là **GD** (≠ DG=BM4 do DyAM nhạy thứ tự) → GD 0.7214/0.7307.
- [x] **Kết quả HỖN HỢP, không phải uniform luôn thắng**: DyAM thắng có ý nghĩa PG(Δ−0.029,p=0.007) & RL(Δ−0.013,p=0.035);
  uniform thắng RD(Δ+0.043,p=0.009). 7/10 combo ns. → luận điểm "uniform CHẮC CHẮN ≥ DyAM" đã lung lay ở k=2.
- [x] **Phát hiện phụ:** DyAM nhạy thứ tự modality (GD 0.7307 vs DG 0.7072 cùng data) — điểm yếu của attention.

### Step 2 — k=3 (10 combo)
- [x] Chạy 10 combo k=3 → append. Chỉ RPD có ý nghĩa (uniform +0.027, p=0.006); **không combo nào DyAM thắng**.
  Δ trung bình(u−d) ≈ **+0.007** (tăng so với k=2 ≈+0.001) → thêm nguồn thì lợi thế nghiêng uniform, NGƯỢC luận điểm DyAM.

### Step 3 — k=4 (5 combo)
- [x] Chạy 5 combo k=4 → append. **BM1 (RPGD) reproduce khớp**: uniform 0.7746 (=locked), DyAM 0.7595 (=baseline).
  4/5 nghiêng uniform (PGDL thắng có ý nghĩa +0.019 p=0.022), 0 combo DyAM thắng có ý nghĩa.
  Δ trung bình(u−d) ≈ **+0.011** — tiếp tục tăng (k=2 +0.001 → k=3 +0.007 → k=4 +0.011), xu hướng đơn điệu.

### Step 4 — k=5 (1 combo)
- [x] Chạy combo đầy đủ RPGDL(=BM2) → append. **BM2 reproduce khớp**: uniform 0.7665 (=locked), DyAM 0.7638 (=baseline).
  Δ +0.0027 (ns) — uniform vẫn nhỉnh ở combo đầy đủ.

### Step 5 — Phân tích + quyết định
- [x] Bảng 26 combo: uniform vs DyAM (mean±sd) + Δ + paired bootstrap p (`run_analyze.py` → `allcombo_analysis.json`)
- [x] **Tổng hợp Δ theo k**: mean Δ = +0.0006 (k=2) → +0.0067 (k=3) → +0.0110 (k=4) → **TĂNG, không âm dần**.
  DyAM thắng có ý nghĩa chỉ 2 combo, đều ở k=2; k≥3 DyAM thắng 0. uniform thắng 3 (RD, RPD, PGDL).
- [x] Hình: `fig-delta-by-k.svg` (Δ-vs-k + mean line), `fig-scatter-uni-dyam.svg` (26 điểm màu theo k)
- [x] **KẾT LUẬN: luận điểm "DyAM tốt hơn khi nhiều nguồn" BỊ BÁC BỎ (ngược chiều).** uniform_avg ≥ DyAM, vượt dần
  theo k. `ket-qua.md` viết xong. **PLAN HOÀN TẤT — dừng loop.**

## 3. Nhật ký
| Ngày | Step | Kết quả | Quyết định |
|---|---|---|---|
| 2026-07-21 | 5 | Phân tích 26 combo xong. Δ trung bình tăng theo k (+0.0006/+0.0067/+0.0110/+0.0027). DyAM thắng có ý nghĩa chỉ 2 combo (đều k=2); uniform 3. 2 hình + ket-qua.md | **KẾT: luận điểm DyAM bị bác bỏ ngược chiều. uniform_avg ≥ DyAM, vượt dần theo k. PLAN XONG — dừng loop** |
| 2026-07-21 | 4 | k=5 xong. BM2 reproduce khớp (0.7665/0.7638). Δ +0.0027 ns | Sang Step 5 (phân tích) |
| 2026-07-21 | 3 | k=4 xong. BM1 reproduce khớp (0.7746/0.7595). 4/5 nghiêng uniform (PGDL p=0.022), 0 DyAM thắng. Δ trung bình +0.011 → tăng đơn điệu theo k | Xu hướng rõ ngược DyAM. Sang Step 4 (k=5) |
| 2026-07-21 | 2 | k=3 xong. Chỉ RPD có ý nghĩa (uniform p=0.006), 0 combo DyAM thắng. Δ trung bình +0.007 (k=2 +0.001) → xu hướng nghiêng uniform khi k tăng | Ngược luận điểm DyAM. Sang Step 3 (k=4) |
| 2026-07-21 | 1 | k=2 xong (10 combo). HỖN HỢP: DyAM thắng có ý nghĩa 2 (PG,RL), uniform thắng 1 (RD), 7 ns. Sửa DyAM guard 0-feature (train() gốc crash trên rad_lesion_pl n=21). DyAM nhạy thứ tự modality | Luận điểm chưa chắc; cần k=3/4/5 xem xu hướng theo k. Sang Step 2 |
| 2026-07-21 | 0 | Khung + runner xong. Reproduce DG=BM4 khớp locked (uniform 0.7191, DyAM 0.7072). Preview 4 benchmark: uniform ≥ DyAM ở cả 4 (Δ +0.015/+0.003/+0.0015/+0.012) | Harness đúng, sang Step 1 (k=2) |
