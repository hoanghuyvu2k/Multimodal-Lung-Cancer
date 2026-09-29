# Kết quả: uniform_avg vs DyAM trên tất cả tổ hợp nguồn dữ liệu

> Ngày: 2026-07-21 · Mã: `experiments/allcombo/` · Kết quả: `results/allcombo.json`, `allcombo_analysis.json`
> Hình: `fig-delta-by-k.svg`, `fig-scatter-uni-dyam.svg`

## TL;DR

Luận điểm "**DyAM tốt hơn khi nhiều nguồn dữ liệu**" **BỊ BÁC BỎ — dữ liệu cho kết quả NGƯỢC LẠI**: chạy đủ
**26 tổ hợp ≥2 nguồn** (5 domain Rad/Path/Gen/PDL1/Labs), Δ(uniform − DyAM) **tăng theo số nguồn k**:
+0.0006 (k=2) → +0.0067 (k=3) → +0.0110 (k=4). DyAM chỉ thắng có ý nghĩa ở **2 combo, đều tại k=2** (ít nguồn
nhất); từ k≥3 DyAM **không thắng combo nào**, còn uniform thắng có ý nghĩa 3 lần. **Càng nhiều nguồn, uniform_avg
càng vượt DyAM** — đúng chiều ngược với kỳ vọng.

## 1. Thiết kế
26 tổ hợp của 5 domain, mỗi combo chạy **uniform_avg** (`train_uniform_avg`) và **DyAM** (`MultiModalDynamicModel`,
attention học) × 5 seed × 10-fold qua harness `run_repeated_cv` (seed-permutation), paired bootstrap trên bệnh nhân.
Cả hai model dùng **cùng thứ tự nguồn + cùng L1 filter + cùng cách bỏ fold 0-feature** → so sánh công bằng.
Reproduce khớp 4 benchmark khoá: BM1 (0.7746/0.7595), BM2 (0.7665/0.7638), BM3 (0.7110/0.7095), BM4-order (RG).

## 2. Δ(uniform − DyAM) theo số nguồn k — câu hỏi chính

| k | #combo | mean Δ | median Δ | uniform thắng (p<.05) | DyAM thắng (p<.05) |
|---|---|---|---|---|---|
| 2 | 10 | +0.0006 | −0.0018 | 1 (RD) | **2 (PG, RL)** |
| 3 | 10 | +0.0067 | +0.0032 | 1 (RPD) | 0 |
| 4 | 5 | +0.0110 | +0.0151 | 1 (PGDL) | 0 |
| 5 | 1 | +0.0027 | — | 0 | 0 |

`fig-delta-by-k.svg`: đường mean Δ đi lên từ ~0 (k=2) đến +0.011 (k=4). Điểm xanh (uniform thắng) tụ **trên** 0
khi k≥3; điểm cam (DyAM) chỉ vượt trội ở k=2. `fig-scatter-uni-dyam.svg`: các điểm k cao (k=4,5) nằm **trên đường
chéo** (uniform ≥ DyAM); điểm k=2 rải hai bên.

## 3. Đọc kết quả (trung thực)

- **Luận điểm "DyAM tốt hơn khi nhiều nguồn": SAI, và sai ngược chiều.** DyAM chỉ có lợi thế (2 combo có ý nghĩa:
  Path+Gen, Rad+Labs) khi **ít nguồn nhất (k=2)**. Thêm nguồn thì lợi thế đó biến mất và uniform vượt lên.
- **"uniform_avg chắc chắn ≥ DyAM": đúng ở chế độ đa nguồn, KHÔNG tuyệt đối ở mọi combo.** Trên 26 combo, uniform
  nhỉnh mean ở 14/26; thắng có ý nghĩa 3 (RD, RPD, PGDL) vs DyAM 2 (PG, RL). Hai ngoại lệ của DyAM đều ở k=2.
  → Phát biểu chính xác: **uniform_avg ít nhất ngang DyAM, và vượt dần khi số nguồn tăng; ở k≥3 DyAM không thắng
  combo nào.** Với một mô hình "fusion đa nguồn" thì đây đúng là chế độ quan trọng.
- **Phát hiện phụ:** DyAM **nhạy thứ tự modality** (Gen+PDL1: thứ tự [Gen,PDL1]→0.7307 vs [PDL1,Gen]→0.7072).
  Một mô hình fusion không nên phụ thuộc thứ tự đầu vào tùy ý — điểm yếu của attention có tham số, uniform_avg
  (đối xứng) không mắc.

## 4. Kết luận cho câu hỏi ban đầu

Không có bằng chứng nào cho thấy DyAM tận dụng nhiều nguồn tốt hơn uniform_avg; **bằng chứng chỉ theo chiều ngược**:
attention học của DyAM không giúp gì khi nguồn tăng, chỉ thêm phương sai và phụ thuộc thứ tự. Khuyến nghị giữ nguyên:
**dùng uniform_avg + seed-ensemble** làm mô hình fusion cho bài toán này. Nếu cần phòng thủ trước phản biện "nhưng
DyAM đôi khi thắng", nêu trung thực: DyAM chỉ thắng ở 2 cấu hình 2-nguồn cụ thể (Path+Gen, Rad+Labs), không phải
xu hướng — và mất ưu thế ngay khi thêm nguồn thứ ba.

## 5. Liên kết
- `experiments/allcombo/{run_combo,run_analyze}.py` · `results/allcombo.json` (26 combo) · `results/allcombo_analysis.json`
- Bối cảnh: `document/2026-07-20_attention-redesign/` (attention trơ), `2026-07-21_stacked-fusion/ket-qua.md`
  (uniform_avg + ensemble = mô hình tốt nhất, 0.785 vs DyAM 0.775)
