# Rà soát cấu hình: vì sao bài báo ~0.79 còn tôi ~0.76?

> Ngày: 2026-07-21 · Mã: `experiments/allcombo/run_audit_singlerun.py`

## Kết luận: cấu hình GIỐNG HỆT, khác biệt thuần ở cách đánh giá CV

`model_params` và `dfs_rad_filters` trong notebook **trùng khít** cái tôi dùng:
- `model_params = {epochs:125, lr:0.01, alpha:0.001, beta:0.0, cross_modality_enabled:False}` = `B.MODEL_PARAMS`.
- `dfs_rad_filters = {pc,pl,ln : l1_strength 0.1}` = `ctx.rad_filters`.
- Cùng hàm `train()` (DyAM), cùng KFold(random_state=0, shuffle=True), cùng torch seed 42.

**Khác biệt duy nhất:** bài báo (cell 18) chạy `train()` **1 LẦN trên thứ tự bệnh nhân tự nhiên**; tôi báo cáo
**trung bình 5 seed có hoán vị nhãn** (mỗi seed một phân hoạch CV khác).

## Bằng chứng — tái lập đúng phương pháp bài báo

| combo | model | paper_ref | **1-run (như bài báo)** | 5-seed mean±sd | 5-seed span |
|---|---|---|---|---|---|
| BM1 Rad+IHC-G+Gen+PDL1 | DyAM | 0.7839 | **0.7976** | 0.7595±0.022 | [0.726, 0.781] |
| BM1 Rad+IHC-G+Gen+PDL1 | uniform | – | 0.7915 | 0.7746±0.019 | [0.745, 0.793] |
| BM3 Rad+Gen | DyAM | 0.7384 | **0.7549** | 0.7095±0.018 | [0.680, 0.722] |
| BM3 Rad+Gen | uniform | – | 0.7472 | 0.7110±0.015 | [0.688, 0.729] |
| BM4 PDL1+Gen | DyAM | 0.6931 | **0.6932** | 0.7072±0.009 | [0.692, 0.713] |
| BM4 PDL1+Gen | uniform | – | 0.7124 | 0.7191±0.008 | [0.710, 0.732] |

- **BM4 khớp CHÍNH XÁC** (ref 0.6931 vs 1-run 0.6932) → tôi tái lập đúng phương pháp bài báo, không sai cấu hình.
- BM1 (0.7976 vs 0.7839) và BM3 (0.7549 vs 0.7384): tôi ra **cao hơn** ref một chút (khác biệt seed/phiên bản nhỏ).

## Vì sao con số lệch nhau

**Chạy 1-lần là một điểm ước lượng phương sai cao, không phải cấu hình tốt hơn:**
- BM1: 1-run (0.7976) rơi vào phân hoạch **thuận lợi** — cao hơn cả max của 5-seed span (0.781). Bài báo "may" ở đây.
- BM4: 1-run (0.6932) là phân hoạch **bất lợi** — thấp hơn 5-seed mean (0.7072). Nên ref BM4 THẤP hơn số của tôi.
- Tức single-run lúc cao lúc thấp; 5-seed mean (± sd) là ước lượng **ổn định và trung thực hơn**.

**Quan trọng — không ảnh hưởng so sánh uniform vs DyAM:** lạm phát single-run tác động **cả hai model như nhau**.
Ở single-run: BM1 DyAM 0.7976 vs uniform 0.7915 (chênh +0.006); BM3 0.7549 vs 0.7472 (+0.008); BM4 DyAM 0.6932 vs
uniform 0.7124 (uniform +0.019). Chênh lệch model luôn nằm trong nhiễu CV (~0.02 sd) — kết luận uniform ≈/≥ DyAM
giữ nguyên bất kể dùng single-run hay 5-seed.

## Chốt

Không có sai khác cấu hình. Bài báo đạt 0.79 vì báo cáo **một lần chạy 10-fold trên thứ tự tự nhiên** — với BM1
lần chạy đó rơi vào ~0.79–0.80 (tôi tái lập 0.7976). Nếu muốn "khớp bảng bài báo", chỉ cần dùng single-run cho MỌI
tổ hợp; nhưng khi đó uniform_avg cũng nhảy lên tương ứng (BM1 0.7915), nên so sánh không đổi. Khuyến nghị giữ 5-seed
làm số chính (trung thực hơn), và có thể kèm cột single-run để đối chiếu trực tiếp với bài báo.
