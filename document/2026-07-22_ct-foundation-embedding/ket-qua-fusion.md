# Kiểm chứng: embedding trong FUSION đa nguồn (radiology + pathology đồng thời)

> Ngày: 2026-07-22 · Mã: `experiments/foundation_embed/fm_fusion_compare.py` · Kết quả: `results/fm_fusion.json`

## Câu hỏi

Embedding đơn lẻ yếu (P1-P3), nhưng khi **kết hợp** với Gen/PDL1/Labs có bổ sung tín hiệu bù trừ → fusion tốt hơn không?
Thay imaging thủ công bằng embedding (Phikon pathology + MedicalNet 3D radiology, PCA-64 cho công bằng), giữ nguyên
Gen/PDL1/Labs. Chạy fusion uniform_avg + DyAM, discovery-CV 5-seed.

## Kết quả

| cấu hình | uniform_avg | DyAM |
|---|---|---|
| **A — thủ công (BM1)** | **0.7746 ± 0.019** | 0.7595 ± 0.022 |
| B — embed cả 2 (rad+path) | 0.6630 ± 0.023 | 0.6394 ± 0.020 |
| C — embed pathology, rad thủ công | 0.7531 ± 0.014 | 0.7468 ± 0.015 |
| D — embed radiology, path thủ công | 0.7192 ± 0.014 | 0.7068 ± 0.016 |
| A+Labs — thủ công | 0.7665 ± 0.011 | 0.7639 ± 0.018 |
| B+Labs — embed cả 2 | 0.6816 ± 0.018 | 0.6478 ± 0.018 |

## Kết luận: KHÔNG — embedding kéo TỤT fusion, nhất quán

- **Embed cả 2 (B) = 0.663, kém hand-crafted (A 0.775) −0.112** — khoảng cách rất lớn, không phải nhiễu.
- Mức hại tỉ lệ với "độ yếu" của embedding: embed radiology (D −0.056) hại hơn embed pathology (C −0.022), vì
  CT-embed gần ngẫu nhiên còn Phikon có tín hiệu thật. Embed cả 2 = cộng dồn hại.
- DyAM không cứu (bám sát uniform, đều tệ).

**Giả thuyết "embedding bù trừ trong fusion" bị bác bỏ.** Trên cohort này, embedding off-the-shelf đưa thêm nhiễu vào
tổ hợp vốn đã tốt với feature thủ công → hại. Hoàn tất bức tranh: **foundation embedding không phải đòn bẩy — dù đơn
lẻ hay fusion.**

Caveat (giữ nguyên): CT dùng MedicalNet/BiomedCLIP (không phải fmcib thật); pathology dùng Phikon mean-pool (không phải
CONCH+ABMIL). Config "đúng chuẩn" chưa test được ở môi trường này.
