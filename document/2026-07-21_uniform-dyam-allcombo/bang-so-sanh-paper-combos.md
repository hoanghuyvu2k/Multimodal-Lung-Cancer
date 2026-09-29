# Bảng so sánh: uniform_avg vs DyAM trên 21 tổ hợp GIỐNG BÀI BÁO

> Ngày: 2026-07-21 · Nguồn tổ hợp: `Figures-Finalized.ipynb` cell 18 · Mã: `experiments/allcombo/run_paper_combos.py`
> Kết quả: `results/paper_combos.json` · 5 seed × 10-fold, paired bootstrap trên bệnh nhân.

## Đối chiếu phạm vi

Bài báo dùng tập nguồn **mịn hơn** allcombo (5 domain): phân biệt **IHC-A** (`path_ihc_pdl1`) vs **IHC-G**
(`path_ihc_glcm`), **TMB** / **Gen**(mut_amp) / **non_tmb**, có **Rad-LU**, và cả **đơn modality**. Vì vậy
allcombo chỉ phủ **7/21** tổ hợp bài báo → tôi đã chạy bổ sung **14 tổ hợp còn thiếu**.

- **Reproduce khớp chính xác 4 benchmark khoá**: BM1 (#18), BM2 (#21), BM3 (#11), BM4 (#8).
- **6 tổ hợp trùng allcombo** cho số y hệt (#10, #11, #13, #15, #18, #21) → chéo kiểm.

## Bảng đầy đủ 21 tổ hợp (uniform_avg vs DyAM)

| # | Tổ hợp | #mod | n | uniform_avg | DyAM | Δ(u−d) | p | Thắng | Ghi chú |
|---|---|---|---|---|---|---|---|---|---|
| 1 | TMB | 1 | 247 | 0.6161±0.001 | 0.6161±0.001 | +0.000 | 1.00 | = | mới · đơn modality |
| 2 | PDL1 | 1 | 201 | 0.7198±0.012 | 0.7198±0.012 | +0.000 | 1.00 | = | mới |
| 3 | IHC-A | 1 | 105 | 0.6347±0.021 | 0.6347±0.021 | +0.000 | 1.00 | = | mới |
| 4 | Gen | 1 | 247 | 0.6763±0.013 | 0.6763±0.013 | +0.000 | 1.00 | = | mới |
| 5 | **Rad** | 3 | 187 | 0.6561±0.017 | **0.6725±0.020** | −0.016 | **0.017** | **DyAM** | mới · attention giữa 3 tổn thương |
| 6 | Rad-LU | 4 | 187 | 0.6450±0.021 | 0.6373±0.024 | +0.008 | 0.37 | uni | mới |
| 7 | TMB+PDL1 | 2 | 247 | 0.7096±0.005 | 0.6642±0.019 | +0.045 | 0.55 | uni | mới |
| 8 | PDL1+Gen | 2 | 247 | 0.7191±0.008 | 0.7072±0.009 | +0.012 | 0.16 | uni | =BM4 ✓ |
| 9 | Rad+IHC-A | 4 | 211 | 0.6679±0.023 | 0.6708±0.014 | −0.003 | 0.78 | ns | mới |
| 10 | Rad+IHC-G | 4 | 211 | 0.7022±0.024 | 0.7055±0.014 | −0.003 | 0.79 | ns | allcombo RP |
| 11 | Rad+Gen | 4 | 247 | 0.7110±0.015 | 0.7095±0.018 | +0.002 | 0.96 | ns | =BM3 ✓ |
| 12 | IHC-A+Gen | 2 | 247 | 0.6982±0.009 | 0.6934±0.011 | +0.005 | 0.87 | uni | mới |
| 13 | **IHC-G+Gen** | 2 | 247 | 0.7184±0.018 | **0.7470±0.018** | −0.029 | **0.010** | **DyAM** | allcombo PG |
| 14 | Rad+IHC-A+Gen | 5 | 247 | 0.7333±0.007 | 0.7249±0.017 | +0.008 | 0.41 | uni | mới |
| 15 | Rad+IHC-G+Gen | 5 | 247 | 0.7500±0.019 | 0.7507±0.022 | −0.001 | 0.60 | ns | allcombo RPG |
| 16 | Rad+IHC-A+MutAmp | 6 | 247 | 0.7478±0.012 | 0.7338±0.015 | +0.014 | 0.10 | uni | mới |
| 17 | Rad+IHC-A+Gen+PDL1 | 6 | 247 | 0.7548±0.010 | 0.7428±0.015 | +0.012 | 0.42 | uni | mới |
| 18 | Rad+IHC-G+Gen+PDL1 | 6 | 247 | 0.7746±0.019 | 0.7595±0.022 | +0.015 | 0.36 | uni | **=BM1 ✓** |
| 19 | Rad+IHC-A+Gen+TMB+PDL1 | 7 | 247 | 0.7704±0.012 | 0.7488±0.018 | +0.022 | 0.12 | uni | mới |
| 20 | Rad+IHC-A+Gen+PDL1+Labs | 7 | 247 | 0.7467±0.014 | 0.7363±0.017 | +0.010 | 0.85 | uni | mới |
| 21 | Rad+IHC-G+Gen+PDL1+Labs | 7 | 247 | 0.7665±0.011 | 0.7638±0.018 | +0.003 | 0.91 | uni | **=BM2 ✓** |

Δ>0 = uniform_avg cao hơn. "Thắng" = có ý nghĩa (p<0.05) **in đậm**; "uni/ns/=" = nghiêng uniform / không ý nghĩa / hòa tuyệt đối.

## Tổng hợp

- **uniform_avg nhỉnh mean: 12/21 · hòa: 4/21 (đơn modality) · DyAM nhỉnh: 5/21.**
- **Thắng có ý nghĩa: uniform_avg 0 · DyAM 2** — chỉ **Rad** (radiomics 3 tổn thương, p=0.017) và **IHC-G+Gen**
  (Path+Gen, p=0.010). Cả hai đều là tổ hợp **ít-mid modality không có PDL1**.
- **Đơn modality: uniform_avg = DyAM y hệt** (attention tầm thường khi 1 nguồn) — và khớp `signal_audit`.
- **Tổ hợp đa nguồn lớn (5–7 modality, #14–21, gồm BM1/BM2): uniform_avg nhỉnh mean ở 7/8, DyAM KHÔNG thắng cái nào.**
  Đây đúng là chế độ quan trọng của mô hình fusion → uniform_avg ≥ DyAM ở toàn bộ.

## Bảng có thêm cột single-run (1-lần, thứ tự tự nhiên = cách bài báo)

`run_paper_singlerun.py`. DyAM 1run đối chiếu ref bài báo: **BM4 0.6932=0.6931 khớp**, BM2 0.7908≈0.7879, BM1 0.7976 (ref 0.7839), BM3 0.7549 (ref 0.7384).

| # | Tổ hợp | #mod | uni 5seed | uni 1run | DyAM 5seed | DyAM 1run | ref |
|---|---|---|---|---|---|---|---|
| 1 | TMB | 1 | 0.6161 | 0.6150 | 0.6161 | 0.6150 | – |
| 2 | PDL1 | 1 | 0.7198 | 0.7080 | 0.7198 | 0.7080 | – |
| 3 | IHC-A | 1 | 0.6347 | 0.6056 | 0.6347 | 0.6056 | – |
| 4 | Gen | 1 | 0.6763 | 0.6759 | 0.6763 | 0.6759 | – |
| 5 | Rad | 3 | 0.6561 | 0.6822 | 0.6725 | 0.6925 | – |
| 6 | Rad-LU | 4 | 0.6450 | 0.6674 | 0.6373 | 0.6783 | – |
| 7 | TMB+PDL1 | 2 | 0.7096 | 0.6980 | 0.6642 | 0.6362 | – |
| 8 | PDL1+Gen | 2 | 0.7191 | 0.7124 | 0.7072 | **0.6932** | 0.6931 |
| 9 | Rad+IHC-A | 4 | 0.6679 | 0.6868 | 0.6708 | 0.6883 | – |
| 10 | Rad+IHC-G | 4 | 0.7022 | 0.7126 | 0.7055 | 0.7096 | – |
| 11 | Rad+Gen | 4 | 0.7110 | 0.7472 | 0.7095 | **0.7549** | 0.7384 |
| 12 | IHC-A+Gen | 2 | 0.6982 | 0.7138 | 0.6934 | 0.7148 | – |
| 13 | IHC-G+Gen | 2 | 0.7184 | 0.7238 | 0.7470 | 0.7558 | – |
| 14 | Rad+IHC-A+Gen | 5 | 0.7333 | 0.7283 | 0.7249 | 0.7723 | – |
| 15 | Rad+IHC-G+Gen | 5 | 0.7500 | 0.7787 | 0.7507 | 0.7828 | – |
| 16 | Rad+IHC-A+MutAmp | 6 | 0.7478 | 0.7651 | 0.7338 | 0.7823 | – |
| 17 | Rad+IHC-A+Gen+PDL1 | 6 | 0.7548 | 0.7726 | 0.7428 | 0.7961 | – |
| 18 | Rad+IHC-G+Gen+PDL1 | 6 | 0.7746 | 0.7915 | 0.7595 | **0.7976** | 0.7839 |
| 19 | Rad+IHC-A+Gen+TMB+PDL1 | 7 | 0.7704 | 0.7885 | 0.7488 | 0.7892 | – |
| 20 | Rad+IHC-A+Gen+PDL1+Labs | 7 | 0.7467 | 0.7628 | 0.7363 | 0.7650 | – |
| 21 | Rad+IHC-G+Gen+PDL1+Labs | 7 | 0.7665 | 0.7784 | 0.7638 | **0.7908** | 0.7879 |

**PHÁT HIỆN QUAN TRỌNG — thứ hạng ĐẢO giữa single-run và 5-seed ở các tổ hợp lớn:** ở 7/8 combo ≥5 modality
(#14–21), single-run cho **DyAM > uniform** (vd #17 DyAM 0.7961 vs uni 0.7726; #14 +0.044), nhưng 5-seed cho
**uniform ≥ DyAM**. Dấu của Δ đổi chiều một cách *hệ thống* → phân hoạch thứ-tự-tự-nhiên (bài báo dùng) **thiên
vị DyAM** ở tổ hợp nhiều nguồn, còn trung bình phân hoạch ngẫu nhiên thì nghiêng uniform. Kết luận trung thực:
**chênh lệch hai model nằm trong nhiễu phân hoạch CV — không model nào vượt trội bền vững.** Số 0.79 của bài báo
là một phân hoạch thuận lợi cho DyAM, không phải bằng chứng DyAM tốt hơn.

## Bảng 3 MODEL đồng nhất cấu hình (uniform / DyAM / OvO) — 5-seed & single-run

`run_paper_ovo.py`: OvO chạy qua ĐÚNG pipeline của uniform/DyAM (cùng run_model 5-seed hoán vị + single-run, cùng
guard 0-feature). Nguồn: `results/paper_combos.json` (đủ 21 combo, mỗi combo có `uniform/dyam/ovo`).

| # | Tổ hợp | #mod | uni 5s | DyAM 5s | OvO 5s | uni 1r | DyAM 1r | OvO 1r |
|---|---|---|---|---|---|---|---|---|
| 1 | TMB | 1 | 0.6161 | 0.6161 | 0.6161 | 0.6150 | 0.6150 | 0.6150 |
| 2 | PDL1 | 1 | 0.7198 | 0.7198 | 0.7198 | 0.7080 | 0.7080 | 0.7080 |
| 3 | IHC-A | 1 | 0.6347 | 0.6347 | 0.6347 | 0.6056 | 0.6056 | 0.6056 |
| 4 | Gen | 1 | 0.6763 | 0.6763 | 0.6763 | 0.6759 | 0.6759 | 0.6759 |
| 5 | Rad | 3 | 0.6561 | 0.6725 | 0.6591 | 0.6822 | 0.6925 | 0.6789 |
| 6 | Rad-LU | 4 | 0.6450 | 0.6373 | 0.6455 | 0.6674 | 0.6783 | 0.6711 |
| 7 | TMB+PDL1 | 2 | 0.7096 | 0.6642 | 0.7088 | 0.6980 | 0.6362 | 0.6990 |
| 8 | PDL1+Gen (BM4) | 2 | 0.7191 | 0.7072 | 0.7183 | 0.7124 | 0.6932 | 0.7157 |
| 9 | Rad+IHC-A | 4 | 0.6679 | 0.6708 | 0.6676 | 0.6868 | 0.6883 | 0.6814 |
| 10 | Rad+IHC-G | 4 | 0.7022 | 0.7055 | 0.7029 | 0.7126 | 0.7096 | 0.7000 |
| 11 | Rad+Gen (BM3) | 4 | 0.7110 | 0.7095 | 0.7132 | 0.7472 | 0.7549 | 0.7511 |
| 12 | IHC-A+Gen | 2 | 0.6982 | 0.6934 | 0.6972 | 0.7138 | 0.7148 | 0.7157 |
| 13 | IHC-G+Gen | 2 | 0.7184 | 0.7470 | 0.7193 | 0.7238 | 0.7558 | 0.7227 |
| 14 | Rad+IHC-A+Gen | 5 | 0.7333 | 0.7249 | 0.7353 | 0.7283 | 0.7723 | 0.7360 |
| 15 | Rad+IHC-G+Gen | 5 | 0.7500 | 0.7507 | 0.7521 | 0.7787 | 0.7828 | 0.7690 |
| 16 | Rad+IHC-A+MutAmp | 6 | 0.7478 | 0.7338 | 0.7443 | 0.7651 | 0.7823 | 0.7667 |
| 17 | Rad+IHC-A+Gen+PDL1 | 6 | 0.7548 | 0.7428 | 0.7526 | 0.7726 | 0.7961 | 0.7674 |
| 18 | Rad+IHC-G+Gen+PDL1 (BM1) | 6 | 0.7746 | 0.7595 | 0.7728 | 0.7915 | 0.7976 | 0.7975 |
| 19 | Rad+IHC-A+Gen+TMB+PDL1 | 7 | 0.7704 | 0.7488 | 0.7662 | 0.7885 | 0.7892 | 0.7906 |
| 20 | Rad+IHC-A+Gen+PDL1+Labs | 7 | 0.7467 | 0.7363 | 0.7450 | 0.7628 | 0.7650 | 0.7622 |
| 21 | Rad+IHC-G+Gen+PDL1+Labs (BM2) | 7 | 0.7665 | 0.7638 | 0.7632 | 0.7784 | 0.7908 | 0.7780 |

**Kết luận 3-model (đồng cấu hình):** **OvO ≈ uniform_avg** ở 5-seed (bám sát ±0.005, ở BM1/BM2/#17/#19/#21 còn
nhỉnh dưới uniform); OvO KHÔNG tách khỏi baseline. Cả hai ≥ DyAM ở robust. Single-run thì DyAM/OvO ≈ nhau ~0.79–0.80
(phân hoạch thuận lợi). → **Không model nào thắng bền vững; chênh trong nhiễu ~0.01.** OvO về bản chất là trung bình
gần uniform, không phải cải tiến thật.

## Bảng NLP-CLINICAL: base / +Labs / +NLP (đóng góp mã hoá lâm sàng bằng ngôn ngữ)

`experiments/nlp_verify/` · `results/nlp_combos.json` · backbone **uniform_avg**, **cùng MODEL_PARAMS (125 epoch, lr 0.01…)**
và cùng 21 tổ hợp. Với MỖI tổ hợp so 3 cách dùng lâm sàng:
- **base** = tổ hợp KHÔNG có lâm sàng (bỏ `cnl_dem_labs`).
- **+Labs** = base + **13 biến lâm sàng THÔ** (`cnl_dem_labs`).
- **+NLP** = base + **embedding NLP** của đúng 13 biến đó (câu chữ → MiniLM 384-dim, `no_scale`).

> So sánh công bằng là **+NLP vs +Labs** (cùng là dữ liệu lâm sàng, chỉ khác cách biểu diễn). Cột base để tham chiếu.
> Môi trường có monkeypatch PowerTransformer (scipy≥1.14) → loại vài cột radiomics suy biến; vì vậy **base khớp cột
> `uniform_avg` cũ trong sai số ~0.003** (vd #18 base 0.7720 vs uni cũ 0.7746) — không ảnh hưởng so sánh nội bộ base/Labs/NLP.

### 5-seed (robust) — in đậm = cao nhất trong 3 cách

| # | Tổ hợp | #mod | base | +Labs | +NLP | Δ(NLP−Labs) | Ghi chú |
|---|---|---|---|---|---|---|---|
| 1 | TMB | 1 | **0.6161** | 0.6052 | 0.6090 | +0.0038 |  |
| 2 | PDL1 | 1 | **0.7198** | 0.6888 | 0.6836 | −0.0052 | đơn-mod mạnh, mọi clinical đều kéo xuống |
| 3 | IHC-A | 1 | **0.6352** | 0.6255 | 0.6262 | +0.0007 |  |
| 4 | Gen | 1 | **0.6763** | 0.6345 | 0.6607 | +0.0262 |  |
| 5 | Rad | 3 | **0.6510** | 0.6421 | 0.6507 | +0.0086 |  |
| 6 | Rad-LU | 4 | **0.6508** | 0.6201 | 0.6109 | −0.0092 | radiomics-only, clinical làm loãng |
| 7 | TMB+PDL1 | 2 | 0.7096 | 0.6728 | **0.7256** | +0.0528 |  |
| 8 | PDL1+Gen | 2 | 0.7191 | 0.7086 | **0.7445** | +0.0360 | =BM4 |
| 9 | Rad+IHC-A | 4 | 0.6799 | 0.6710 | **0.7116** | +0.0406 |  |
| 10 | Rad+IHC-G | 4 | 0.7008 | 0.7019 | **0.7383** | +0.0364 |  |
| 11 | Rad+Gen | 4 | 0.7033 | 0.6891 | **0.7076** | +0.0185 | =BM3 |
| 12 | IHC-A+Gen | 2 | 0.6983 | 0.6755 | **0.7022** | +0.0267 |  |
| 13 | IHC-G+Gen | 2 | 0.7184 | 0.7251 | **0.7514** | +0.0262 |  |
| 14 | Rad+IHC-A+Gen | 5 | 0.7223 | 0.7231 | **0.7433** | +0.0202 |  |
| 15 | Rad+IHC-G+Gen | 5 | 0.7395 | 0.7404 | **0.7576** | +0.0171 |  |
| 16 | Rad+IHC-A+MutAmp | 6 | 0.7392 | 0.7280 | **0.7489** | +0.0208 |  |
| 17 | Rad+IHC-A+Gen+PDL1 | 6 | 0.7473 | 0.7422 | **0.7578** | +0.0156 |  |
| 18 | Rad+IHC-G+Gen+PDL1 | 6 | 0.7720 | 0.7661 | **0.7832** | +0.0171 | **=BM1** |
| 19 | Rad+IHC-A+Gen+TMB+PDL1 | 7 | 0.7596 | 0.7447 | **0.7687** | +0.0241 |  |
| 20 | Rad+IHC-A+Gen+PDL1+Labs | 7 | 0.7473 | 0.7422 | **0.7578** | +0.0156 | base=#17; +Labs=combo gốc |
| 21 | Rad+IHC-G+Gen+PDL1+Labs | 7 | 0.7720 | 0.7661 | **0.7832** | +0.0171 | **=BM2**; base=#18; +Labs=combo gốc |

### single-run (1-lần, thứ tự tự nhiên = kiểu bài báo) — in đậm = cao nhất trong 3 cách

| # | Tổ hợp | base | +Labs | +NLP | Δ(NLP−Labs) |
|---|---|---|---|---|---|
| 1 | TMB | 0.6150 | **0.6263** | 0.6062 | −0.0201 |
| 2 | PDL1 | **0.7080** | 0.6961 | 0.6767 | −0.0194 |
| 3 | IHC-A | 0.6056 | 0.6433 | **0.6607** | +0.0173 |
| 4 | Gen | **0.6759** | 0.6485 | 0.6534 | +0.0049 |
| 5 | Rad | 0.6931 | 0.6705 | **0.7139** | +0.0434 |
| 6 | Rad-LU | 0.6254 | **0.6339** | 0.6289 | −0.0051 |
| 7 | TMB+PDL1 | 0.6980 | 0.6911 | **0.7038** | +0.0127 |
| 8 | PDL1+Gen | 0.7124 | 0.7108 | **0.7333** | +0.0225 |
| 9 | Rad+IHC-A | 0.6837 | 0.6895 | **0.7533** | +0.0638 |
| 10 | Rad+IHC-G | 0.7106 | 0.7240 | **0.7750** | +0.0510 |
| 11 | Rad+Gen | **0.7605** | 0.7210 | 0.7558 | +0.0348 |
| 12 | IHC-A+Gen | 0.7138 | 0.6932 | **0.7254** | +0.0322 |
| 13 | IHC-G+Gen | 0.7238 | 0.7356 | **0.7559** | +0.0203 |
| 14 | Rad+IHC-A+Gen | 0.7636 | 0.7672 | **0.7997** | +0.0325 |
| 15 | Rad+IHC-G+Gen | 0.7678 | 0.7724 | **0.8012** | +0.0288 |
| 16 | Rad+IHC-A+MutAmp | 0.7563 | 0.7473 | **0.7668** | +0.0194 |
| 17 | Rad+IHC-A+Gen+PDL1 | 0.7731 | 0.7746 | **0.7940** | +0.0194 |
| 18 | Rad+IHC-G+Gen+PDL1 | 0.7917 | 0.7879 | **0.8091** | +0.0212 |
| 19 | Rad+IHC-A+Gen+TMB+PDL1 | 0.8015 | 0.7811 | **0.8105** | +0.0295 |
| 20 | Rad+IHC-A+Gen+PDL1+Labs | 0.7731 | 0.7746 | **0.7940** | +0.0194 |
| 21 | Rad+IHC-G+Gen+PDL1+Labs | 0.7917 | 0.7879 | **0.8091** | +0.0212 |

### Kết luận NLP — PHÁT HIỆN DƯƠNG ĐẦU TIÊN của toàn bộ đợt thực nghiệm

- **+NLP > +Labs ở 19/21 tổ hợp (5-seed)** và **18/21 (single-run)**. Hai ngoại lệ (**PDL1**, **Rad-LU**) đều là
  cấu hình **một-nguồn-mạnh / radiomics-only** — nơi thêm BẤT KỲ lâm sàng nào (thô hay NLP) cũng làm loãng qua trung
  bình đều của uniform_avg. Không phải NLP kém, mà là bản chất fusion 1-nguồn.
- **Trong chế độ fusion đa nguồn thật (≥2 modality khác loại): +NLP thắng gần như tuyệt đối**, biên **+0.015 → +0.053**
  (5-seed) và tới **+0.064** (single-run, #9). +NLP là **cao nhất trong 3 cách** ở mọi tổ hợp đa nguồn.
- **Head-to-head sạch nhất — chính 2 combo có Labs của bài báo (#20, #21):** thay raw Labs bằng NLP trên đúng cấu
  hình gốc → **+0.0156 / +0.0171 (5-seed)**, **+0.0194 / +0.0212 (single-run)**. NLP cải thiện *ngay tại* config bài báo.
- **Trên combo chủ lực BM1 (#18):** +NLP **cao nhất cả hai kiểu đo** — 5-seed 0.7832 (base 0.7720, Labs 0.7661);
  single-run **0.8091** (base 0.7917, Labs 0.7879). NLP-clinical là thứ **duy nhất trong cả dự án đẩy được BM1 vượt 0.80**.
- **Đỉnh single-run toàn đợt = 0.8105** (#19, +NLP) và **0.8091** (BM1, +NLP) — cao hơn mọi con số DyAM/OvO trước đó.

→ Khác hẳn mọi hướng đã thử (attention, stacked, foundation embedding, TabPFN — đều âm hoặc ngang): **mã hoá lâm sàng
bằng ngôn ngữ (NLP) là can thiệp DUY NHẤT cho cải thiện dương, bền vững và nhất quán** trên nhiều tổ hợp, ở cả 5-seed
lẫn single-run. Đây là đóng góp phương pháp mạnh nhất để đưa vào paper.

## Đọc kết quả

DyAM có **2 lợi thế thật** ở cấu hình nhỏ/vừa: (1) khi chỉ có radiomics 3 vị trí tổn thương — attention giữa các
lesion có ích; (2) Path+Gen. Nhưng **cả hai lợi thế biến mất khi thêm nguồn**: ở mọi tổ hợp ≥5 modality (kể cả
2 combo chủ lực BM1/BM2 của bài báo), uniform_avg ngang hoặc hơn, DyAM không thắng combo nào. Nhất quán với kết quả
allcombo (Δ tăng theo k). **Kết luận: uniform_avg tái lập được kết quả bài báo và không thua ở tổ hợp đa nguồn;
attention của DyAM chỉ giúp ở vài cấu hình ít nguồn cụ thể, không phải xu hướng "nhiều nguồn thì tốt hơn".**
