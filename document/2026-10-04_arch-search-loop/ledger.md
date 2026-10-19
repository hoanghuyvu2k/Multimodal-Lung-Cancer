# Sổ cái vòng lặp tìm kiến trúc

> Giao thức: [protocol.md](protocol.md). Mỗi ứng viên MỘT dòng, kể cả thất bại. Không xoá dòng nào.

## Trạng thái

- **ĐỢT 1 ĐÃ KẾT THÚC (2026-10-04).** 10/10 ứng viên · 1 vào Bước B (C18) · 0 đạt đủ tiêu chí chấp nhận.
- Vòng lặp ĐÃ DỪNG, chờ người dùng quyết định có chạy đợt 2 không.
- Con số chốt mới: chồng đầy đủ (uniform + NLP + miss + C18) BM1 = 0.7957 ± 0.011 / ensemble 0.8039 (§2.27).
- Hướng còn mở cho đợt 2 (đều là THÊM DỮ LIỆU): `pdl1_tiss_site`, `n_lesions`, `sex`, `smoking_status`,
  `histo` chi tiết — các biến nền có trong omnibus mà mô hình chưa dùng.

**Phát hiện dữ liệu (2026-10-04):** file omnibus có các biến nền CHƯA dùng: `clinical_pdl1_score` có ở
246/247 bệnh nhân (kênh PD-L1 hiện dùng điểm chấm lại, chỉ 201/247 → 45 người thiếu chỉ dấu mạnh nhất
dù có sẵn điểm lâm sàng; AUC đơn biến 0.738 vs 0.744); `pdl1_tiss_site` (vị trí sinh thiết: adrenal 36%
không đáp ứng n=11, bone 93% n=15, chest wall 100% n=6); `n_lesions` (AUC 0.598, 189/247); `sex`,
`smoking_status`, `histo` chi tiết. Đây là hướng "thêm nguồn dữ liệu", không phải kiến trúc.

**Bài học từ C1–C3, C10 (quy tắc hợp nhất không tham số):** phép trung bình các điểm tanh KHÔNG thật sự
"đều" — độ lớn |r_i| của mỗi chuyên gia tự mã hoá độ tự tin của nó (đầu yếu cho điểm gần 0). Chuẩn hoá
z xoá thông tin đó → −0.035. Tức mô hình vốn đã có "attention ngầm theo độ lớn", giải thích vì sao mọi
attention tường minh (DyAM, OvO, tin cậy học được) không thêm được gì.

## Đã thử TRƯỚC vòng lặp (26 lần — tính vào đa so sánh)

| Nhóm | Ứng viên | Kết quả ở tổ hợp nhiều nguồn | Mục nhật ký |
|---|---|---|---|
| Hợp nhất | DyAM (gốc) | mốc | §2.5 |
| Hợp nhất | OvO | +0.0064/seed vs DyAM, +0.0022 ensemble (p=0.57); −0.0015 vs uniform | §2.24 |
| Hợp nhất | uniform_avg | mốc phải vượt | §2.24 |
| Hợp nhất | GMU, precision weighting (method B) | không hơn | §2.2 |
| Hợp nhất | stacked LR, lr_late | −0.01 … −0.02 | §2.4 |
| Hợp nhất | trọng số tin cậy học gradient | −0.0165 | §2.18 |
| Hợp nhất | trọng số tin cậy ước lượng CV | −0.0077 | §2.18 |
| Hợp nhất | tiên nghiệm sẵn-có dạng cộng | +0.0076 (5/6), BM1 −0.006 | §2.18 |
| Dạng bảng | TabPFN | −0.04 … −0.06 | §2.7 |
| Dạng bảng | cooperative learning | −0.04 … −0.05 | §2.13 |
| Dạng bảng | XGBoost | −0.06 … −0.10 | §2.15 |
| Đồ thị | tương đồng bệnh nhân (NW) | −0.04 … −0.14; cộng vào còn hại | §2.19 |
| Đồ thị | GraphSAGE ×3 biến thể (kiểu MAGNET) | −0.03 … −0.04 | §2.20 |
| Bộ mã hoá | chia sẻ trọng số radiomics | −0.11 (1 seed) | §2.17 |
| Bộ mã hoá | chế độ huấn luyện theo modality | −0.004 … −0.014 ở tổ hợp có radiomics | §2.17 |
| Huấn luyện | pairwise loss | +0.0045 rồi triệt tiêu | §2.16 |
| Huấn luyện | modality dropout | −0.003 | §2.16 |
| Huấn luyện | siêu tham số (epochs/lr/alpha) | −0.0035 ở BM1 (5 seed) | §2.16 |
| Huấn luyện | đa nhiệm Cox / ordinal | −0.018 … −0.072, đơn điệu | §2.21 |
| Huấn luyện | tăng cường nhiễu loạn radiomics | +0.0021, p=0.23 (20 seed ghép cặp) | §2.22 |
| Dữ liệu | ComBat radiomics | −0.023 nội bộ; ngoại kiểm 0.51 | §2.16 |
| Dữ liệu | cả hai bộ pathology cùng lúc | −0.007 | §2.22 |
| Đầu vào | mã hoá NLP lâm sàng | **+0.017** (19/21 tổ hợp) ✅ | §2.8 |
| Đầu vào | chỉ báo mẫu khuyết (kênh) | **+0.009** (14/21) ✅ | §2.14 |
| Phương sai | ensemble theo seed | **+0.013 … +0.016** (bão hoà K≈10) ✅ | §2.22 |

## Vòng lặp — đợt 1

| # | Ứng viên | Bước A: Δ sàng lọc | Bước B: Δ gộp, CI, p | Thắng/6 | Kết luận | Code |
|---|---|---|---|---|---|---|
| C1 | Hợp nhất phân cấp cân bằng theo miền (radiomics = 1 phiếu) | **+0.0020** (BM1 +0.0063, #19 −0.0012, R+G+Gen +0.0008) | — | — | Không qua sàng lọc (ngưỡng +0.005); gần trung tính | đã xoá |
| C2 | Chuẩn hoá z điểm chuyên gia trước khi trung bình | **−0.0346** | — | — | Không qua; xoá mất độ tự tin mã hoá trong độ lớn | đã xoá |
| C3 | Trung vị thay trung bình | **−0.0988** | — | — | Không qua | đã xoá |
| C10 | Tổng thay trung bình | **−0.0368** | — | — | Không qua | đã xoá |
| C15 | Bộ mã hoá radiomics không giám sát (PCA 10 chiều trên mọi tổn thương + bản nhiễu loạn) | **−0.0077** (BM1 −0.011, #19 −0.003, R+G+Gen −0.010) | — | — | Không qua; chọn đặc trưng CÓ giám sát theo vị trí vẫn hơn | đã xoá |
| C8 | Chuyên gia phi tuyến dạng dư (MLP 4 nút, dropout 0.5, khởi đầu = tuyến tính) | **−0.0092** | — | — | Không qua; phi tuyến overfit dù khởi đầu trùng baseline | đã xoá |
| C7 | Mixup trong không gian điểm hợp nhất (Beta 0.4) | **−0.0042** | — | — | Không qua | đã xoá |
| C16 | Gộp cohort ngoài chỉ-có-pathology (path_valid, 52 bệnh nhân) vào mọi fold train | **−0.0148** (BM1 −0.025, #19 +0.007, R+G+Gen −0.027) | — | — | Không qua; pathology GLCM của cohort ngoài lệch phân bố (IHC-A thì +0.007) | đã xoá |
| C4 | Chọn đặc trưng radiomics ổn định (elastic-net trên 8 bootstrap, giữ ≥5/8) | **dừng sớm vì vô vọng** sau 9/15 lượt: BM1 −0.0076 (5 seed), #19 −0.0196 (4 seed); cần +0.042 ở tổ hợp thứ ba mới qua | — | — | Không qua; chọn ổn định hơn làm MẤT tín hiệu | đã xoá |
| C18 | **Lấp PD-L1 thiếu (45 bệnh nhân) bằng điểm PD-L1 trong báo cáo lâm sàng** — THÊM DỮ LIỆU, không phải kiến trúc | **+0.0070** ✅ QUA (BM1 +0.011 5/5; #19 +0.010 5/5; R+G+Gen 0 vì không có PD-L1) | Gộp 6: **+0.0067**, CI [−0.0018, +0.0155], p=0.118 | 4/6 (2 tổ hợp không có PD-L1 → hoà theo cấu trúc) | **Xu hướng dương rõ, CHƯA đạt tiêu chí chấp nhận** (cần ≥+0.010, p<0.05, ≥5/6). Theo seed: thắng 20/20 (p≈2e-6). BM1 ensemble 0.7994 vs 0.7844: p boot 0.053 / DeLong 0.050 / hoán vị 0.042 | GIỮ (`archloop/cands.py`) |

### Phân tích HẬU NGHIỆM cho C18 (không thuộc tiêu chí định trước — ghi rõ để không nhầm)

Chỉ trên 4 tổ hợp có PD-L1 (nơi C18 có thể tác động):

| So với | Δ ensemble gộp | CI 95% | p bootstrap | Thắng | Theo seed |
|---|---|---|---|---|---|
| uniform_avg | +0.0100 | [−0.0030, +0.0232] | 0.129 | 4/4 | +0.0132, 20/20 |
| OvO | +0.0126 | [−0.0002, +0.0262] | 0.054 | 4/4 | +0.0157, 20/20 |
| **DyAM (mô hình gốc)** | **+0.0184** | **[+0.0021, +0.0348]** | **0.028** | 4/4 | +0.0268, 19/20 |

Lưu ý đa so sánh: đây là ứng viên thứ 36 được thử trên cùng 247 bệnh nhân (26 trước vòng lặp + 10 trong
vòng lặp); p=0.028 chưa hiệu chỉnh. So với DyAM còn gộp hai thay đổi (backbone uniform + lấp PD-L1).

## Vòng lặp — đợt 2 (kiến trúc theo tài liệu balanced multimodal learning)

Trạng thái: **ĐỢT 2 KẾT THÚC (10/10)**. Không ứng viên nào đạt tiêu chí chấp nhận. D6b = "xu hướng dương, chưa xác nhận" (giữ code). Vòng lặp đã dừng, chờ người dùng quyết định.

| # | Ứng viên | Bước A: Δ sàng lọc | Bước B: Δ gộp, CI, p | Thắng/6 | Kết luận | Code |
|---|---|---|---|---|---|---|
| D2 | OGM-GE điều biến gradient, α=0.3 (CVPR 2022) | **−0.0006** | — | — | Không qua; trung tính — mô hình không bị mất cân bằng modality | đã xoá |
| D1 | Giám sát phụ từng chuyên gia, λ=0.5 (arXiv 2304.00379) | **−0.0017** | — | — | Không qua; trung tính | đã xoá |
| D4 | Dropout 0.2 trên đặc trưng đầu vào | **−0.0062** | — | — | Không qua | đã xoá |
| D3 | Chuyên gia huấn luyện đơn-modality độc lập (MLA, CVPR 2024) | **−0.0127** (thua 15/15 lượt) | — | — | Không qua; **huấn luyện CHUNG hơn độc lập ở mọi seed** → các nguồn học bổ trợ nhau | đã xoá |
| D9 | Hợp nhất trước phi tuyến: tanh(TB_i W_i x_i) | **−0.0335** | — | — | Không qua; tanh riêng từng chuyên gia là cốt lõi (cùng bài học với C2) | đã xoá |
| D5 | Cập nhật luân phiên từng chuyên gia theo mất mát tập thể (kiểu ReconBoost, ICML 2024) | **+0.0006** | — | — | Không qua; trung tính — tương đương huấn luyện chung | đã xoá |
| D10 | Chuyên gia tương tác chéo PD-L1 × genomics (tích từng cặp đặc trưng, khả dụng khi có cả hai nguồn) | **+0.0030** (BM1 +0.0053 4/5; #19 +0.0038 4/5; R+G+Gen 0 vì không có PD-L1) | — | — | Không qua (ngưỡng +0.005); ứng viên KIẾN TRÚC sát ngưỡng nhất, nghiêng dương 8/10 lượt trên tổ hợp áp dụng được | đã xoá |
| D6 | Nhiệm vụ phụ dự đoán chéo: lớp ẩn 4 nút của chuyên gia pathology phải dự đoán PD-L1 lâm sàng, λ=0.5 (arXiv 2304.00379) | **−0.0043** (BM1 −0.0101 0/5; **#19 +0.0137 5/5**; R+G+Gen −0.0165 0/5). Đối chứng chỉ-lớp-ẩn: −0.0037 / −0.0005 / −0.0046 → phần dương ở #19 đến hoàn toàn từ nhiệm vụ phụ | — | — | Không qua theo giao thức. Tách đôi theo loại pathology: IHC-A (18 chiều) +0.014, IHC-G (150 chiều) âm | giữ lại cho D6b |
| D6b | **NGOẠI LỆ CÓ GHI NHẬN** — như D6 nhưng chỉ áp khi chuyên gia pathology ≤ 32 đặc trưng (tức IHC-A; với IHC-G trùng khít uniform). Luật đặt SAU khi nhìn sàng lọc D6 → hậu nghiệm; đưa thẳng vào Bước B vì bộ sàng lọc có 2/3 tổ hợp IHC-G | (bỏ qua) | **+0.0070** [−0.0009, +0.0152], p = 0.078. Riêng 4 tổ hợp IHC-A: +0.0105, p = 0.072. Theo seed: +0.0081, 19/20, Wilcoxon p < 1e-5. 15 seed MỚI trên 4 tổ hợp IHC-A: +0.0115, 14/15 | 4/6 (2 hoà) | **Xu hướng dương, CHƯA xác nhận** — trượt tiêu chí 1, 2, 3. Đối chứng (4 tổ hợp IHC-A × 20 seed): chỉ lớp ẩn −0.0035; **đích phụ bị XÁO +0.0084 (16/20)**; PD-L1 thật +0.0121 → ~70% mức tăng là điều chuẩn chung của một đầu hồi quy phụ, chỉ +0.0037 (18/20, p = 0.0003) là do PD-L1 thật. Không tác động BM1 (IHC-G) nên không nâng AUC cao nhất | GIỮ `archloop/cands.py` (`train_aux`), điểm ở `ovo_proof/d6b_auxlow__*.csv`, `tests_d6b.json` |
| D7 | Tự chưng cất: 5 thầy bootstrap trong fold, nhãn mềm out-of-bag, trò học 0.5·BCE(y) + 0.5·BCE(nhãn mềm) | **−0.0014** (BM1 −0.0039 1/5; #19 −0.0001 3/5; R+G+Gen −0.0001 3/5) | — | — | Không qua; trung tính — lợi ích của ensemble KHÔNG chưng cất được vào một mô hình | đã xoá |

### Tổng kết đợt 2

- 10 ứng viên: 9 không qua sàng lọc (đã xoá), 1 (D6b) vào Bước B nhưng trượt tiêu chí chấp nhận.
- Nhóm cân bằng modality (D1, D2, D5) trung tính: chẩn đoán cho thấy mô hình không bị mất cân bằng.
- Tách/giảm phối hợp giữa chuyên gia thì thua (D3 −0.013, D9 −0.034).
- Tín hiệu dương duy nhất: nhiệm vụ phụ trên chuyên gia pathology ít chiều (D6b +0.012 trên tổ hợp IHC-A, 19/20 seed), nhưng ~70% là
  điều chuẩn chung (đích xáo vẫn +0.008), không tác động BM1, luật IHC-A hậu nghiệm.
- Cộng dồn cả dự án: 0/41 thay đổi kiến trúc/huấn luyện đạt tiêu chí chấp nhận so với `uniform_avg`.

### Sau đợt 2 — D6b chồng lên bộ tốt nhất (người dùng chọn hướng 1)

3 tổ hợp IHC-A × 20 seed, so với `c18stack` (uniform + NLP + miss + C18) cùng seed: Δ một mô hình +0.0064 / +0.0063 / +0.0041
(18, 16, 15 trên 20 seed); gộp theo seed +0.0056, Wilcoxon p = 0.0003; Δ ensemble gộp +0.0040 [−0.0062, +0.0151], p = 0.48.
→ cộng dồn khoảng một nửa, chưa xác nhận ở mức bệnh nhân. AUC cao nhất dự án không đổi (BM1 0.7957 / ens 0.8039).
Script `experiments/archloop/run_stack_d6b.py`, chi tiết `result/training-results.md` §2.29.

### Đợt dữ liệu đầu vào (người dùng yêu cầu thử tiếp) — trên bộ tốt nhất, 2 tổ hợp × 5 seed vs `c18stack`

| # | Ứng viên | Δ sàng lọc | Kết luận | Code |
|---|---|---|---|---|
| E1 | Chuyên gia số từ 5 biến nền chưa dùng (sex, smoking_status, histo, n_lesions, pdl1_tiss_site) | **−0.0149** (0/10 lượt) | Không qua; chuyên gia yếu nhận trọn một phiếu nên kéo trung bình xuống | đã xoá |
| E2 | Cùng các biến đó nối vào câu mô tả NLP rồi nhúng lại | **+0.0001** | Không qua; trung tính | đã xoá |

Không còn biến nền nào trong omnibus chưa thử (trừ tham số máy CT / scanner / mã định danh — cố ý không dùng). Chi tiết §2.30.

### Đặc trưng mô hình nền từ ảnh gốc (2026-10-05, người dùng nhắc về `datasets/pathology` và `datasets/radiology`)

| # | Ứng viên | Δ sàng lọc vs bộ tốt nhất | Xác nhận 20 seed × 4 tổ hợp | Kết luận | Code |
|---|---|---|---|---|---|
| F1 | Thêm chuyên gia Phikon (tiêu bản gốc, PCA-16, 163 bệnh nhân) | **+0.0154** (10/10 lượt) | PCA trong fold: theo seed +0.0152 (20/20); Δ ens gộp **+0.0159** [−0.003, +0.036], p = 0.10; thắng 4/4. vs DyAM +0.0386, p = 0.052 | **GIỮ — con số chốt mới: BM1 0.8095 / ens 0.8182.** Đạt Δ và số tổ hợp thắng, chưa đạt p < 0.05 | `archloop/run_fm.py` |
| F2 | Ghép embedding CT fmcib (PCA-16) vào chuyên gia radiomics PC | **−0.0056** | — | Không qua; CT mô hình nền không có tín hiệu | kết quả thô đã xoá; nhánh `f2` còn trong `run_fm.py` |

Chi tiết §2.32. Bước mở tiếp: tách độ phủ vs chất lượng đặc trưng; chạy 21 tổ hợp; mô hình nền mới hơn (UNI/CONCH) + gộp attention.

**ĐÍNH CHÍNH F1 (2026-10-05, §2.33):** hai đối chứng 20 seed × 4 tổ hợp cho thấy mức tăng của F1 không đến từ nội dung ảnh.
Phikon chỉ ở bệnh nhân đã có pathology thủ công: **−0.0086** (0/20). Chuyên gia chỉ có bias trên 163 bệnh nhân có tiêu bản (không
ảnh): **+0.0111** (20/20) — bằng ~73% của Phikon (+0.0152); ở BM1 0.8087 vs 0.8095. Phikon − bias: +0.0040 (p seed 0.021; p bệnh nhân
0.66). → F1 thực chất là "thêm biến có-tiêu-bản vào tín hiệu sẵn-có". Giữ số liệu, KHÔNG gọi là thắng của mô hình nền.

### Trích lại tiêu bản bằng mô hình nền mới hơn (2026-10-05) — KẾ HOẠCH ĐỊNH TRƯỚC khi có số

Mô hình có cổng đăng ký (UNI, CONCH, Virchow2, H-optimus-0, Hibou, GigaPath) KHÔNG tải được vì máy chưa có token Hugging Face.
Dùng hai mô hình mở: `owkin/phikon-v2` (ViT-L, 1024 chiều) và `kaiko-ai/midnight` (ViT-g, CLS + trung bình patch = 3072 chiều).
Trích mức Ô: ≤ 500 ô 256 px @20x trong vùng u HALO, 163 slide discovery, GPU RTX 3060 qua env `doan-env`
(`experiments/foundation_embed/tile_fm_extract.py` → `cache/tiles/{model}/{slide}.npy`).

**Mốc phải vượt KHÔNG phải bộ tốt nhất cũ mà là đối chứng `f1avail`** (chuyên gia chỉ có bias trên 163 bệnh nhân có tiêu bản;
BM1 0.8087, TB 4 tổ hợp +0.0111 so với c18stack) — vì mọi chuyên gia pathology mới đều tự động mang tín hiệu "có tiêu bản".

| # | Ứng viên | Cách đánh giá |
|---|---|---|
| G1 | Phikon-v2, trung bình ô → PCA-16 trong fold → chuyên gia thêm | 4 tổ hợp × 20 seed, Δ vs `f1avail` cùng seed |
| G2 | Midnight, trung bình ô → PCA-16 trong fold → chuyên gia thêm | như trên |
| G3 | Gộp bằng attention MIL (chỉ làm nếu G1 hoặc G2 hơn `f1avail` ≥ +0.005) | như trên |

Số chiều PCA (16) và số ô (500) cố định, không dò. Tiêu chí: Δ ens gộp vs `f1avail` ≥ +0.010 và p bootstrap < 0.05.

**KẾT QUẢ G1/G2 (2026-10-05, §2.34):** so với đối chứng `f1avail` — G1 Phikon-v2 **+0.0026** theo seed (14/20), Δ ens gộp +0.0035, p 0.65;
G2 Midnight **+0.0012** (13/20), +0.0038, p 0.69. Không cái nào đạt +0.005 → **G3 không chạy**. Chẩn đoán: embedding tiêu bản tương
quan 0.77–0.82 với PD-L1 TPS (tiêu bản là nhuộm PD-L1) → trùng thông tin với kênh PD-L1. Đặc trưng mức ô được giữ ở
`experiments/foundation_embed/cache/tiles/`. Hướng mô hình nền trên tiêu bản: đóng với các mô hình mở; UNI/CONCH cần token.


### Đợt tự thiết kế (2026-10-06) — người dùng yêu cầu tự sáng tạo kiến trúc, không bám bài báo

Mốc: **bộ tốt nhất `c18stack`** (uniform + NLP + miss_pattern + C18; KHÔNG Phikon/“có tiêu bản”), 3 tổ hợp (BM1, Rad+IHC-A+Gen+TMB+PDL1,
Rad+IHC-A+MutAmp) × 5 seed chuẩn, so cùng seed. Qua sàng lọc nếu Δ ≥ +0.005; xác nhận 4 tổ hợp × 20 seed + 5 phép kiểm.
Ba ý tưởng, mỗi ý bắt nguồn từ MỘT phát hiện của chính dự án này; tham số định trước, không dò:

| # | Thiết kế | Cơ sở trong dữ liệu của dự án |
|---|---|---|
| N1 | **Bagged evidence network**: trong MỖI fold huấn luyện B=8 thành viên trên 8 mẫu bootstrap của tập train, trung bình đầu ra; thay cho ensemble theo seed (+0.013…0.016) bằng phương sai giảm NGAY TRONG mô hình. Khác D7 (chưng cất, thất bại): dùng trực tiếp trung bình thành viên | sai số chủ yếu là phương sai do mẫu huấn luyện (ensemble bão hoà K≈10) |
| N2 | **Nhãn mềm theo lợi ích bền vững**: 29/48 bệnh nhân SD có DCB (PFS ≥ 6 tháng, định nghĩa của chính cohort: cột `DCR`) đang bị gán nhãn "không đáp ứng". Mục tiêu huấn luyện của họ = 0.5 (N2a) hoặc 0.0 (N2b); ĐÁNH GIÁ vẫn trên nhãn nhị phân gốc, nên so được với mọi số trước | nhiễu nhãn thật: `bor`=3 ⇒ SD (đã đối chiếu với bảng MIND công khai), PFS trung vị 7.0 tháng so với 1.8 ở POD |
| N3 | **Khuếch đại đồng thuận**: s = mean(r_i)·(1 + γ·c), c = |mean r|/mean|r| ∈ [0,1] (các chuyên gia cùng dấu → c = 1); γ học được, khởi đầu 0 nên chế độ tắt = baseline | độ lớn |r_i| đã là độ tự tin ngầm; chưa từng dùng mức ĐỒNG THUẬN giữa các chuyên gia |

**Quy tắc định trước bổ sung (2026-10-06, sau rà soát, TRƯỚC khi có số):**
1. Chế độ tắt của mọi ứng viên N phải tái lập `c18stack` seed 42 BM1 (giá trị lấy trực tiếp từ `c18stack__Rad_IHCG_Gen_PDL1.csv`); N3 "tắt" = γ bị đóng băng ở 0.
2. N2: `pos_weight` và `response_zscore` dùng nhãn CỨNG; nhãn mềm chỉ vào hàm mất mát. DCB lấy từ omnibus (`bor`==3 & `pfs`≥6 tháng = 29/48). Kỳ vọng N2b ÂM theo cấu trúc
   (29 bệnh nhân DCB vẫn là "không đáp ứng" ở nhãn kiểm tra). Báo thêm điểm cuối phụ KHÔNG dùng để chọn: AUC với DCB tính là đáp ứng.
3. N1 được đánh giá bằng Δ của ENSEMBLE cùng seed so với ensemble của `c18stack` (không phải Δ một mô hình, vì bagging giảm phương sai gần như chắc chắn nâng AUC một mô hình).
   Nếu N1 chỉ ngang ensemble sẵn có thì kết luận "rẻ hơn, không tốt hơn".
4. N3: nếu qua sàng lọc, chạy đối chứng `k_only` (c được thay bằng trung bình c của đúng nhóm số chuyên gia k, thống kê lấy ở lượt huấn luyện) — nếu mức tăng còn nguyên thì
   tín hiệu là SỐ NGUỒN CÓ SẴN chứ không phải đồng thuận. Số bệnh nhân đã thử trên cùng 247 người: khoảng 49 ứng viên kể cả đợt này.

**KẾT QUẢ đợt tự thiết kế (2026-10-06, §2.35) — 0/4 qua sàng lọc, code và điểm thô đã xoá:**

| # | Δ TB 1 mô hình | Δ TB ensemble | Kết luận |
|---|---|---|---|
| N1 bagged evidence network | −0.0010 | **−0.0037** | Không hơn ensemble theo seed; bagging trong fold kém đa dạng hơn ensemble qua nhiều phân hoạch |
| N2a nhãn mềm DCB 0.5 | −0.0181 | −0.0163 | Thua 15/15 lượt |
| N2b nhãn mềm DCB 0.0 | −0.0534 (1 tổ hợp) | −0.0476 | Dừng sớm; âm theo cấu trúc như đã dự báo |
| N3 khuếch đại đồng thuận | +0.0010 | +0.0018 | Không qua; đối chứng k_only không cần chạy |


### Kiểm chứng NGOÀI bằng hai cohort ngoài (2026-10-06) — KẾ HOẠCH ĐỊNH TRƯỚC khi chạy

Phát hiện: `rad_valid` (n=50) và `path_valid` (n=71) có bảng lâm sàng (age, pack_years, ecog, albumin, dnlr, hist_adeno đủ ~100%), nhãn `bor`, và
`path_valid` có điểm PD-L1 lâm sàng 71/71. Thiếu 6/13 biến của câu NLP gốc → dùng câu RÚT GỌN 6 biến, huấn luyện lại trên discovery bằng đúng 6 biến đó.
Huấn luyện TRÊN TOÀN BỘ discovery, kiểm tra MỘT LẦN trên cohort ngoài, không chỉnh gì sau khi nhìn kết quả ngoài. Siêu tham số chỉ chọn bằng CV trong discovery.
- **A — mã hoá lâm sàng:** raw6 vs NLP6 (MiniLM 384d), cả hai qua LR (C ∈ {0.01, 0.1, 1, 10} chọn bằng CV 10-fold discovery). Tập kiểm tra = gộp path_valid ∪ rad_valid theo
  hàng omnibus (loại trùng). Báo AUC + CI bootstrap, và Δ(NLP6 − raw6) bootstrap ghép cặp 5000 lần.
- **B — ba nguồn trên path_valid (n=71):** IHC-G (GLCM) + PD-L1 lâm sàng (−TPS/100, cùng cách C18) + NLP6. Mô hình `uniform_avg`, 5 hạt khởi tạo trung bình. Cánh: PD-L1 một mình ·
  GLCM một mình · NLP6 một mình · GLCM+PD-L1 · GLCM+PD-L1+NLP6 · DyAM cùng ba nguồn. So sánh chính: ba nguồn vs PD-L1 một mình; ba nguồn vs hai nguồn (tác dụng của NLP6); uniform vs DyAM.
- Không dùng `miss_pattern` (cohort ngoài có mẫu sẵn-có khác hẳn). Không có ngưỡng "thắng/thua"; kết quả mô tả, vì n = 71 (≈ 21 bệnh nhân đáp ứng) chỉ thấy được chênh lệch lớn.

**KẾT QUẢ kiểm chứng ngoài (2026-10-06, §2.36):** A (n = 119): raw6 0.673 [0.564, 0.776], NLP6 0.554 [0.433, 0.672], NLP6 − raw6 = −0.120, p = 0.021 — NHƯNG 6 biến này ≈ ngẫu nhiên ở discovery
(CV 0.496), nên không kiểm được tuyên bố NLP13. B (path_valid, n = 71): PD-L1 0.739 · GLCM 0.755 · ba nguồn 0.721 · DyAM 0.708; ba nguồn − PD-L1 −0.018, p = 0.71. PD-L1 và pathology khái quát;
lợi ích hợp nhất và NLP không kiểm được ở cỡ mẫu này.


### Các hướng còn lại để cải thiện bài (2026-10-06) — KẾ HOẠCH ĐỊNH TRƯỚC khi chạy

Hướng "đổi kiến trúc" đã cạn (≈ 49 ứng viên, 0 qua). Còn lại hai loại: (i) hướng MÔ HÌNH chưa thử, (ii) phân tích làm bài báo chắc hơn. Chạy theo thứ tự, mỗi hướng một dòng kết quả, kể cả âm.

| # | Hướng | Loại | Đánh giá |
|---|---|---|---|
| R1 | Khoảng tin cậy cấp bệnh nhân + hiệu chuẩn (độ dốc, Brier) cho con số chốt | báo cáo | trên điểm đã lưu `c18stack` (20 seed), bootstrap 5000, hiệu chuẩn chéo 10-fold |
| R2 | Điểm cuối phụ: PFS (C-index, HR) và điểm cuối DCB (SD có lợi ích bền vững tính là đáp ứng) | báo cáo | điểm đã lưu; DyAM / uniform / mô hình chốt; pfs_censor = 1 là SỰ KIỆN |
| R3 | Bỏ từng nguồn một (leave-one-source-out) ở BM1 | phân tích | 20 seed, so với `c18stack` cùng seed; nhóm: radiomics (3 vị trí), pathology, genomics, PD-L1, NLP, miss_pattern |
| R4 | Bộ mã hoá câu: all-mpnet-base-v2, bge-small-en-v1.5, và trung bình nhiều mẫu câu (MiniLM) | MÔ HÌNH | sàng lọc 3 tổ hợp × 5 seed, ngưỡng Δ ≥ +0.005 so với `c18stack`; hai giá trị tham số định trước, không dò |
| R5 | Kiểm chứng ngoài genomics trên MSK-IMPACT (Samstein, 307 NSCLC không trùng) bằng sống còn | báo cáo | đặc trưng rút gọn (TMB + đột biến 6 gen; không có CNA); C-index OS |

Không làm (nêu lý do cho người dùng): thêm bệnh nhân / cohort đa phương thức độc lập; mô hình nền có cổng (UNI, CONCH…) vì thiếu token; mở rộng thêm kiến trúc.

**Quy tắc bổ sung sau rà soát (2026-10-06, trước khi có số):**
- R3: `miss_pattern` đóng băng từ danh sách nguồn ĐẦY ĐỦ; cánh `*_avail` bỏ thêm bit sẵn-có. Khi bỏ radiomics dùng `l1 = {}`; `no_scale` tính lại sau mỗi lần bỏ. So sánh cặp chỉ trên bệnh nhân
  có điểm ở CẢ hai lần chạy (hai seed của `c18stack` có ~25 bệnh nhân NaN vì một fold bị bỏ qua khi bộ lọc radiomics trả 0 đặc trưng; mọi mô hình có cùng tập NaN).
- R2: C-index tính bằng `concordance_index(T, -điểm, E)` với E = `pfs_censor` (1 = sự kiện); HR = Cox liên tục theo 1 độ lệch chuẩn của điểm ensemble, kèm log-rank theo trung vị. DCB dùng
  định nghĩa của cohort (`bor`==3 & `pfs`≥6); SD bị kiểm duyệt trước 6 tháng bị LOẠI khỏi điểm cuối DCB. PFS KHÔNG phải điểm cuối độc lập (nhãn sinh từ BOR, PFS dự đoán nhãn với AUC 0.93).
- R1: chỉ báo cáo khoảng tin cậy cấp bệnh nhân của ensemble 20 seed và độ lệch giữa các seed; BỎ hiệu chuẩn vì điểm là z-score theo fold, bài không trình bày xác suất.
- R4: mỗi bộ mã hoá ghi parquet RIÊNG (`nlp_clinical_embed_{tên}.parquet`, không đụng `nlp_clinical_embed.parquet`); mọi embedding được L2-chuẩn hoá tường minh. Nếu có biến thể qua sàng lọc, chạy đủ 21 tổ hợp trước khi gọi là kết quả.
  Biến thể (định trước): e1 `all-mpnet-base-v2` (câu gốc P1) · e2 `BAAI/bge-small-en-v1.5` (câu gốc P1) · e3 MiniLM, trung bình của 4 mẫu câu P1–P4 đã chuẩn hoá rồi chuẩn hoá lại.
  P1 = `df_to_text_prompts` (câu gốc của dự án); P2 = `df_to_text_prompts_v2`; P3 = "Advanced NSCLC, {hist}, primary site {site}. Age {age}, ECOG {ecog}, albumin {albumin:.2f} g/dL, dNLR {dnlr:.2f},
  tumor burden {tb:.2f}, {pack_years} pack-years. {brain} brain and {liver} liver metastases. Immunotherapy line {line}, {combo}."; P4 = "A {age}-year-old with {hist} of the lung, performance status {ecog},
  smoking {pack_years} pack-years. Laboratory values: albumin {albumin:.2f}, dNLR {dnlr:.2f}. Tumour burden {tb:.2f}. Metastasis: {brain} brain, {liver} liver. Treatment: line {line}, {combo}; {prior PD-L1 status}."
  (brain/liver = with|without; combo = combination therapy|monotherapy; site = lung|extra-pulmonary; hist = adenocarcinoma|non-adenocarcinoma).
- R5: chỉ dùng TMB + đột biến "bất kỳ không đồng nghĩa" ở 6 gen (không có CNA, nên mất 4 đặc trưng khuếch đại); đây KHÔNG đồng nhất với `driver_binarized`, đặc biệt EGFR. Loại cả 50 ID trùng. Dùng cả 307 bệnh nhân; báo thêm 288 đơn trị.
- Toàn cục: chỉ R4 có thể đổi con số tiêu đề; thêm ~3 ứng viên vào ~49 đã thử trên cùng 247 bệnh nhân.

**KẾT QUẢ R1, R2, R5 (2026-10-06, §2.37):**
- R1: BM1 ensemble 0.804 [0.740, 0.861] vs DyAM 0.780 [0.711, 0.843] — khoảng tin cậy chồng nhau gần hoàn toàn.
- R2: lợi thế AUC đáp ứng KHÔNG chuyển sang PFS (C-index ≈ 0.62 cho mọi mô hình) và gần như không sang DCB (chênh TB +0.011 so với DyAM, BM1 −0.001).
- R5: genomics 7 đặc trưng trên Samstein NSCLC (n = 307) C-index OS 0.539 [0.494, 0.583] ≈ ngẫu nhiên; không xác nhận được ngoài.

**KẾT QUẢ R4 (2026-10-06, §2.38) — 0/3 qua sàng lọc, code và điểm thô đã xoá:** e1 mpnet Δ TB ensemble −0.0008 · e2 bge-small +0.0001 · e3 MiniLM trung bình 4 mẫu câu +0.0002 (ngưỡng +0.005).
Kết luận có ích: lợi ích của NLP không phụ thuộc bộ mã hoá hay mẫu câu.

**KẾT QUẢ R3 (2026-10-06, §2.39):** bỏ từng nguồn ở BM1 (20 seed, ghép cặp): genomics −0.0285 (20/20) · pathology + bit có-mặt −0.0330 (nội dung riêng −0.0114 → ~2/3 là tín hiệu có-tiêu-bản) ·
PD-L1 −0.0144 · radiomics −0.0129 · NLP **−0.0034** · miss_pattern **−0.0010** (p = 0.47). Trong mô hình chốt NLP và miss chỉ còn đóng góp biên nhỏ; thang cộng dần §2.31 phụ thuộc thứ tự.
**Tổng kết đợt (R1–R5):** không có hướng nào nâng con số tiêu đề; thu được 4 phân tích làm bài chắc hơn (R1 CI bệnh nhân, R2 lợi thế không chuyển sang PFS/DCB, R3 đóng góp theo nguồn, R4 độ bền theo bộ mã hoá)
và 1 giới hạn (R5 genomics không xác nhận ngoài). Khoảng 52 ứng viên đã thử trên cùng 247 bệnh nhân.


### Vòng lặp thực hiện các hướng từ tài liệu (2026-10-06) — KẾ HOẠCH ĐỊNH TRƯỚC khi chạy

| # | Hướng | Cách làm (cố định trước) |
|---|---|---|
| H1 | Mức thổi phồng do chọn mô hình ("winner's curse") | Từ sai số chuẩn SE của hiệu cặp (bootstrap bệnh nhân) ở các cặp mô hình đã lưu, mô phỏng kỳ vọng của MAX trên K = 52 ứng viên KHÔNG có hiệu ứng thật với tương quan giữa ứng viên ρ ∈ {0, 0.5, 0.8}: E[max] = √(1−ρ)·σ·E[max của K chuẩn độc lập]; so với mức tăng quan sát. Không huấn luyện |
| H2 | Độ bền của tín hiệu sẵn-có | BM1, mô hình chốt CÓ `miss_pattern` (A) và KHÔNG có (B); ở tập kiểm tra, mỗi bệnh nhân có pathology bị đặt "không có" với xác suất p ∈ {0.25, 0.5} (A: cập nhật cả bit has_path và n_available); 20 seed; so ΔAUC(p) so với p = 0 (đã có: `c18stack`, `loso_miss`) |
| H3 | TRIPOD+AI / PROBAST+AI | Lấy danh mục mục từ bài gốc, đối chiếu với kho mã và nhật ký; ghi Đạt / Một phần / Chưa; KHÔNG bịa mục |
| H4 | MSK-CHORD | Chỉ thử nếu API công khai cho tách được bệnh nhân NSCLC dùng miễn dịch kèm sống còn; thời gian giới hạn; nếu không thì ghi rõ lý do |
| H5 | Mốc tabular kiểu Chowell | Random forest trên 13 biến lâm sàng + 11 đặc trưng genomics + PD-L1 lâm sàng, discovery 10-fold × 5 seed; so với PD-L1 một mình và mô hình chốt |
| H6 | Cohort đa phương thức độc lập | Không làm được trong phiên này (cần hợp tác) |

**KẾT QUẢ vòng lặp H1–H6 (2026-10-06, §2.40):**
- H1: E[max Δ | không hiệu ứng] ≈ 0.011–0.024 (ρ = 0.8 → 0.0 ; SE = 0.0105, K = 52) so với mức quan sát +0.0195 → mức tăng so với trọng số đều nằm trong dải thổi phồng do chọn.
- H2: khi 50% bệnh nhân bị ẩn pathology, mô hình có miss_pattern mất −0.0225, không có miss_pattern mất −0.0165; hiệu −0.0060 (p = 0.019). Ở p = 0 hai mô hình ngang nhau (+0.0010). → khuyên bỏ miss_pattern khỏi mô hình chính.
- H3: `document/2026-10-06_reporting-checklists/tripod-ai-probast-self-assessment.md`.
- H4: không làm (không có nhãn đáp ứng ICI ở mức thuộc tính; xem §2.40). H5: RF 0.727 / LR 0.743 trên lâm sàng + genomics + PD-L1 (< mô hình chốt 0.796). H6: không làm được.


### Kiểm tra bỏ `miss_pattern` trên các tổ hợp k ≥ 5 (2026-10-06) — KẾ HOẠCH ĐỊNH TRƯỚC khi chạy

Lý do: bằng chứng bỏ vector mẫu khuyết (§2.39, §2.40) chỉ ở BM1, trong khi sweep 21 tổ hợp cũ (5 seed, trước C18) cho thấy vector này giúp +0.02…0.03 ở tổ hợp KHÔNG có PD-L1. Sáu tổ hợp k ≥ 5 phân biệt:
Rad+IHC-A+Gen · Rad+IHC-G+Gen · Rad+IHC-A+MutAmp · Rad+IHC-A+Gen+PDL1 · BM1 · Rad+IHC-A+Gen+TMB+PDL1 (BM1 đã có: `c18stack` vs `loso_miss`).
So ghép cặp mô hình chốt CÓ vs KHÔNG `miss_pattern` (uniform + NLP + C18), 20 seed, AUC mỗi seed. **Quy tắc quyết định (định trước):** giữ vector (và báo cáo cả hai phiên bản) nếu Δ(có − không) trung bình qua sáu tổ hợp ≥ +0.005,
hoặc có tổ hợp nào Δ ≥ +0.010 với Wilcoxon p < 0.05; ngược lại bỏ khỏi mô hình chính. Hai tổ hợp không có PD-L1 cần chạy cả nhánh CÓ (chưa có điểm từng bệnh nhân).

**KẾT QUẢ kiểm tra miss_pattern trên sáu tổ hợp k ≥ 5 (2026-10-06, §2.41):** Δ(có − không) = +0.0166 (Rad+IHC-A+Gen, 20/20) · +0.0180 (Rad+IHC-G+Gen, 19/20) · +0.0007 · +0.0003 · +0.0010 (BM1) · +0.0005; TB +0.0062.
Theo quy tắc định trước → **GIỮ**. **Đính chính khuyến nghị trước ("nên bỏ khỏi mô hình chính", §2.40):** đúng cho các tổ hợp có PD-L1, sai cho hai tổ hợp k = 5 không có PD-L1 (mất 0.017–0.018).


### Hướng THÊM THÔNG TIN genomics / PD-L1 để nâng ngưỡng AUC (2026-10-06) — KẾ HOẠCH ĐỊNH TRƯỚC khi có số

Căn cứ: genomics là nguồn có giá trị lớn nhất trong mô hình chốt (bỏ đi mất −0.029, §2.39) nhưng chuyên gia chỉ dùng 11 đặc trưng (TMB + 6 driver + 4 khuếch đại). Bộ MIND công khai trên cBioPortal (`lung_msk_mind_2020`, CÙNG 247 bệnh nhân, tải qua API
công khai) còn đột biến toàn gen, số bản sao (FGA), điểm MSI mà mô hình chưa dùng. Tài liệu: KEAP1/STK11/SMARCA4/KRAS/TP53 liên quan kết cục ICI; gánh nặng bất thường số bản sao (FGA) cao liên quan lợi ích ICI giảm. Đây là họ "thêm thông tin", họ duy nhất từng thắng.
Mốc: bộ chốt `c18stack` (có miss_pattern), 3 tổ hợp (BM1, #19, MutAmp) × 5 seed chuẩn; ngưỡng Δ ≥ +0.005; qua thì xác nhận 4 tổ hợp × 20 seed.

| # | Ứng viên | Định nghĩa (cố định) |
|---|---|---|
| GX1 | thêm vào chuyên gia genomics: FGA + MSI_SCORE | 2 đặc trưng liên tục, điền trung vị nếu thiếu |
| GX2 | thêm KRAS, TP53, KEAP1, SMARCA4 | 4 nhị phân "có đột biến không đồng nghĩa" (danh sách chọn theo tài liệu, KHÔNG theo dữ liệu) |
| GX3 | GX1 + GX2 | 6 đặc trưng thêm |
| PL1 | chuyên gia PD-L1 thêm ngưỡng lâm sàng | tps ≥ 1% và tps ≥ 50% (2 chỉ báo) cạnh điểm liên tục; điền C18 làm trước (fill=False) |
Chế độ tắt phải tái lập c18stack seed 42 BM1 (0.8117). Đếm đa so sánh: thêm 4 ứng viên vào ~52 đã thử. KHÔNG nhìn AUC đơn biến để chọn đặc trưng.

**KẾT QUẢ GX1–GX3, PL1 (2026-10-06, §2.42) — 0/4 qua, code và điểm thô đã xoá:** GX1 −0.0123 · GX2 −0.0057 · GX3 −0.0219 (càng thêm càng hại) · PL1 +0.0049 (sát ngưỡng) → xác nhận thăm dò 20 seed × 4 tổ hợp: +0.0011 mỗi seed (p = 0.37),
ensemble −0.0005 (p = 0.89). Sàng lọc 5 seed phóng đại ~0.004. Không hướng nào nâng con số; khoảng 56 ứng viên đã thử trên cùng 247 bệnh nhân.


### Kết hợp với bộ dữ liệu công khai khác (2026-10-06) — KẾ HOẠCH ĐỊNH TRƯỚC khi chạy

Nguồn (cBioPortal, API công khai): `nsclc_pd1_msk_2018` (Rizvi 2018, 240 bệnh nhân, DCB 69 có / 158 không / 13 không đánh giá được → loại 13, PD-L1 `PDL1_SCORE`, TMB, đột biến) và `nsclc_mskcc_2018` (Hellmann 2018, 75 bệnh nhân, DCB 37/38, PD-L1 `PDL1_EXP`, TMB, đột biến).
Loại MỌI bệnh nhân trùng với 368 bệnh nhân của bạn (Rizvi 23; Hellmann 0). Điểm cuối thống nhất = DCB (nhãn: 1 = KHÔNG lợi ích bền vững, cùng chiều với nhãn gốc). Cohort MIND: DCB = PR/CR, hoặc SD có PFS ≥ 6 tháng (cùng định nghĩa DCR của cohort).
Đặc trưng thống nhất (8): PD-L1 (TPS/100; thiếu → chuyên gia PD-L1 bị che), TMB, đột biến "bất kỳ không đồng nghĩa" ở EGFR, ERBB2, BRAF, MET, STK11, ARID1A. KHÔNG có khuếch đại (chỉ Rizvi có CNA). Lưu ý WES (Hellmann) vs panel (MIND, Rizvi) cho TMB; khác phác đồ.
- **V1 — kiểm chứng ngoài:** `uniform_avg` hai chuyên gia (PD-L1; genomics 7 đặc trưng), huấn luyện trên toàn bộ MIND (DCB), kiểm tra MỘT lần trên Rizvi (đánh giá được, không trùng) và Hellmann; 5 hạt khởi tạo trung bình. Cánh: PD-L1 · TMB một mình · genomics · PD-L1 + genomics. AUC + CI bootstrap, Δ ghép cặp.
- **V2 — chuyên gia tiên nghiệm:** huấn luyện cùng mô hình PD-L1 + genomics CHỈ trên dữ liệu ngoài (Rizvi + Hellmann, không trùng), tính điểm cho 247 bệnh nhân của bạn, thêm làm một modality cố định (1 cột) vào bộ chốt c18stack. Sàng lọc 3 tổ hợp (BM1, #19, MutAmp) × 5 seed, ngưỡng Δ ≥ +0.005 trên NHÃN GỐC (PR/CR vs SD/PD); điểm cuối DCB báo thêm, không dùng để chọn.
  Qua sàng lọc → xác nhận 4 tổ hợp × 20 seed. Chế độ tắt = c18stack. Đếm đa so sánh: +1 ứng viên.

**KẾT QUẢ kết hợp dữ liệu công khai (2026-10-07, §2.43):**
- V1 (n = 133 có PD-L1, huấn luyện MIND → Rizvi + Hellmann, DCB): PD-L1 0.615 · genomics 0.629 · TMB 0.649 · **PD-L1 + genomics 0.689 [0.594, 0.784]**; hợp nhất − genomics +0.060 (p = 0.043), − PD-L1 +0.075 (p = 0.115). Đã sửa lỗi đánh giá lần đầu (cánh PD-L1 tính trên bệnh nhân thiếu PD-L1).
- V2 (điểm tiên nghiệm huấn luyện CHỈ trên 283 bệnh nhân ngoài, thêm vào bộ chốt): sàng lọc +0.0084 → xác nhận 20 seed × 4 tổ hợp: Δ ensemble gộp +0.0068, p = 0.34; MutAmp (không có TMB) +0.0180 (p = 0.034), BM1 −0.0003.
  Xu hướng dương, chưa xác nhận; mức tăng tập trung ở tổ hợp không có TMB. Giữ `experiments/external_prior/`.


### Dùng tiếp dữ liệu ngoài công khai — hai hướng (2026-10-07) — KẾ HOẠCH ĐỊNH TRƯỚC khi có số

**Hướng 1 — chuyên gia tiên nghiệm GIÀU đặc trưng, huấn luyện chỉ trên dữ liệu ngoài** (283 bệnh nhân: Rizvi 208 + Hellmann 75, đã loại 23 trùng và 9 DCB không đánh giá được; nhãn DCB, 1 = không lợi ích).
Ba bộ đặc trưng: S0 = PD-L1 + TMB + 6 đột biến (đã có, §2.43) · S1 = S0 + KRAS, TP53, KEAP1, SMARCA4 (cùng chuyên gia genomics) · S2 = S1 + chuyên gia lâm sàng (hút thuốc từng/chưa, giới, tuổi).
Chọn bộ đặc trưng CHỈ bằng kiểm chứng chéo trong dữ liệu ngoài (10-fold × 5 seed, AUC DCB; hòa trong 0.003 thì chọn bộ ít đặc trưng hơn) — KHÔNG nhìn AUC trên 247 bệnh nhân MIND để chọn.
Bộ được chọn → điểm cho 247 bệnh nhân → thêm làm 1 modality cố định vào bộ chốt c18stack; sàng lọc 3 tổ hợp (BM1, #19, MutAmp) × 5 seed, ngưỡng Δ ≥ +0.005 trên nhãn gốc; qua thì xác nhận 4 tổ hợp × 20 seed.

**Hướng 2 — gộp Hellmann (nhãn BOR, định nghĩa trùng nhãn gốc: CR/PR vs SD/PD; 7 NE loại → 68 bệnh nhân) vào tập HUẤN LUYỆN của mô hình đầy đủ** như bệnh nhân chỉ có PD-L1 + genomics (mask các nguồn ảnh và NLP = 0), KHÔNG vào tập đánh giá.
Bản đối chiếu: mô hình chốt KHÔNG có `miss_pattern` (vì mẫu sẵn-có của bệnh nhân ngoài k = 2 sẽ làm hỏng tín hiệu đã học: ở MIND k nhỏ là gần như toàn không đáp ứng) = `loso_miss` (BM1), `nm_nomiss` (#19, MutAmp), 5 seed đầu.
Đặc trưng genomics ngoài ánh xạ theo TÊN CỘT: TMB → TMB; "<GEN> driver_binarized" → đột biến bất kỳ không đồng nghĩa ở gen đó (xấp xỉ: so với cột DRIVER ở MIND — EGFR 22 vs 22, ERBB2 19 vs 19, MET 10 vs 12, STK11 51 vs 44, ARID1A 21 vs 16, BRAF 14 vs 7); "<GEN>: AMP_binarized" → thiếu (NaN).
PD-L1 ngoài = −PDL1_EXP/100 (cùng dấu C18). Ngưỡng Δ ≥ +0.005 trên 3 tổ hợp × 5 seed; qua thì xác nhận 20 seed. Đếm đa so sánh: +2 ứng viên vào ~57.

**KẾT QUẢ dùng tiếp dữ liệu ngoài (2026-10-07, §2.44) — 0/2 qua, code và điểm thô đã xoá:**
- H1 (tiên nghiệm giàu đặc trưng): CV ngoài S0 0.6548 > S1 0.6334 > S2 0.6064 → chọn S0 = điểm của §2.43; thêm đặc trưng hại cả ở cỡ mẫu 283.
- H2 (gộp Hellmann BOR vào huấn luyện, bản không miss): Δ TB một mô hình +0.0044 (BM1 +0.0014, #19 +0.0085 5/5, MutAmp +0.0033) < ngưỡng +0.005 → không qua.
- Sự cố: lần chạy đầu của H2 lỗi mà báo "đang chạy" khi chưa kiểm tra nhật ký; đã sửa và chạy lại theo dõi được.


### Kiểm chứng theo thứ tự (proxy thời gian) và đường cong học theo cỡ mẫu (2026-10-07) — KẾ HOẠCH ĐỊNH TRƯỚC khi chạy

**Phát hiện dữ liệu trước khi chạy:** thuộc tính `DRUG_START` của MIND công khai KHÔNG phải ngày lịch: giá trị 15 925–30 582 khớp với TUỔI tính bằng ngày tại thời điểm bắt đầu thuốc (bệnh nhân 83 tuổi → 30 582). Dữ liệu hiện có KHÔNG có thời gian lịch.
Proxy duy nhất: số thứ tự chụp CT (`radiology_accession_number`, 247 giá trị khác nhau, 190 323–348 069): liên hệ yếu với thế hệ máy chụp (Revolution cao hơn, p = 0.0014; Discovery thấp hơn, p = 0.0044 — chiều không thống nhất), Spearman với mã bệnh nhân 0.44; tỉ lệ không đáp ứng theo ngũ phân vị số chụp
0.62 · 0.67 · 0.84 · 0.84 · 0.78 (AUC số chụp → nhãn 0.613). **Không xác minh được số chụp là thời gian**; kết quả chỉ được diễn giải là "kiểm chứng giữ-ngoài theo thứ tự số chụp (proxy, có lệch tỉ lệ nhãn)", không phải kiểm chứng theo thời gian.
- **T1 — forward chaining theo số chụp (BM1 và Rad+IHC-A+Gen+TMB+PDL1):** sắp xếp 247 bệnh nhân theo số chụp; huấn luyện 50% đầu → kiểm tra 17% kế; 67% đầu → 17% kế; 83% đầu → 17% cuối; gộp dự đoán ngoài-thứ-tự cho 50% bệnh nhân sau (~123), AUC gộp + CI bootstrap, kèm AUC từng khối.
  Bốn cánh: DyAM · uniform · mô hình chốt CÓ miss_pattern · mô hình chốt KHÔNG miss_pattern; 3 hạt khởi tạo trung bình. Chọn đặc trưng radiomics CHỈ trên bệnh nhân huấn luyện (loại cả bệnh nhân tương lai).
- **T2 — đường cong học:** BM1, mô hình chốt không miss và DyAM; trong mỗi fold của 10-fold × 5 seed lấy ngẫu nhiên (phân tầng theo nhãn) 40/60/80/100% bệnh nhân huấn luyện, chọn đặc trưng radiomics chỉ trên phần lấy ra; AUC trên cùng fold kiểm tra. Báo AUC theo cỡ mẫu và độ dốc.
Không có ngưỡng thắng/thua (phân tích mô tả).

**KẾT QUẢ T1, T2 (2026-10-07, §2.45):** T1 (giữ-ngoài theo số chụp, 124 bệnh nhân cuối, 21 đáp ứng): BM1 DyAM 0.686 · uniform 0.681 · chốt-có-miss 0.722 · chốt-không-miss 0.704; #19 0.665 · 0.678 · 0.716 · 0.707; CI ±0.12, chồng nhau. AUC thấp hơn CV ≈ 0.07–0.09.
T2 (đường cong học BM1): chốt 0.7265 → 0.7567 → 0.7742 → 0.7931 ở 89/133/178/222 bệnh nhân; DyAM 0.668 → 0.707 → 0.732 → 0.757; lợi thế thu hẹp 0.058 → 0.036; chưa phẳng.
Giới hạn: không có thời gian lịch trong dữ liệu (DRUG_START là tuổi theo ngày); số chụp chỉ là proxy chưa xác minh.


### Cải thiện OvO THUẦN KIẾN TRÚC (2026-10-10) — theo yêu cầu: KHÔNG NLP, KHÔNG sửa dữ liệu đầu vào (kế hoạch định trước)

Mục tiêu người dùng: làm OvO tốt hơn DyAM bằng thay đổi kiến trúc OvO; dữ liệu giữ nguyên (không NLP, không C18, không miss_pattern, PD-L1 chấm lại gốc). Mốc: OvO gốc và DyAM trong `ovo_proof/{ovo,dyam}__*.csv`, 3 tổ hợp (BM1, Rad+IHC-A+Gen+TMB+PDL1, Rad+IHC-A+MutAmp) × 5 seed chuẩn.
Chẩn đoán đã có (§2.19 ghi chú): cổng OvO phẳng (sigmoid 0.58–0.60, sd 0.034–0.037) vì điểm attention bị CHIA cho số đặc trưng của nguồn (logit thô 3e-5…1.5e-3) nên OvO ≈ trọng số đều.
Cách cài: lớp con của `AttentionMatrixOvO` ghi đè `forward` (vá `lung_helpers.AttentionMatrixOvO` quanh `train_ovo_safe`); chế độ tắt phải tái lập OvO gốc (seed 42, BM1).
| # | Biến thể (định trước) |
|---|---|
| OV-A | không chia logit attention cho số đặc trưng (hệ số 1) |
| OV-B | chia cho căn bậc hai số đặc trưng |
| OV-C | cổng theo ĐỘ TIN CẬY: attention_i = sigmoid(γ·(|tanh r_i| − trung bình |tanh r_j| các nguồn khác)), γ học được, khởi đầu 1; không dùng lớp attention theo đặc trưng |
| OV-D | OV-A + beta = 0.1 (phạt chuẩn L2 của attention, kéo về đều) |
Qua sàng lọc nếu Δ(biến thể − OvO gốc) một mô hình ≥ +0.005 trung bình 3 tổ hợp; qua thì xác nhận 6 tổ hợp k ≥ 5 × 20 seed, năm phép kiểm so với OvO gốc VÀ DyAM. "Thành công" (theo mục tiêu người dùng) = vượt DyAM ở mức bệnh nhân gộp p < 0.05 và không kém OvO gốc.
Đếm đa so sánh: +4 ứng viên.

**KẾT QUẢ OvO thuần kiến trúc (2026-10-10, §2.48) — 0/4 qua, code và điểm thô đã xoá:** OV-A −0.0024 · OV-B −0.0006 · OV-C −0.0016 · OV-D −0.0037 so với OvO gốc (ngưỡng +0.005). OvO gốc vẫn hơn DyAM +0.0129 ở 3 tổ hợp (5 seed); không biến thể nào mở rộng khoảng cách.


### So sánh DyAM + NLP với OvO + NLP (2026-10-10) — KẾ HOẠCH ĐỊNH TRƯỚC khi chạy

Yêu cầu người dùng: so sánh hai mô hình CÙNG thêm NLP (không miss_pattern, không C18, PD-L1 gốc; chỉ thêm modality `cnl_nlp` thay `cnl_dem_labs`). DyAM+NLP chưa từng chạy ở 21 tổ hợp.
- **Pha 1 (21 tổ hợp, 5 seed + một lần chạy tự nhiên, như §2.9/§2.23):** `run_variant` của `nlp_verify` (có `clean_rad_filters`), cho DyAM {base, +NLP}; OvO {base, +NLP} đã có ở `ovo_nlp_combos.json` cùng đường chạy. Báo AUC cao nhất mỗi mô hình và số tổ hợp thắng.
- **Pha 2 (6 tổ hợp k ≥ 5, 20 seed, lưu điểm từng bệnh nhân):** DyAM+NLP và OvO+NLP cùng phân hoạch; báo AUC trung bình, ensemble, số seed thắng, khoảng tin cậy bệnh nhân dạng ước lượng (theo quyết định 2026-10-10: không lấy p-value làm kết luận).
Không có ngưỡng thắng/thua (so sánh mô tả).

**KẾT QUẢ DyAM+NLP so với OvO+NLP (2026-10-10, §2.49):** 21 tổ hợp 5 seed: cao nhất DyAM+NLP 0.7732 vs OvO+NLP 0.7822 (một lần chạy 0.8038 vs 0.8133); 6 tổ hợp k ≥ 5 × 20 seed: trung bình 0.7488 vs 0.7564 (Δ +0.0077; OvO thắng 16/20 seed, 4/6 tổ hợp), cao nhất BM1 0.7706 vs 0.7800.


### MC-OvO: bù chuyên gia thiếu kiểu MCMoE rút gọn trên OvO (2026-10-10) — KẾ HOẠCH ĐỊNH TRƯỚC khi chạy

Theo yêu cầu "có" sau khi nêu MCMoE (arXiv 2511.17397); OvO THUẦN KIẾN TRÚC (không NLP, không sửa dữ liệu). Ứng viên MC-OvO: với mỗi chuyên gia i có bộ sinh g_i = tanh(Linear([r_j·mask_j (j≠i), mask_j (j≠i)])); chuyên gia THIẾU được thay r_i bằng g_i và tham gia attention với điểm attention nhân 0.5; huấn luyện tái dựng MSE(g_i, r_i.detach()) trên chuyên gia quan sát được (trọng số 0.5, đi qua tham số beta). Siêu tham số CỐ ĐỊNH (0.5/0.5), không dò.
Bước A: 3 tổ hợp (BM1, #19, Rad+IHC-G+Gen) × 5 seed so với OvO gốc cùng seed (ovo__*.csv); ngưỡng qua = Δ TB ≥ +0.005 (như OV-A..D). Qua thì Bước B 20 seed × 6 tổ hợp; không qua thì xoá code + điểm thô. Báo hướng và độ lớn, không p-value (quyết định 2026-10-10).

**MC-OvO lần chạy 1 (2026-10-10) — LỖI CÀI ĐẶT, không tính là kết quả:** Δ −0.123 (0/5 seed ở cả 3 tổ hợp). Chẩn đoán BM1 seed 42: tắt tái dựng 0.7818, bật 0.6924 → gradient của mất mát tái dựng chảy ngược vào r_j qua đầu vào bộ sinh và phá chuyên gia. Sửa: `rm.detach()` ở đầu vào bộ sinh (thiết kế định trước đã nói đích detach; đầu vào detach là sửa lỗi, siêu tham số không đổi). Xoá điểm thô lần 1, chạy lại Bước A.

**KẾT QUẢ MC-OvO lần 2 (2026-10-10, §2.50):** Δ TB −0.0065 (BM1 −0.0065, 7 nguồn −0.0068, R+G+Gen −0.0062; 2/5 seed mỗi tổ hợp) → không qua ngưỡng +0.005; code + điểm thô đã xoá. Cộng dồn ≈ 60 ứng viên, 0 qua.

### OvO cổng Laplace kiểu FuseMoE (2026-10-12) — KẾ HOẠCH ĐỊNH TRƯỚC khi chạy

Theo yêu cầu "có" (FuseMoE arXiv 2402.03226 chưa thử). OvO THUẦN KIẾN TRÚC. Biến thể OvO-Lap: điểm attention của chuyên gia i = −‖x_i − c_i‖₂ / √n_i với tâm c_i học được (khởi tạo 0) thay cho Linear(x_i)/n_i; phần còn lại giữ nguyên (sigmoid(s_i − mean_others), softplus, chuẩn hoá L1). Lý do: điểm gốc chia cho n_i nên cổng phẳng (sigmoid 0.58–0.60); khoảng cách/√n có biên độ O(1). Siêu tham số không có, không dò.
Bước A: BM1, bảy nguồn, Rad+IHC-G+Gen × 5 seed so với OvO gốc cùng seed (ovo__*.csv), ngưỡng Δ TB ≥ +0.005. Không qua → xoá code + điểm thô. Báo hướng và độ lớn, không p-value.

**KẾT QUẢ OvO-Laplace (2026-10-12, §2.51):** Δ TB +0.0024 (BM1 −0.0010, 7 nguồn +0.0095 4/5, R+G+Gen −0.0014) → không qua ngưỡng +0.005; code + điểm thô đã xoá. Cộng dồn ≈ 61 ứng viên, 0 qua.

### Hoàn thiện phần còn thiếu trên nền OvO+NLP (2026-10-12) — KẾ HOẠCH ĐỊNH TRƯỚC khi chạy

Theo yêu cầu "chạy loop thực hiện nốt các phần còn thiếu" sau khi bài chuyển sang OvO (Q1 B). Tất cả dùng đường chạy `run_variant`/`build_l1` (như `nlp20_*`), mô hình `train_ovo_safe` + `cnl_nlp` (no_scale), chuẩn §0, 20 seed, đối chiếu `nlp20_ovo__*.csv` cùng seed. Mô tả, không ngưỡng thắng/thua.
- **W1 Bỏ từng nguồn (LOSO) ở BM1:** bỏ radiomics (3 vị trí), pathology, genomics, PD-L1, NLP. → `ovo_proof/loso_ovo_{arm}.csv`.
- **W2 Miss_pattern (Q2 A):** OvO+NLP+miss_pattern so với OvO+NLP trên 6 tổ hợp k ≥ 5. → `ovo_proof/nlp20miss_ovo__*.csv`.
- **W3 Giữ-ngoài theo số chụp (T1) và đường cong học (T2):** OvO+NLP và DyAM+NLP ở BM1 (kế hoạch chi tiết khi cài đặt, dựa `run_order_curve.py`).
- **W4 SK-6:** môi trường riêng numpy ≥ 2 + scikit-survival, tính tdAUC (6/12/18 tháng) và IBS cho một phân hoạch tự nhiên (seed 42) của 7+4 biến thể nhánh IHC-A/IHC-G; chỉ cập nhật Hình 6B/6D nếu tính được.
Sau mỗi mục: ghi `training-results.md`, cập nhật bài (qua `make_results_tables.py` / văn bản), dựng lại PDF và Word.

**KẾT QUẢ W1 bỏ từng nguồn OvO+NLP (2026-10-12, §2.53):** đầy đủ 0.7800; bỏ pathology −0.0325, radiomics −0.0212, PD-L1 −0.0163, NLP −0.0154, genomics −0.0145 (18–20/20 seed giảm).

**KẾT QUẢ W2 miss_pattern OvO+NLP (2026-10-12, §2.54):** +0.0175 và +0.0165 ở hai tổ hợp không PD-L1 (20/20, 19/20), +0.0047 đến +0.0086 ở bốn tổ hợp còn lại (BM1 +0.0033, 12/20). Giữ ở phụ lục.

**KẾT QUẢ W3 (2026-10-12, §2.55; môi trường GPU, so nội bộ):** T1 BM1 DyAM+NLP 0.708 vs OvO+NLP 0.743 (7 nguồn 0.684 vs 0.715), CI chồng nhau; T2 khoảng cách 0.030 → 0.002 khi cỡ mẫu 89 → 222.

### W4 SK-6 rút gọn: tdAUC trên 5 seed (2026-10-12) — KẾ HOẠCH ĐỊNH TRƯỚC khi chạy
Mã tính tdAUC/IBS của bản gốc không còn trong kho, nên không thể tái tạo định nghĩa IBS cũ; chỉ làm phần xác định rõ: tdAUC tích luỹ/động (Uno) tại 6, 12, 18 tháng từ điểm nguy cơ ngoài-fold của đúng 11 biến thể × 5 seed của `verify_nlp_5seed` (DyAM, cùng harness). Bước 1 (env chính): `experiments/survival/run_survival_scores.py` lưu điểm vào `results/survival_scores/*.csv`. Bước 2 (venv numpy ≥ 2 + scikit-survival mới): `compute_tdauc.py` đọc điểm + PFS, ghi `results/tdauc_5seed.json`. IBS: không tính; Hình 6D giữ nhãn minh hoạ. Báo mean ± sd, hướng và độ lớn.

**KẾT QUẢ W4 (2026-10-12, §2.56):** tdAUC 0.64–0.70 cho mọi biến thể, chênh giữa các mã hoá trong một nhánh ≲ 0.02; tuyên bố cũ về lợi thế NLP ở 12–18 tháng không tái lập. IBS không tính (không còn mã gốc).

### Tái tạo notebook + OvO/NLP (2026-10-12, §2.57)
Chạy `run_notebook_repro.py` (cấu hình notebook, 21 tổ hợp × DyAM/OvO/+NLP × single-run + 5 seed). DyAM single-run khớp bài ≤0.003 ở BM1–BM4; ~0.8 chỉ ở single-run (OvO 0.8015, DyAM+NLP 0.8123); 5 seed BM1: DyAM 0.7599, OvO 0.7704, DyAM+NLP 0.7789, OvO+NLP 0.7847. Không sửa bài.

### Dao động single-run (2026-10-12, §2.58)
Hàm gốc train/train_ovo, BM1–BM3: seed khởi tạo vô tác dụng (manual_seed(42) cứng trong __init__, cả ở main); dao động hoàn toàn do phân hoạch fold (sd 0.012–0.019, khoảng 0.04–0.08). Phân hoạch của notebook thuận lợi (cao hơn 89–100% phân hoạch khác). OvO−DyAM trên các phân hoạch chung: BM1 +0.009 (16/18), BM2 +0.010 (17/18), BM3 −0.003 (7/18). Hàm gốc crash '0 feature' ở 2/20 phân hoạch. Không sửa bài.

### Môi trường gốc (2026-10-12, §2.59)
Dựng `doan-orig` đúng requirements.txt + Python 3.9.13 (main). Code main trong môi trường này trùng 6 chữ số với output .ipynb (Rad+IHC-A+Gen+PDL1 0.764), không phải HTML (0.798). 0.798/0.80 của bài không tái hiện được; không do phiên bản thư viện.

### Chạy từng ô notebook (2026-10-12, §2.60)
Hàng không radiomics khớp HTML (≤0.0004); hàng có radiomics lệch (LR Rad-LN +0.054; DyAM Rad+IHC-A+Gen+PDL1 0.764 vs 0.798). Lệch cả ở LR → nghi dữ liệu/lọc radiomics, không do DyAM. Ghi ở sai-khac-can-dieu-tra.md.

### Đính chính nguồn (2026-10-12)
Output .ipynb gốc (main) == HTML (0.798). Bản .ipynb ở thư mục làm việc là bản chạy lại (0.764). Code + dữ liệu main KHỚP MD5 Synapse; chạy lại trong môi trường ghim vẫn không ra 0.798 ở hàng radiomics. Loại trừ: seed saga/tol/RNG toàn cục, 19 biến thể chọn đặc trưng. Còn nghi: thứ tự bệnh nhân/pandas, môi trường thật của tác giả.

### Điều tra lệch radiomics (2026-10-12, §2.61)
Loại: nhiễu số học, 19 biến thể chọn đặc trưng, pandas khác phiên bản (code gốc chỉ chạy được pandas ≤1.4.x). Thứ tự bệnh nhân (=> phân hoạch KFold) làm AUC LR Rad-LN dao động sd 0.041; giá trị của tác giả nằm trong phân bố. Cơ chế chưa rõ.

### Vòng điều tra môi trường (2026-10-12/13, §2.62)
Môi trường tốt nhất: sklearn 0.24.2 (không phải 1.1.1), numpy 1.20.3, scipy 1.8.1, pandas 1.4.2, torch 1.11.0, py3.9.13 (cố định trong doan-orig). Khớp 6 chữ số LR Rad-PC/LN, DyAM Rad, Rad+IHC-A, IHC-A, Gen. Còn lệch các hàng gộp Genomics (BM1-IHC-A 0.7879 vs 0.7982). Loại: scipy/numpy/torch/pandas/threads/params. Chưa khớp hoàn toàn.

### Tái hiện trên Linux (2026-10-13, §2.63)
Linux (WSL2) + py3.8.15, torch 1.8.1, sklearn 0.24.2, numpy 1.20.3: DyAM Rad+IHC-A+Gen+PDL1 0.798256 vs 0.798169 (một cặp bệnh nhân); Windows 0.7879. Nền tảng là biến số còn thiếu.

### 21 tổ hợp × 4 biến thể trong môi trường tham chiếu (2026-10-13, §2.64)
env ref-linux-py38; 1 seed + 5 seed. 5 seed: DyAM 0.7067, OvO 0.7111, NLP+DyAM 0.7163, NLP+OvO 0.7204 (trung bình 21); OvO > DyAM 7/8 tổ hợp lớn (+0.0112); NLP cải thiện cả hai mô hình (8/8 tổ hợp lớn). 1 seed đảo thứ hạng OvO/DyAM. DyAM 1 seed khớp tác giả tới một cặp bệnh nhân. Bảng: document/2026-10-13_reference-training-config/ref-all21-results.md.

### Chạy lại mọi số liệu của bài trong ref-linux-py38 (2026-10-13, §2.65)
Bảng 21 tổ hợp, 20-seed (nlp20/miss/loso), T1/T2, Bảng 2–5 (+tdAUC) đều chạy lại với `ctx.rad_filters` gốc; bản làm-sạch-cột cũ để ở `results/ref_cleanfilter/`. Bài, PDF, Word cập nhật. BM1 5 seed: DyAM 0.763, OvO 0.771, DyAM+NLP 0.786, OvO+NLP 0.785; 20 seed OvO+NLP 0.787 vs DyAM+NLP 0.781. Chưa chạy lại: Hình 4/5/S3/S4 (phân hoạch đơn gốc), ước lượng thổi phồng.

### Hình 4/5/S3/S4 dựng lại từ dữ liệu thật (2026-10-13, §2.65b)
Hình KM cũ là mô phỏng; thay bằng KM thật (paper/make_real_figs.py). Chia nhóm tại trung vị, không phải 0. IBS bỏ.

## 2026-10-14 — Cách chia fold và dao động giữa các seed (kế hoạch ghi TRƯỚC khi chạy)
- **Câu hỏi:** dao động AUC giữa các seed (sd ≈ 0.02) có giảm khi cách chia fold cân bằng theo nhãn và theo mẫu nguồn có mặt của bệnh nhân không? (giả thuyết của người dùng: số nguồn khác nhau giữa các bệnh nhân làm cách chia fold ảnh hưởng).
- **Thiết kế:** BM1 (Rad+IHC-G+Gen+PDL1), DyAM và OvO, 20 seed (SEEDS20), ref-linux-py38, mã `experiments/reference_env/run_fold_strat_ref.py`. Ba cách chia: random (chuẩn), label (phân tầng nhãn), pattern (nhãn x mẫu nguồn có mặt, tầng < 10 gộp 'rare'). Chỉ đổi bộ chia fold.
- **Chỉ số định trước:** sd của AUC giữa 20 seed theo từng cách chia; so sánh pattern với label (tách tác dụng của mẫu nguồn khỏi tác dụng của nhãn) và với random. Không có ngưỡng "thắng/thua": đây là thí nghiệm chẩn đoán, không đổi số trong bài.
- **Diễn giải định trước:** nếu sd(pattern) thấp hơn rõ sd(label) → số nguồn có mặt góp phần; nếu sd(label) ≈ sd(pattern) < sd(random) → chủ yếu do tỉ lệ nhãn từng fold; nếu cả ba ≈ nhau → do cỡ mẫu/bộ lọc L1.

## 2026-10-14 — Kiểm chứng NGOÀI DyAM / OvO / DyAM+NLP / OvO+NLP trên path_valid (kế hoạch ghi TRƯỚC khi chạy)
- **Câu hỏi:** hướng lợi thế của OvO so với DyAM và của NLP so với Labs thô (thấy ở CV nội bộ, §2.64/§2.65) có giữ ở cohort ngoài không?
- **Thiết kế:** huấn luyện trên TOÀN BỘ discovery (n=247), dự đoán path_valid (n=71, 21 đáp ứng) MỘT lần. Chạy một lần, KHÔNG chỉnh sau khi xem số ngoài. env ref-linux-py38, mã `experiments/extval_ref/run_extval_ref.py`, bộ nhớ đệm `experiments/extval_ref/make_cache.py` (embedding NLP6 tính ở môi trường chính vì ref env không có sentence-transformers).
- **Ba bộ nguồn** (cùng nguồn có ở ngoài: GLCM, PD-L1 lâm sàng, 6 biến lâm sàng): A1 = GLCM+PD-L1 · A2 = GLCM+PD-L1+Labs6 thô · A3 = GLCM+PD-L1+NLP6. Hai mô hình (DyAM, OvO) mỗi bộ → DyAM=A2, OvO=A2, DyAM+NLP=A3, OvO+NLP=A3; A1 là đối chứng không lâm sàng. Thêm đối chứng PD-L1 đơn nguồn.
- **Chạy:** 1 seed (seed 42, thứ tự tự nhiên) và 5 seed (42,7,123,2024,31337, hoán vị thứ tự hàng như harness). Cấu hình chuẩn §0 (125 epoch, lr 0.01, alpha 0.001, beta 0). Báo: AUC seed 42 + CI bootstrap bệnh nhân; mean±sd của AUC từng seed; AUC của điểm trung bình 5 seed (ghi riêng là ensemble); CV 10-fold nội bộ cùng bộ nguồn rút gọn trong cùng env để thấy khoảng cách nội bộ→ngoài; Δ bootstrap ghép cặp OvO−DyAM và NLP6−Labs6.
- **Lệch chuẩn (phải ghi):** 6 biến lâm sàng thay vì 13 (NLP6 ≠ NLP13: ở discovery 6 biến này chỉ ~0.496 nên arm NLP KHÔNG kiểm chứng phát hiện NLP13); không phải 21 tổ hợp; không radiomics/genomics (cohort ngoài không có); embedding tính ngoài ref env. rad_valid không chạy (chỉ có radiomics, đã ≤0.46 ở §2.3).
- **Diễn giải định trước:** n_đáp ứng=21 nên CI ≈ ±0.1; chênh +0.01…+0.03 không xác nhận/bác bỏ được. Kết quả chỉ nói về HƯỚNG. Không so với số §2.36 (Windows).

- **Kết quả §2.81 (2026-10-14):** sd AUC BM1 20 seed: DyAM random 0.0157 / label 0.0116 / pattern 0.0151; OvO 0.0178 / 0.0127 / 0.0128. Theo quy tắc định trước: pattern không thấp hơn rõ label, nên mẫu nguồn không phải nguồn dao động chính; dao động còn lại chủ yếu do cỡ mẫu và L1. OvO−DyAM +0.009..+0.012 ở cả ba cách. Chi tiết: result/training-results.md §2.81.
