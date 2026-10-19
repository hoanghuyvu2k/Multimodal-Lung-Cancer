# Các chỗ sai khác cần điều tra sau (tái hiện bài báo gốc)

Tạo 2026-10-12. Mỗi mục: hiện tượng, bằng chứng, giả thuyết chưa kiểm chứng. Chưa có mục nào được kết luận.
Môi trường tái hiện: conda `doan-orig` (Python 3.9.13, torch 1.11.0, numpy 1.22.4, scipy 1.8.1, scikit-learn 1.1.1, pandas 1.4.2, lifelines 0.27.1), code = `lung_helpers.py` nguyên bản của nhánh `main`. Xem `original-environment.md`.

## D1. Output lưu của tác giả (`.html` = `.ipynb` của `main`) khác lần chạy lại ở các tổ hợp có radiomics

> **Đính chính 2026-10-12:** bản trước của mục này nói "HTML khác `.ipynb`". Sai: `.ipynb` gốc của tác giả (nhánh `main`, trùng MD5 với Synapse) có output giống hệt HTML. Bản `.ipynb` ở thư mục làm việc là bản chạy lại trên máy này. Xem D11.

- Output lưu của tác giả (HTML và `.ipynb` của `main`) khác lần chạy lại trong môi trường ghim; lần chạy lại trùng 6 chữ số với `.ipynb` ở thư mục làm việc (cũng là bản chạy lại).
- Lệch (orig-env trừ HTML): Rad+IHC-A+Gen+PDL1 −0.0342 (0.7640 so với 0.7982), Rad+Gen −0.0209, Rad −0.0143, Rad+IHC-G+Gen +0.0045, Rad+IHC-G+Gen+PDL1 +0.0005. Hàng không radiomics (IHC-A+Gen, Gen, IHC-A) khớp ≤0.0002.
- Con số ~0.80 của bài (README mục 1.4: 0.798 và 0.759) lấy từ HTML, không tái hiện được.
- Giả thuyết cần kiểm: (a) bước lọc radiomics `l1_filter_features_list` ở lần chạy HTML khác phiên bản đang có (kiểm: so với phiên bản `lung_helpers.py` cũ hơn nếu còn trong lịch sử, kiểm tra các tham số `robustness_cutoff`/`outlier_cutoff`); (b) file dữ liệu radiomics lúc chạy HTML khác bản hiện tại (các `.parquet` gốc đều ngày 2025-04-30 18:36, cùng ngày HTML; nhưng chưa có băm file để chứng minh giống); (c) ngẫu nhiên của bước chọn đặc trưng elastic-net/L1 phụ thuộc phiên bản BLAS/CPU; (d) HTML chạy trên GPU (kiểm: chạy `lung_helpers.py` của `main` trên GPU trong môi trường cũ).

## D2. Hạt giống ngẫu nhiên không có tác dụng trong code gốc

- `AttentionMatrix.__init__` gọi cứng `torch.manual_seed(42)` (cả ở `main`); huấn luyện dùng toàn bộ dữ liệu mỗi bước, nên kết quả một lần chạy đơn không đổi theo seed.
- Hệ quả: dao động của lần chạy đơn hoàn toàn đến từ cách chia fold (`KFold(random_state=0)` theo thứ tự bệnh nhân). Xem `result/training-results.md` §2.58. Phân hoạch của notebook cao hơn 89–100% các phân hoạch hoán vị khác.

## D3. Hàm gốc báo lỗi `0 feature(s)` ở một số phân hoạch

- `lung_helpers.train` / `train_ovo` nguyên bản dừng với `ValueError: Found array with 0 feature(s)` khi bộ lọc radiomics loại hết đặc trưng của một nguồn trong một fold. Xảy ra ở 2/20 phân hoạch hoán vị (thứ 9 và 10) ở BM1, BM2, BM3, cho cả DyAM lẫn OvO. Không xảy ra ở phân hoạch tự nhiên của notebook.

## D4. Hàng "DyAM TMB+PDL1" trong notebook là PDL1+Gen

- Ô 17 của notebook chạy hai lần cùng khoá `summary_dfs['DyAM TMB+PDL1']`: lần đầu với `['gen_driver_tmb','cnl_pdl1_score']` (gate tắt), lần sau với `['cnl_pdl1_score','gen_driver_mut_amp']` (gate bật). Lần sau ghi đè, nên hàng mang nhãn TMB+PDL1 (AUC 0.693) thực chất là PDL1+Gen. Khớp với PDL1+Gen của harness (0.6932). Cần xem lại nhãn này khi trích số từ bài gốc.

## D5. README mục 1.4 ghi nguồn "từ bài báo" nhưng số lấy từ HTML

- Đã thêm ghi chú nguồn vào README. Còn cần xác nhận số trong bản bài báo công bố (chưa có bản đó).

## D6. Bộ nhớ đệm dữ liệu của harness

- `experiments/common/data_setup.py` nạp từ pickle `attn_redesign_data_cache.pkl` (trong thư mục tạm của một phiên trước), tạo bằng code hiện tại; `prepare_rad_modality_by_size` hiện thêm `select_dtypes(include=number)` mà bản `main` không có. Lần chạy ô-notebook nạp dữ liệu từ đầu bằng code `main` để loại khả năng này (kết quả ghi ở mục D7).

## D7. Kết quả chạy từng ô notebook (đã chạy xong, 2026-10-12)

Chạy các ô 1–13, 16, 17 của `Figures-Finalized.ipynb` (nhánh `main`), môi trường `doan-orig`, `lung_helpers.py` của `main`, nạp dữ liệu từ đầu từ file `.parquet` (KHÔNG dùng bộ nhớ đệm). Số thô: `experiments/results/notebook_cells_orig_env.{log,json}`. Bỏ ô 14 (lưu bảng), ô 19–25 (subsample, MILR, kiểm định ngoài).

| Mô hình (AUC, một lần chạy) | chạy ô-notebook | HTML | lệch |
|---|---|---|---|
| LR Clinical / IHC-G / IHC-A / PDL1-TPS | 0.5695 / 0.6338 / 0.6240 / 0.7295 | 0.5695 / 0.6338 / 0.6243 / 0.7293 | ≤0.0004 |
| LR Gen-Only-TMB / Gen-No-TMB / Gen-Combined / Gen-Average | 0.6069 / 0.6051 / 0.6548 / 0.6580 | giống hệt | 0.0000 |
| LR Path-A-Average / Path-G-Average | 0.6951 / 0.6942 | giống hệt | 0.0000 |
| DyAM IHC-A / Gen / IHC-A+Gen | 0.6056 / 0.6758 / 0.7148 | 0.6056 / 0.6758 / 0.7150 | ≤0.0002 |
| DyAM "TMB+PDL1" (thật ra PDL1+Gen) | 0.6931 | 0.6931 | 0.0000 |
| **LR Rad-PC** | 0.6407 | 0.6393 | +0.0014 |
| **LR Rad-LN** | 0.6812 | 0.6269 | **+0.0543** |
| **LR Rad-Average** | 0.6564 | 0.6448 | +0.0117 |
| **DyAM Rad** | 0.6871 | 0.7014 | −0.0143 |
| **DyAM Rad+IHC-A** | 0.6730 | 0.6808 | −0.0078 |
| **DyAM Rad+Gen** | 0.7384 | 0.7593 | −0.0209 |
| **DyAM Rad+IHC-A+Gen** | 0.7567 | 0.7542 | +0.0024 |
| **DyAM Rad+IHC-G+Gen** | 0.7871 | 0.7826 | +0.0045 |
| **DyAM Rad+IHC-A+Gen+PDL1** | 0.7640 | **0.7982** | **−0.0342** |
| **DyAM Rad+IHC-G+Gen+PDL1** | 0.7839 | 0.7833 | +0.0005 |
| LR Multimodal-Average | 0.7292 | 0.7235 | +0.0057 |

Kết luận ban đầu (chưa kiểm chứng nguyên nhân):

1. **Mọi hàng không chứa radiomics khớp HTML (≤0.0004), mọi hàng chứa radiomics lệch** (0.0005 đến 0.054). Phần còn lại của notebook (dữ liệu genomics, PD-L1, bệnh học, Labs, mô hình LR và DyAM) tái hiện đúng.
2. **Lệch xuất hiện cả ở mô hình LR đơn nguồn** (LR Rad-LN +0.054, LR Rad-Average +0.012), nên nguyên nhân KHÔNG nằm ở DyAM. Nghi ngờ chuyển sang dữ liệu radiomics, cách chọn tổn thương (`decorate_with_site_index`, `prepare_rad_modality_by_size`) hoặc bước lọc radiomics `l1_filter_features_list`. LN (hạch) lệch nhiều nhất.
3. **D6 đã loại:** dữ liệu nạp từ đầu bằng code `main` cho số trùng với bộ nhớ đệm của harness (ví dụ Rad+IHC-A+Gen+PDL1 0.763993 ở cả hai). Bộ nhớ đệm và `select_dtypes` không phải nguyên nhân.
4. Cách kiểm tra đề xuất cho lần điều tra sau: (a) so từng lesion/site (số bệnh nhân có PC/PL/LN, thứ tự tổn thương theo kích thước, trùng kích thước) giữa HTML và lần chạy lại, HTML có in `np.unique` của số tổn thương ở ô 8; (b) in tập đặc trưng radiomics được chọn ở từng fold cho LR Rad-LN; (c) băm SHA256 các file `.parquet` radiomics rồi so với nguồn Synapse gốc nếu có; (d) chạy trên GPU/CPU khác nếu nghi ngờ phép toán số.

## D8. Hàng "LR Rad-PL" có AUC 0.2885 (dưới ngẫu nhiên)

- Chạy ô-notebook cho `LR Rad-PL` = 0.2885 (n nhỏ: PL chỉ 21 bệnh nhân, xem ô 8). HTML không có hàng này trong bảng AUC. Chưa rõ vì sao HTML bỏ hàng này; cần xem ô tổng hợp của HTML. Cũng là một chỉ dấu của tính không ổn định của đường radiomics ở PL/LN (ít bệnh nhân).

## D9. Hàng chỉ có ở lần chạy lại, không có trong bảng HTML

- `DyAM TMB`, `DyAM PDL1`, `DyAM Rad-LU`, `DyAM Rad+IHC-G`, `DyAM IHC-G+Gen`, `DyAM Rad+IHC-A+MutAmp`, `DyAM Rad+IHC-A+Gen+TMB+PDL1`, `DyAM Rad+IHC-A+Gen+PDL1+Labs`, `DyAM Rad+IHC-G+Gen+PDL1+Labs`, `LR Clinical-Albumin`, `LR Rad-LU`: có trong notebook `main` nhưng không có trong bảng AUC của HTML (HTML có thể bị cắt hoặc là phiên bản notebook trước). Cần xem HTML có các ô tương ứng không. Hàng `Full Fit` (0.9046) là đánh giá trên chính dữ liệu huấn luyện, không dùng so sánh.
- Ngược lại, HTML có `MILR Rad-Lesions` mà lần chạy này chưa chạy (ô 21 bỏ qua).

## D10. Nguồn gốc trên Synapse (syn26642505), kiểm tra qua API công khai ngày 2026-10-12

Dự án: "Multimodal integration of radiology, pathology, and genomics for prediction of response to PD-L1 blockade in patients with non-small cell lung cancer". Tạo 2022-01-04, sửa 2022-11-21.

- Thư mục `code` (syn26828464, sửa 2022-08-18): `Figures-Finalized.ipynb` (syn26828481), `Figures-Finalized.html` (syn26828498), `Figures-Finalized-Permutation-Test.ipynb`/`.html`, `lung_helpers.py` (syn26828488), `requirements.txt` (syn34639082). Tức đúng bộ file có ở nhánh `main`.
- Thư mục `datasets` (syn34638177, sửa 2022-08-18): 9 file (omnibus CSV, `FINAL_COHORT_LISTING.csv`, 4 file pathology, 2 file radiomics, `PDL1_SCORE.parquet`, `genomic_data_v3.parquet`). Trùng tên với bản local.
- Thư mục `tests` (syn52364210, 2023-08-30): `null_permutation`, `null_permutation_MILR`.
- Cả `.ipynb` và `.html` trên Synapse đều được đưa lên 2022-08-18 (file local ngày 2025-04-30 là ngày tải về). Hai file này cho cùng output (xem D11); khác biệt là giữa output lưu của tác giả và lần chạy lại của ta.
- API ẩn danh chỉ xem được tên/ngày; tải file và xem MD5 đều trả 403 ("Anonymous users have only READ access permission"). Không có thông tin đăng nhập Synapse nào dùng được trên máy này (`~/.synapseConfig` rỗng, không có biến môi trường, `login()` báo `SynapseNoCredentialsError`), nhưng cache `~/.synapseCache` giữ `content_md5` lúc tải nên vẫn so được (xem D11).

## D11. Xác minh bằng MD5: code và dữ liệu giống hệt bản Synapse; output lưu của tác giả không tái hiện được

- `~/.synapseCache/**/.cacheMap` ghi `content_md5` của từng file lúc tải (2025-04-30). So với hiện tại: 9/9 file dữ liệu khớp; nhánh `main` khớp cho `lung_helpers.py`, `Figures-Finalized.ipynb`, `requirements.txt`, `Figures-Finalized.html`, `Figures-Finalized-Permutation-Test.ipynb`. Thư mục làm việc có 3 file code khác bản gốc (đã sửa/chạy lại).
- Output lưu trong `.ipynb` của `main` giống `.html`: DyAM Rad+IHC-A+Gen+PDL1 = 0.798169, Rad+Gen = 0.759285, LR Rad-LN = 0.626938, LR Rad-PC = 0.639341, LR Multimodal-Average = 0.72354.
- Chạy lại ô-notebook của `main` trên code + dữ liệu giống hệt, môi trường ghim (`doan-orig`): 0.763993, 0.738361, 0.6812, 0.6407, 0.7292. Hàng không radiomics khớp, hàng radiomics lệch.
- Phép thử loại trừ (thư mục `experiments/results/radiomics_*`):
  - Đổi seed `saga` (selection), `tol` 1e-3 và 1e-6, seed ngẫu nhiên toàn cục: LR Rad-LN chỉ dao động 0.680–0.688 → nhiễu số học không giải thích được (cần 0.627).
  - Quét 19 biến thể cấu trúc của bước chọn đặc trưng (ngưỡng ngoại lai, ngưỡng độ bền, bộ chuẩn hoá, l1_ratio, C): không biến thể nào tái tạo cả LR Rad-PC (0.6393) và LR Rad-LN (0.6269); cấu hình gốc gần nhất cho PC (0.6407); `robustscaler` gần nhất cho LN (0.6366) nhưng PC rơi xuống 0.5517.
  - Số bệnh nhân theo tổn thương (163/21/67 và 133/6/4/5/28/11) trùng với output của tác giả.
- Còn lại cần điều tra: (a) thứ tự bệnh nhân/phân hoạch KFold phụ thuộc phiên bản pandas (đang đo, `radiomics_order.log`); (b) môi trường thật của tác giả khác `requirements.txt` (OS, BLAS, bản sklearn/scipy thực dùng); (c) `lung_helpers.py` của tác giả khi chạy khác bản đã lưu (ví dụ chạy trước khi sửa code).

## D12. Độ nhạy theo thứ tự bệnh nhân và loại trừ phiên bản pandas (2026-10-12)

- **Thứ tự bệnh nhân quyết định phân hoạch.** `KFold(random_state=0)` chia theo vị trí hàng; chỉ số bệnh nhân của nhóm radiomics không được sắp xếp. Cấu hình gốc, `train_LR`, 30 thứ tự (`experiments/results/radiomics_order.{log,json}`): LR Rad-PC thứ tự gốc 0.6407, 29 thứ tự khác mean 0.6275 sd 0.021 [0.593, 0.671]; LR Rad-LN thứ tự gốc 0.6812, 29 thứ tự khác mean 0.6548 sd 0.041 [0.564, 0.707]. Giá trị của tác giả (PC 0.6393, LN 0.6269) nằm trong phân bố này (phân vị 69% và 24%). Chênh 0.054 ở LN chỉ cỡ 1,3 sd, nên một thứ tự bệnh nhân khác là đủ để giải thích; **cơ chế làm thứ tự khác chưa xác định** (chưa kiểm: sắp xếp trong `prepare_rad_modality_by_size`, thứ tự hàng của parquet, `groupby`/`dropna`).
- **Loại trừ pandas khác phiên bản:** `lung_helpers.py` gốc dòng 66 dùng `pd.core.common.SettingWithCopyWarning`, chỉ tồn tại ở pandas ≤1.4.x. Với pandas 1.5.3, 2.0.3, 2.2.3 code gốc lỗi `AttributeError` ngay khi nhập (`experiments/results/pandas_*.log`). Vậy tác giả bắt buộc dùng pandas ≤1.4.x (khớp `requirements.txt` 1.4.2); khác biệt giữa 1.4.x và 1.4.2 khó đổi thứ tự.
- Còn lại để điều tra: (a) thứ tự hàng của parquet radiomics và hành vi sắp xếp/`groupby` ở numpy, python khác nhau (numpy 1.22.4 đã dùng; python 3.9.13 đã dùng); (b) tác giả chạy ô theo thứ tự khác trong notebook (HTML ghi `In [2]` hai lần, có thể chạy lại ô giữa chừng; ô 9 nhân `-1/100` vào PD-L1 mỗi lần chạy lại); (c) hệ điều hành/BLAS khác (Windows 11 ở đây); (d) `lung_helpers.py` lúc tác giả chạy là bản khác bản đã lưu lên Synapse.

## D13. Phiên bản scikit-learn quyết định LR Rad-PC; scipy/numpy/số luồng không (2026-10-12)

Mục tiêu 6 chữ số (output lưu của tác giả): LR Rad-PC 0.639341, LR Rad-LN 0.626938. Cùng ô notebook, `lung_helpers.py` của `main`, dữ liệu gốc; chỉ đổi thư viện (cài vào thư mục `--target`, đặt qua `PYTHONPATH`, in phiên bản thực được nạp).

| scikit-learn | scipy | numpy | LR Rad-PC | LR Rad-LN |
|---|---|---|---|---|
| 1.1.0 / 1.1.1 / 1.1.2 / 1.1.3 / 1.2.2 / 1.3.2 | 1.8.1 | 1.22.4 | 0.640698 | 0.681202 (1.3.2: 0.638566) |
| 1.0.0 | 1.8.1 | 1.22.4 | **0.639341** | 0.646318 |
| 1.0.1 | 1.8.1 | 1.22.4 | **0.639341** | 0.647287 |
| 1.0.2 | 1.8.1 | 1.22.4 | **0.639341** | 0.646318 |
| 1.0.2 | 1.7.3 / 1.8.0 / 1.9.0 / 1.9.3 / 1.10.1 | 1.22.4 | **0.639341** | 0.646318 |
| 1.0.2 | 1.8.1 | 1.21.6 | **0.639341** | 0.646318 |
| 1.0.2 | 1.8.1 | 1.23.5 | 0.639147 | 0.646318 |
| 1.0.2, 1 luồng BLAS | 1.8.1 | 1.22.4 | 0.639147 | 0.646318 |
| 1.0.2, 2–12 luồng | 1.8.1 | 1.22.4 | **0.639341** | 0.647287 |

- **scikit-learn ≤1.0.x tái tạo đúng 6 chữ số LR Rad-PC (0.639341)**; từ 1.1.0 trở lên không. Nghĩa là output lưu của tác giả sinh ra với scikit-learn 1.0.x (hoặc cũ hơn), **không phải 1.1.1 như `requirements.txt` ghi**. `requirements.txt` không phản ánh môi trường lúc chạy notebook.
- scipy 1.7.3–1.10.1 và numpy 1.21.6–1.23.5 không đổi PC/LN (trừ đổi cỡ 2e-4 ở một số tổ hợp numpy/số luồng).
- LR Rad-LN với scikit-learn 1.0.x là 0.646–0.647, vẫn cách đích 0.627 khoảng 0.02. Cần thêm một biến số (đang thử scikit-learn 0.24.2 và các thành phần khác).
- Bài học phương pháp: lần đầu quét scipy/numpy ghép hai đường dẫn bằng `:` nên Windows không nhận; các hàng đó vô hiệu và đã chạy lại bằng `;`. Mọi dòng kết quả nay in kèm phiên bản thực được nạp.

## D14. scikit-learn 0.24.x tái tạo các hàng radiomics; phần lệch còn lại là sai số số học (2026-10-12/13)

**Kết quả chính.** Môi trường tái hiện gần nhất: scikit-learn **0.24.2** (không phải 1.1.1 như `requirements.txt`), scipy 1.8.1, numpy 1.20.3 (hoặc 1.22.4), pandas 1.4.2, torch 1.11.0, Python 3.9.13, `lung_helpers.py` và dữ liệu gốc (khớp MD5 Synapse).

Chạy toàn bộ ô 1–13, 16, 17 của notebook gốc (sklearn 0.24.2, numpy 1.22.4; `experiments/results/notebook_cells_sk0242.{log,json}`), so với output lưu của tác giả:

| Hàng | chạy lại | tác giả | lệch |
|---|---|---|---|
| LR Rad-PC / LR Rad-LN | 0.639341 / 0.626938 | 0.639341 / 0.626938 | **0.000000** |
| DyAM Rad | 0.7014 | 0.7014 | **0.0000** |
| DyAM Rad+IHC-A | 0.6808 | 0.6808 | **0.0000** |
| DyAM IHC-A / Gen / PDL1+Gen | 0.6056 / 0.6758 / 0.6931 | 0.6056 / 0.6758 / 0.6931 | ≤0.00004 |
| DyAM Rad+Gen | 0.7578 | 0.7593 | −0.0015 |
| DyAM Rad+IHC-G+Gen | 0.7814 | 0.7826 | −0.0011 |
| DyAM Rad+IHC-A+Gen | 0.7597 | 0.7542 | +0.0055 |
| DyAM Rad+IHC-G+Gen+PDL1 | 0.7785 | 0.7833 | −0.0049 |
| **DyAM Rad+IHC-A+Gen+PDL1** | 0.7879 | **0.7982** | **−0.0103** |
| LR Multimodal-Average | 0.7241 | 0.7235 | +0.0006 |

Với scikit-learn ≥1.0.x (kể cả 1.1.1 ghi trong `requirements.txt`), LR Rad-LN là 0.646–0.681 chứ không phải 0.627. Bản cũ hơn (0.24.x) cần dùng. Các lần chạy ở scikit-learn 1.1.0 trở lên cũng làm LR Rad-PC lệch +0.001.

**Các yếu tố đã loại (không đổi kết quả hoặc không khớp tốt hơn):** scipy 1.7.3–1.10.1; numpy (đổi nhỏ cỡ 1e-4); phiên bản torch 1.10.2, 1.11.0, 1.12.1, 1.13.1 (cho cùng kết quả đến 6 chữ số); số luồng BLAS cho LR (đổi cỡ 2e-4).

**Lưu ý quan trọng về tính tất định.**
- Các hàng LR dùng `saga` cuối không đặt seed (`random_state=None`), nên một số hàng LR (Rad-Average, Gen-Average, IHC-A, Gen-No-TMB) thay đổi từ lần chạy này sang lần khác. Không dùng các hàng này làm tiêu chí khớp tuyệt đối.
- Hàng DyAM một nguồn (IHC-A, Gen) khớp tuyệt đối với mọi torch; hàng hai nguồn không radiomics (PDL1+Gen, IHC-A+Gen) lệch 4e-5 và 2e-4 và không phụ thuộc phiên bản torch; nghi sai số dấu phẩy động theo phần cứng/thư viện số học của máy tác giả.
- Hàng DyAM đa nguồn có radiomics lệch tới 0.010, nhiều khả năng do sai số nhỏ lan truyền qua 125 epoch (chưa chứng minh). Chưa tái hiện được 0.798169 (tốt nhất 0.7879).

**Việc còn lại:** (a) xem số luồng tính toán và phần cứng có đổi hàng hai nguồn không (đang chạy); (b) kiểm tra vì sao hàng PDL1+Gen/IHC-A+Gen lệch nhẹ nếu torch không ảnh hưởng; (c) cân nhắc xem bước chọn đặc trưng radiomics ở scikit-learn 0.24.2 vs 1.0 có thể tăng/giảm cho từng fold.

## D15. Tổng kết vòng điều tra (2026-10-13): khớp một phần, phần còn lại chưa giải thích được

**Môi trường tái hiện tốt nhất** (cố định trong conda env `doan-orig`): Python 3.9.13, **scikit-learn 0.24.2**, numpy 1.20.3, scipy 1.8.1, pandas 1.4.2, torch 1.11.0+cpu, lifelines 0.27.1; `lung_helpers.py` nguyên bản của nhánh `main`; dữ liệu khớp MD5 Synapse. Kiểm tra lại sau khi cố định: LR Rad-PC 0.639341, LR Rad-LN 0.626938 (khớp 6 chữ số).

**Khớp tuyệt đối (6 chữ số) với output lưu của tác giả:** LR Rad-PC, LR Rad-LN, DyAM Rad (0.7014), DyAM Rad+IHC-A (0.680753, với torch ≥1.9.1), DyAM IHC-A (0.605631), DyAM Gen (0.675850).

**Còn lệch (tất định trên máy này, không đổi theo thư viện thử):**

| Hàng | chạy lại | tác giả | lệch |
|---|---|---|---|
| DyAM PDL1+Gen (nhãn TMB+PDL1) | 0.693069 | 0.693112 | −0.00004 |
| DyAM IHC-A+Gen | 0.714778 | 0.714996 | −0.0002 |
| DyAM Rad+Gen | 0.7578 | 0.7593 | −0.0015 |
| DyAM Rad+IHC-G+Gen | 0.7814 | 0.7826 | −0.0011 |
| DyAM Rad+IHC-A+Gen | 0.7597 | 0.7542 | +0.0055 |
| DyAM Rad+IHC-G+Gen+PDL1 | 0.7785 | 0.7833 | −0.0049 |
| **DyAM Rad+IHC-A+Gen+PDL1** | **0.787881** | **0.798169** | **−0.0103** |

**Mẫu hình:** mọi hàng khớp tuyệt đối đều là hàng KHÔNG gộp nguồn Genomics với nguồn khác; mọi hàng gộp Genomics (`gen_driver_mut_amp`) với nguồn khác đều lệch. Genomics đơn lẻ khớp.

**Đã thử và loại (không đổi hoặc không tới được đích):** phiên bản scikit-learn 0.24.0, 0.24.1, 1.0.0–1.0.2, 1.1.0–1.1.3, 1.2.2, 1.3.2 (chỉ 0.24.2 ra LR Rad-LN 0.626938); scipy 1.7.3–1.10.1; numpy 1.19.5–1.24.4; torch 1.8.1, 1.9.1, 1.10.2, 1.11.0, 1.12.1, 1.13.1 (hàng lệch giữ nguyên ở mọi bản; torch 1.8.1 chỉ làm lệch thêm Rad+IHC-A 2e-4); pandas 1.2.5, 1.3.5, 1.4.2 (pandas ≥1.5 không chạy được code gốc); số luồng BLAS/OMP (1, 2, 12) cho DyAM; seed `saga` của bước chọn đặc trưng, `tol`, seed toàn cục; 13 cấu hình tham số huấn luyện DyAM (cổng chú ý, cross-modality, alpha, beta, epochs, lr), cấu hình gốc gần đích nhất (IHC-A+Gen −0.0002; mọi biến thể khác lệch ≥0.001); 19 biến thể bước chọn đặc trưng radiomics; thứ tự bệnh nhân (không đổi thứ tự cho mọi nguồn); mã trong ô notebook (các lệnh `train(` giống hệt giữa HTML và `.ipynb`).

**Kết luận:** khác biệt còn lại nằm ở thứ không có trong gói ta có: (a) trạng thái `lung_helpers.py` lúc tác giả chạy (output có thể sinh từ năm 2021 khi mô hình hoặc hàm dữ liệu còn khác bản đã đưa lên Synapse tháng 8/2022); (b) hoặc sai số số học theo nền tảng (CPU/BLAS/hệ điều hành) tích luỹ qua 125 epoch ở các mô hình đa nguồn có Genomics. Chưa chứng minh được cái nào; không thể phân biệt khi chưa có nhật ký chạy gốc.

**Hướng còn lại (nếu cần tiếp):** liên hệ tác giả/xem nhánh `tests` và Permutation-Test trên Synapse để xem có file số liệu đi kèm; thử Python 3.8 + scikit-learn 0.24.2 + torch 1.8 trên Linux (hệ điều hành khác); đối chiếu `lung_helpers.py` với lịch sử commit của tác giả (nếu có).

## D16. Kiểm tra dữ liệu radiomics (2026-10-13): không phát hiện sai khác

Script `D:\repro_linux\radiomics_audit.py`. File `lung_radiomics_spacing1.0_mirpon_window1350.250_allimagetypes_bw20.parquet`: MD5 khớp cache Synapse (4127cabc), sha256 a22380aa…; file validation sha256 9fe5acd6…. Bảng 3996 dòng × 1690 cột (1688 đặc trưng float64), 187 bệnh nhân; `job_tag`: filtered 333, unfiltered 333, perturbation 3330 (đúng 10 bản nhiễu mỗi tổn thương); không NaN/inf; không trùng (bệnh nhân, tổn thương). Discovery: 247 trong danh sách, 187 có radiomics; PC 163, PL 21, LN 67 (khớp output của tác giả); số tổn thương mỗi bệnh nhân 1/2/3/4 = 89/60/28/10; 29/67 bệnh nhân LN có >1 tổn thương LN.
Đặc điểm nhạy số: 39 cột hằng, 93 cột std<1e-3, 29 cột có |giá trị|>1e10 (làm `PowerTransformer`/chọn đặc trưng nhạy với sai số làm tròn). Dữ liệu không có dấu hiệu khác bản gốc; nguyên nhân lệch còn lại nằm ở code/phiên bản/nền tảng.

## D17. TÁI HIỆN THÀNH CÔNG trên Linux (2026-10-13): nền tảng là biến số còn thiếu

**Môi trường:** WSL2 Ubuntu-22.04 (kernel 6.6.87.2-microsoft-standard-WSL2), Python 3.8.15 (python-build-standalone), torch 1.8.1+cpu, scikit-learn 0.24.2, numpy 1.20.3 (OpenBLAS), scipy 1.6.3, pandas 1.2.5, lifelines 0.26.0, matplotlib 3.4.2, seaborn 0.11.1, pyarrow 4.0.1. Cài hoàn toàn trên ổ D: (`D:\envs\py38sa`; conda không dùng được trên NTFS vì gói ncurses va chạm hoa/thường). `lung_helpers.py` và notebook của `main` (MD5 khớp Synapse), dữ liệu gốc, các ô 1–13, 16, 17. Kết quả: `experiments/results/notebook_cells_linux_py38.{log,json}`; script: `experiments/nlp_verify/linux_repro/`.

| Hàng DyAM | Linux | Tác giả | Lệch | Windows (cùng thư viện) |
|---|---|---|---|---|
| **Rad+IHC-A+Gen+PDL1** | **0.798256** | **0.798169** | **+0.000087** | 0.7879 (−0.0103) |
| Rad+IHC-G+Gen+PDL1 | 0.783435 | 0.783348 | +0.000087 | 0.7785 |
| Rad+Gen | 0.759285 | 0.759285 | **0.000000** | 0.7578 |
| Rad+IHC-A+Gen | 0.754228 | 0.754228 | **0.000000** | 0.7597 |
| Rad+IHC-G+Gen | 0.782476 | 0.782563 | −0.000087 | 0.7814 |
| Rad | 0.701413 | 0.701413 | 0.000000 | 0.7014 |
| IHC-A / Gen | 0.605631 / 0.675850 | cùng số | 0.000000 | cùng số |
| IHC-A+Gen | 0.714778 | 0.714996 | −0.000218 | −0.000218 |
| PDL1+Gen (nhãn TMB+PDL1) | 0.693069 | 0.693112 | −0.000043 | −0.000043 |
| Rad+IHC-A | 0.680527 | 0.680753 | −0.000226 | 0.680753 (torch ≥1.9.1) |
| LR Rad-PC / LR Rad-LN | 0.639341 / 0.626938 | cùng số | 0.000000 | cùng số |

- Số bệnh nhân: 62 đáp ứng, 185 không đáp ứng, nên một cặp bệnh nhân đổi thứ hạng làm AUC lệch 1/(62×185)=0.0000872. Các lệch ±0.000087 là đúng một cặp. Hàng đích 0.798169 được tái hiện đến mức một cặp bệnh nhân.
- 11/25 hàng khớp tuyệt đối (6 chữ số); các hàng LR còn lệch ≤0.003 do `saga` cuối không đặt seed (không tất định giữa các lần chạy).
- Lệch ≤0.0003 còn lại ở vài hàng DyAM ít nguồn (IHC-A+Gen, Rad+IHC-A, PDL1+Gen), cùng giá trị trên cả Windows lẫn Linux, nên không do nền tảng; có thể do phiên bản torch hoặc thư viện cũ hơn nữa (chưa điều tra).
- Kết luận: các D14–D15 ghi "chưa tái hiện được 0.798" là đúng với Windows; nguyên nhân là hệ điều hành/thư viện số học (OpenBLAS Linux so với bản Windows), không phải code, dữ liệu hay tham số. Cần scikit-learn 0.24.2 (không phải 1.1.1 như `requirements.txt`).
