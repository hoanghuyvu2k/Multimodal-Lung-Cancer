# Môi trường chạy của bài báo gốc và kết quả tái hiện (2026-10-12)

Nhánh `main` là code gốc của bài báo (commit `0704f80`, 2026-03-17). Tài liệu này ghi môi trường mà code gốc khai báo và kết quả khi dựng lại đúng môi trường đó.

## 1. Code gốc khai báo phiên bản nào?

Hai nguồn trong nhánh `main`:

- `requirements.txt` (198 dòng, kiểu `pip freeze`, ghim đủ phiên bản).
- Siêu dữ liệu của `Figures-Finalized.ipynb`: `language_info.version = 3.9.13`, kernel `Python 3 (ipykernel)`, `nbformat 4.4`.

Các gói quyết định kết quả:

| Gói | Bản gốc (`main`) | Môi trường chính hiện tại | `doan-env` (GPU) | `doan-orig` (mới dựng) |
|---|---|---|---|---|
| Python | 3.9.13 | 3.13.2 | 3.9.23 | **3.9.13** |
| torch | 1.11.0 | 2.13.0 | 2.5.1 | **1.11.0** |
| numpy | 1.22.4 | 1.26.4 | 1.22.4 | **1.22.4** |
| scipy | 1.8.1 | 1.14.1 | 1.8.1 | **1.8.1** |
| scikit-learn | 1.1.1 | 1.6.1 | 1.1.1 | **1.1.1** |
| pandas | 1.4.2 | 2.3.3 | 1.4.2 | **1.4.2** |
| lifelines | 0.27.1 | 0.30.3 | 0.27.1 | **0.27.1** |
| pyarrow / matplotlib / seaborn | 8.0.0 / 3.5.2 / 0.11.2 | - | - | cùng bản gốc |

Ghi chú: `requirements.txt` còn có `pyradiomics==3.0.1`, `SimpleITK==2.1.1.2`, `statannotations==0.4.4`; hai gói đầu không cần để chạy DyAM trên đặc trưng đã trích sẵn.

## 2. Dựng lại môi trường

```bash
conda create -n doan-orig python=3.9.13 -y
conda run -n doan-orig python -m pip install torch==1.11.0 numpy==1.22.4 scipy==1.8.1 \
  scikit-learn==1.1.1 pandas==1.4.2 lifelines==0.27.1 pyarrow==8.0.0 openpyxl==3.0.10 \
  matplotlib==3.5.2 seaborn==0.11.2 tqdm==4.64.0 joblib==1.1.0 threadpoolctl==3.1.0 \
  requests==2.28.0 statannotations==0.4.4
```

Môi trường: `C:\Users\Admin\miniconda3\envs\doan-orig` (torch bản CPU `1.11.0+cpu`). Chỉ cài các gói thư viện chính; không cài toàn bộ 198 gói (gồm jupyter, dask, docker...) vì không ảnh hưởng đến tính toán.

## 3. Kết quả: chạy `lung_helpers.py` NGUYÊN BẢN từ `main` trong `doan-orig`

Script: `experiments/nlp_verify/` không dùng; script tạm ở scratchpad gọi `train()` của `main`, bộ lọc radiomics `l1_strength 0.1`, `epochs 125, lr 0.01, alpha 0.001, beta 0.0`. Kết quả thô: `experiments/results/orig_env_repro.{json,log}`.

| Tổ hợp (DyAM, 1 lần chạy) | `doan-orig` | `.ipynb` ở thư mục làm việc (chạy lại trên máy này) | output lưu của tác giả (`.html` = `.ipynb` của `main`) |
|---|---|---|---|
| Rad+IHC-A+Gen+PDL1 | 0.7640 | 0.7640 | **0.7982** |
| Rad+IHC-G+Gen+PDL1 | 0.7839 | 0.7839 | 0.7833 |
| Rad+IHC-G+Gen | 0.7871 | 0.7871 | 0.7826 |
| Rad+Gen | 0.7384 | 0.7384 | 0.7593 |
| Rad | 0.6871 | 0.6871 | 0.7014 |
| IHC-A+Gen (không radiomics) | 0.7148 | 0.7148 | 0.7150 |

## 4. Kết luận

1. Dựng lại đúng môi trường ghim cho kết quả **trùng đến 6 chữ số với bản `.ipynb` ở thư mục làm việc** (bản đã chạy lại trên máy này). Đây là bản chạy lại, không phải output của tác giả.
2. Output lưu của tác giả (`.html`, và output trong `.ipynb` của `main`, hai file cho cùng số; bản phát hành Synapse 2022-08-18) khác ở các tổ hợp có radiomics (lệch tới 0.034), trong khi các hàng không radiomics khớp (lệch ≤0.0004). Code, dữ liệu và danh sách phiên bản đều giống bản Synapse (xem mục 5), nên khác biệt không do phiên bản thư viện đã ghim; chưa rõ nguyên nhân (xem sai-khac-can-dieu-tra.md).
3. Con số **0.798** của `README.md` mục 1.4 và con số ~0.80 của bài đến từ HTML. Ngay trong môi trường gốc, code gốc cho 0.764 với cùng tổ hợp. Con số 0.80 vì vậy không tái hiện được bằng dữ liệu và code hiện có, và nên coi là kết quả của một lần chạy với đầu vào radiomics khác.
4. Các con số "tái tạo khớp bài ≤0.003" trước đây (BM1–BM4) đúng với output lưu của tác giả; riêng Rad+IHC-A+Gen+PDL1 (0.798) không khớp.

## 5. Xác minh nguồn gốc bằng MD5 (cache Synapse, 2026-10-12)

`~/.synapseCache` giữ `content_md5` lúc tải về từ Synapse (2025-04-30). So với file hiện có:

- 9 file dữ liệu trong `../datasets/` (omnibus CSV, cohort listing, genomic, PD-L1, 2 radiomics, 4 pathology): **khớp MD5**.
- Nhánh `main`: `lung_helpers.py` (1e18c21c), `Figures-Finalized.ipynb` (90b332ac), `requirements.txt` (6eabf1b1), `Figures-Finalized.html` (98fa7f63), `Figures-Finalized-Permutation-Test.ipynb` (407122dc): **khớp MD5** với bản Synapse. Nhánh `main` là bản gốc nguyên vẹn.
- Thư mục làm việc (nhánh `feature/update-ovo`): `lung_helpers.py`, `Figures-Finalized.ipynb`, `requirements.txt` **khác** bản gốc (đã sửa/chạy lại).

**Đính chính (2026-10-12):** các ghi chú trước đó trong tài liệu này, trong README và trong `result/training-results.md` §2.59–2.60 coi output của `.ipynb` ở thư mục làm việc là "output của repo/tác giả". Điều đó sai: đó là bản chạy lại. Output lưu của tác giả (trong `.ipynb` của `main`) giống hệt `.html` (0.798169, 0.759285, 0.626938…). Chỗ ghi "HTML chạy 2025-04-30" cũng sai: 2025-04-30 là ngày tải về, bản phát hành là 2022-08-18.

## 6. Cập nhật 2026-10-13: scikit-learn thực của tác giả là 0.24.x, không phải 1.1.1

`requirements.txt` ghi `scikit-learn==1.1.1`, nhưng chạy lại notebook gốc chỉ tái tạo được output lưu của tác giả (LR Rad-PC 0.639341, LR Rad-LN 0.626938, DyAM Rad 0.7014, DyAM Rad+IHC-A 0.680753) với **scikit-learn 0.24.2** (và numpy 1.20.x). Với 1.0.x và 1.1.x trở lên các hàng radiomics lệch. Conda env `doan-orig` đã được cố định ở: Python 3.9.13, scikit-learn 0.24.2, numpy 1.20.3, scipy 1.8.1, pandas 1.4.2, torch 1.11.0+cpu, lifelines 0.27.1 (bảng ở mục 1 cột `doan-orig` là bản đầu tiên, trước khi hạ scikit-learn và numpy).

Các hàng DyAM gộp Genomics với nguồn khác vẫn lệch (hàng 0.798169 ra 0.787881). Xem `sai-khac-can-dieu-tra.md` D13–D15.

## 7. Cập nhật 2026-10-13: TÁI HIỆN THÀNH CÔNG trên Linux

Chạy notebook gốc trên Linux (WSL2 Ubuntu-22.04; Python 3.8.15, torch 1.8.1+cpu, scikit-learn 0.24.2, numpy 1.20.3, scipy 1.6.3, pandas 1.2.5) cho DyAM Rad+IHC-A+Gen+PDL1 = **0.798256** so với 0.798169 của tác giả (lệch đúng một cặp bệnh nhân, 0.000087), Rad+Gen và Rad+IHC-A+Gen khớp 6 chữ số. Trên Windows cùng phiên bản thư viện chỉ ra 0.7879. Vậy mục 4 và mục 6 ("không tái hiện được 0.798") đúng cho Windows; nguyên nhân là hệ điều hành/thư viện số học, không phải code hay dữ liệu. Chi tiết và bảng: `sai-khac-can-dieu-tra.md` D17. Môi trường Linux ở `D:\envs\py38sa`, script ở `experiments/nlp_verify/linux_repro/`.
