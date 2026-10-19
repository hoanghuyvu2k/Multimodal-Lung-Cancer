# Cấu hình train tham chiếu `ref-linux-py38` (2026-10-13)

Cấu hình tái hiện được mức cao nhất của bài báo gốc (DyAM Rad+IHC-A+Gen+PDL1: AUC 0.798169 trong output lưu của tác giả; chạy lại 0.798256, lệch đúng một cặp bệnh nhân). Từ nay mọi lần train dùng cấu hình này (xem rule trong `CLAUDE.md`). Bản máy đọc: `experiments/reference_env/reference_config.json`.

## 1. Môi trường

| Mục | Giá trị |
|---|---|
| Hệ điều hành | Linux, WSL2 Ubuntu-22.04 (kernel 6.6.87.2-microsoft-standard-WSL2) |
| Python | 3.8.15 (python-build-standalone), ở `D:\envs\py38sa` |
| torch | 1.8.1+cpu |
| scikit-learn | **0.24.2** (không phải 1.1.1 như `requirements.txt` của tác giả) |
| numpy / scipy / pandas | 1.20.3 (OpenBLAS) / 1.6.3 / 1.2.5 |
| lifelines / pyarrow | 0.26.0 / 4.0.1 |
| matplotlib / seaborn / statannotations | 3.4.2 / 0.11.1 / 0.4.4 |
| Biến môi trường | `MPLBACKEND=Agg`, `OMP_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4`, `ATTN_CACHE_DIR=/mnt/d/repro_linux/cache` |

Dựng lại từ đầu: `experiments/reference_env/setup_ref_env.sh` (chạy trong WSL; mọi thứ ghi vào ổ D:). Không dùng conda: NTFS không phân biệt hoa/thường làm hỏng gói ncurses.

## 2. Cách chạy

Từ thư mục `code/` (Git Bash trên Windows):

```bash
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-22.04 -- bash /mnt/d/code/master/doan/code/experiments/reference_env/run_ref.sh experiments/<thư mục>/<script>.py [tham số]
```

`MSYS_NO_PATHCONV=1` bắt buộc, nếu không Git Bash đổi `/mnt/d/...` thành đường dẫn Windows. `run_ref.sh` đặt biến môi trường, `cd` vào `code/`, đặt `PYTHONPATH` và gọi Python 3.8.15.

## 3. Cấu hình huấn luyện (giữ nguyên chuẩn §0 của `result/training-results.md`)

- `MODEL_PARAMS`: epochs 125, lr 0.01, alpha 0.001, beta 0.0, `cross_modality_enabled` False.
- Đơn nguồn: `attention_gate_enabled=False` (như notebook gốc).
- Bộ lọc radiomics: `ctx.rad_filters` với `l1_strength=0.1` (không dùng `clean_rad_filters`).
- CV: `KFold(10, shuffle=True, random_state=0)`; cohort discovery n=247 (62 đáp ứng, 185 không đáp ứng).
- Lần chạy đơn: seed 42, thứ tự bệnh nhân tự nhiên (như notebook). Nhiều seed: `SEEDS=[42,7,123,2024,31337]`, hoán vị nhãn mỗi seed.
- Code: `lung_helpers.py` hiện tại của nhánh làm việc (đã kiểm tra cho cùng kết quả với `lung_helpers.py` gốc của `main` trong môi trường này).
- Dữ liệu: các file trong `../datasets/` (khớp MD5 Synapse).

## 4. Kiểm tra môi trường (chạy khi thay đổi môi trường hoặc nghi ngờ)

`experiments/reference_env/verify_ref_env.py` (chạy bằng `run_ref.sh`): DyAM Rad+IHC-A+Gen+PDL1, single-run seed 42, kỳ vọng **0.798256** (sai số 1e-5). Bằng chứng ban đầu: `experiments/results/ref_env_verify_2026-10-13.log`. Trong lần chạy notebook gốc trên môi trường này còn khớp 6 chữ số: LR Rad-PC 0.639341, LR Rad-LN 0.626938, DyAM Rad 0.701413, Rad+Gen 0.759285, Rad+IHC-A+Gen 0.754228 (`experiments/results/notebook_cells_linux_py38.log`).

## 5. Lưu ý và giới hạn

1. **Không so sánh trực tiếp số liệu giữa nền tảng.** Cùng thư viện nhưng chạy trên Windows, hàng 0.798 ra 0.7879. Các kết quả trong `result/training-results.md` §2.x trước 2026-10-13 sinh ra trên Windows (Python 3.13 hoặc `doan-env`): giữ nguyên, nhưng muốn so sánh OvO và DyAM thì chạy cả hai trong cùng một môi trường. Số liệu trong bài báo (Bảng 4, 5...) hiện sinh trên Windows; nếu muốn đồng nhất cần chạy lại trên môi trường tham chiếu.
2. Các hàng LR dùng `saga` cuối không đặt seed nên không tất định giữa các lần chạy; không dùng chúng làm tiêu chí khớp tuyệt đối.
3. Lệch ±0.000087 là một cặp bệnh nhân đổi thứ hạng (62×185 cặp), không phải lỗi.
4. Môi trường không có `sentence-transformers`; nhánh NLP dùng embedding đã cache (`datasets/nlp_clinical_embed.parquet`). Muốn tạo embedding mới thì làm ở môi trường Windows chính rồi cache lại.
5. Không có GPU trong môi trường này (torch CPU). Muốn dùng GPU thì chạy ở môi trường khác và ghi rõ là lệch khỏi cấu hình tham chiếu.
6. Hai bản scikit-learn khác nhau cho kết quả khác nhau (chỉ 0.24.2 khớp LR Rad-LN 0.626938). Không nâng cấp scikit-learn, numpy hay torch trong môi trường tham chiếu.
7. OvO ở tổ hợp 0.798 (single-run seed 42) cho 0.768003 trong môi trường này, thấp hơn DyAM ở phân hoạch tự nhiên của tổ hợp đó. Đây là một lần chạy đơn, không phải kết luận về OvO (xem các so sánh nhiều phân hoạch).

## 6. Bằng chứng nguồn gốc

`document/2026-10-12_original-environment/` (`original-environment.md`, `sai-khac-can-dieu-tra.md` D11–D17): MD5 dữ liệu và code khớp Synapse; các môi trường đã thử và loại; tái hiện thành công trên Linux.
