#!/bin/bash
# Chạy một script Python trong MÔI TRƯỜNG THAM CHIẾU (Linux WSL2, Python 3.8.15, torch 1.8.1, scikit-learn 0.24.2...).
# Gọi từ Windows (Git Bash), từ thư mục code/:
#   MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-22.04 -- bash /mnt/d/code/master/doan/code/experiments/reference_env/run_ref.sh experiments/<...>/script.py [args]
set -e
export MPLBACKEND=Agg PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export ATTN_CACHE_DIR=/mnt/d/repro_linux/cache           # cache dữ liệu đã xử lý (ổ D:)
export TMPDIR=/mnt/d/envs/tmp PIP_CACHE_DIR=/mnt/d/envs/pipcache PIP_NO_CACHE_DIR=1
cd /mnt/d/code/master/doan/code
export PYTHONPATH=/mnt/d/code/master/doan/code:/mnt/d/code/master/doan/code/experiments
exec /mnt/d/envs/py38sa/python/bin/python3.8 "$@"
