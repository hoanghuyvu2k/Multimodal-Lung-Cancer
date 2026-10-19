#!/bin/bash
# Chạy tuần tự các job run_paper_ref.py (mỗi job: các đối số trong ngoặc kép), log vào results/ref/<tên>.log
# dùng: run_chain.sh "surv" "order t2"
cd /mnt/d/code/master/doan/code
for j in "$@"; do
  n=$(echo "$j" | tr ' ' '_')
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash experiments/reference_env/run_ref.sh experiments/reference_env/run_paper_ref.py $j > experiments/results/ref/$n.log 2>&1
done
