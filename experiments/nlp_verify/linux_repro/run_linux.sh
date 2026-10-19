#!/bin/bash
export MPLBACKEND=Agg OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONIOENCODING=utf-8
cd /mnt/d/repro_linux
/mnt/d/envs/py38sa/python/bin/python3.8 /mnt/d/repro_linux/run_cells_linux.py 1,2,3,4,5,6,7,8,9,10,11,12,13,16,17 /mnt/d/repro_linux/result_linux.json
