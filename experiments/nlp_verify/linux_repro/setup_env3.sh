#!/bin/bash
# Python 3.8.15 (python-build-standalone) + gói 2021 trên Ổ D:, không dùng conda (NTFS không phân biệt hoa/thường làm hỏng gói ncurses).
set -e
export TMPDIR=/mnt/d/envs/tmp PIP_CACHE_DIR=/mnt/d/envs/pipcache PIP_NO_CACHE_DIR=1
SA=/mnt/d/envs/py38sa
if [ ! -x $SA/python/bin/python3.8 ]; then
  mkdir -p $SA
  curl -sL -o /mnt/d/envs/tmp/py38.tar.gz "https://github.com/astral-sh/python-build-standalone/releases/download/20221106/cpython-3.8.15%2B20221106-x86_64-unknown-linux-gnu-install_only.tar.gz"
  tar -xzf /mnt/d/envs/tmp/py38.tar.gz -C $SA && rm -f /mnt/d/envs/tmp/py38.tar.gz
fi
$SA/python/bin/python3.8 --version
V=$SA/python
PIP="$V/bin/python3.8 -m pip"
$PIP install -q --no-cache-dir -U "pip<24.1" "setuptools<66" wheel
$PIP install -q --no-cache-dir numpy==1.20.3 scipy==1.6.3 scikit-learn==0.24.2 pandas==1.2.5 pyarrow==4.0.1 tqdm requests joblib threadpoolctl matplotlib==3.4.2 seaborn==0.11.1 lifelines==0.26.0 statannotations==0.4.4 openpyxl
$PIP install -q --no-cache-dir torch==1.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
$V/bin/python3.8 -c "import sys,numpy,scipy,sklearn,pandas,torch,lifelines;print('OK',sys.version.split()[0],'torch',torch.__version__,'sklearn',sklearn.__version__,'numpy',numpy.__version__,'scipy',scipy.__version__,'pandas',pandas.__version__,'lifelines',lifelines.__version__)"
