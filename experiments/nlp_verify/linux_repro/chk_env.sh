#!/bin/bash
export MPLBACKEND=Agg
/mnt/d/envs/py38sa/python/bin/python3.8 -c "
import sys,numpy,scipy,sklearn,pandas,torch,lifelines,pyarrow,matplotlib,seaborn,statannotations
print('OK py',sys.version.split()[0],'torch',torch.__version__,'sklearn',sklearn.__version__,'numpy',numpy.__version__,'scipy',scipy.__version__,'pandas',pandas.__version__,'lifelines',lifelines.__version__,'matplotlib',matplotlib.__version__,'seaborn',seaborn.__version__)
import ctypes.util; print('BLAS', numpy.__config__.get_info('blas_opt_info').get('libraries'))
"
