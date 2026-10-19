"""Kiểm tra code HIỆN TẠI của dự án (lung_helpers.py đã sửa + harness + OvO) trong môi trường Linux tham chiếu.
Hồi quy: DyAM Rad+IHC-A+Gen+PDL1 phải ra 0.798256 (= kết quả Linux của code gốc `main`)."""
import sys, os, time
os.environ.setdefault("ATTN_CACHE_DIR", "/mnt/d/repro_linux/cache")
sys.path.insert(0, "/mnt/d/code/master/doan/code/experiments")
sys.path.insert(0, "/mnt/d/code/master/doan/code")
import numpy as np, torch, sklearn, scipy, pandas as pd
print("ENV", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.cuda.is_available(), "sklearn", sklearn.__version__, "numpy", np.__version__, "scipy", scipy.__version__, "pandas", pd.__version__, flush=True)
import lung_helpers
print("lung_helpers:", lung_helpers.__file__, flush=True)
from sklearn.metrics import roc_auc_score
from common import benchmarks as B
from common.data_setup import load_all
ctx = load_all(use_cache=False)
RAD = ["rad_lesion_pc", "rad_lesion_pl", "rad_lesion_ln"]
mods = RAD + ["path_ihc_pdl1", "gen_driver_mut_amp", "cnl_pdl1_score"]
d, m, y = lung_helpers.get_training_data(mods, ctx.modality_dict, ctx.modality_MASK, ctx.df_outcomes)
def auc(s): return float(roc_auc_score(s["label"].astype(float), s["score"].astype(float)))
t = time.time()
s, _ = lung_helpers.train(d, m, y, ctx.rad_filters, dict(B.MODEL_PARAMS), folds=B.FOLDS, seed=42)
a = auc(s); print(f"SMOKE DyAM (code hien tai) Rad+IHC-A+Gen+PDL1 = {a:.6f} (muc tieu 0.798256; tac gia 0.798169) {'KHOP' if abs(a-0.798256)<5e-6 else 'LECH'} [{time.time()-t:.0f}s]", flush=True)
try:
    from allcombo.run_ovo_singlerun import train_ovo_safe
    s2, _ = train_ovo_safe(d, m, y, ctx.rad_filters, dict(B.MODEL_PARAMS), B.FOLDS, 42)
    print(f"SMOKE OvO (code hien tai) cung to hop = {auc(s2):.6f}", flush=True)
except Exception as e:
    print("SMOKE OvO LOI:", type(e).__name__, str(e)[:160], flush=True)
print("SMOKE_DONE")
