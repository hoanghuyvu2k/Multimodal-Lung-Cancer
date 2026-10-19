"""W4 bước 2 (chạy trong venv numpy>=2): tdAUC tích luỹ/động (Uno) tại 6, 12, 18 tháng từ điểm ngoài-fold đã lưu.
    C:/Users/Admin/venv_np2/Scripts/python.exe experiments/survival/compute_tdauc.py
PFS tính bằng THÁNG như trong bảng lâm sàng (cột pfs), pfs_censor = 1 là sự kiện."""
import json
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

ROOT = Path(__file__).resolve().parents[2]
CLIN = ROOT.parent / "datasets" / "18193MSKMINDProjectM-OmnibusInventory_DATA_2021-12-20_1540-WITH-TB-and-SCANNER.csv"
COH = ROOT.parent / "datasets" / "final_cohort_listing.csv"
TIMES = [6, 12, 18]
BASE = ROOT / "experiments" / "results" / ("ref" if "--ref" in sys.argv else "")   # --ref: điểm sinh trong ref-linux-py38

coh = pd.read_csv(COH).set_index("main_index")
disc = coh[coh["cohort"] == "discovery"]
clin = pd.read_csv(CLIN)
key = "dmp_pt_id"
clin = clin.drop_duplicates(key).set_index(key)
res = {}
for f in sorted(glob.glob(str(BASE / "survival_scores" / "*.csv"))):
    name = Path(f).stem
    sc = pd.read_csv(f, index_col=0)
    c = clin.reindex(sc.index)
    ok = c["pfs"].notna() & c["pfs_censor"].notna()
    y = Surv.from_arrays(event=c.loc[ok, "pfs_censor"].astype(bool).values, time=c.loc[ok, "pfs"].astype(float).values)
    per = {t: [] for t in TIMES}
    for col in [x for x in sc.columns if x.startswith("seed_")]:
        s = sc.loc[ok, col]
        m = s.notna().values
        auc, _ = cumulative_dynamic_auc(y[m], y[m], s.values[m], TIMES)
        for t, a in zip(TIMES, auc):
            per[t].append(float(a))
    res[name] = {str(t): {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "per_seed": v} for t, v in per.items()}
    print(name, {t: round(res[name][str(t)]["mean"], 3) for t in TIMES}, flush=True)
(BASE / "tdauc_5seed.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print("XONG")
