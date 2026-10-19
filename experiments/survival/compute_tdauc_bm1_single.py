"""tdAUC (Uno, 6/12/18 thang) cho 4 mo hinh BM1 o mot phan hoach. Chay bang venv_np2 (scikit-survival):
  C:/Users/Admin/venv_np2/Scripts/python.exe experiments/survival/compute_tdauc_bm1_single.py
Ra: experiments/results/ref/single/tdauc_bm1_single.json"""
import json
from pathlib import Path

import pandas as pd
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

ROOT = Path(__file__).resolve().parents[2]
CLIN = ROOT.parent / "datasets" / "18193MSKMINDProjectM-OmnibusInventory_DATA_2021-12-20_1540-WITH-TB-and-SCANNER.csv"
clin = pd.read_csv(CLIN).drop_duplicates("dmp_pt_id").set_index("dmp_pt_id")
D = ROOT / "experiments" / "results" / "ref" / "single"
res = {}
for m in ("dyam", "ovo", "dyam_nlp", "ovo_nlp"):
    sc = pd.read_csv(D / "scores" / f"bm1_{m}.csv", index_col=0)
    c = clin.reindex(sc.index)
    ok = c["pfs"].notna() & c["pfs_censor"].notna()
    y = Surv.from_arrays(event=c.loc[ok, "pfs_censor"].astype(bool).values, time=c.loc[ok, "pfs"].astype(float).values)
    auc, _ = cumulative_dynamic_auc(y, y, sc.loc[ok, "score"].values, [6, 12, 18])
    res[m] = {str(t): float(a) for t, a in zip([6, 12, 18], auc)}
    print(m, {k: round(v, 3) for k, v in res[m].items()}, flush=True)
(D / "tdauc_bm1_single.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print("XONG")
