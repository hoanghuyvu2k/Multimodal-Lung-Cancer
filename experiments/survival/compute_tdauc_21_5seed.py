"""tdAUC (Uno, 6/12/18 thang) tung seed cho 21 to hop 5 seed. venv_np2:
  C:/Users/Admin/venv_np2/Scripts/python.exe experiments/survival/compute_tdauc_21_5seed.py
Ra: experiments/results/ref/multi21/tdauc_21_5seed.json  ({"<idx>__<model>": {"6": [5 seed], "12": [...], "18": [...]}})"""
import glob
import json
from pathlib import Path

import pandas as pd
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

ROOT = Path(__file__).resolve().parents[2]
CLIN = ROOT.parent / "datasets" / "18193MSKMINDProjectM-OmnibusInventory_DATA_2021-12-20_1540-WITH-TB-and-SCANNER.csv"
clin = pd.read_csv(CLIN).drop_duplicates("dmp_pt_id").set_index("dmp_pt_id")
D = ROOT / "experiments" / "results" / "ref" / "multi21"
res = {}
for f in sorted(glob.glob(str(D / "scores" / "*.csv"))):
    sc = pd.read_csv(f, index_col=0)
    c = clin.reindex(sc.index)
    ok = c["pfs"].notna() & c["pfs_censor"].notna()
    y = Surv.from_arrays(event=c.loc[ok, "pfs_censor"].astype(bool).values, time=c.loc[ok, "pfs"].astype(float).values)
    out = {"6": [], "12": [], "18": []}
    for col in sc.columns:
        okc = ok & sc[col].notna()          # seed 42 de lai 21 benh nhan khong co diem o to hop 9, 10 (cac seed khac day du)
        yc = Surv.from_arrays(event=c.loc[okc, "pfs_censor"].astype(bool).values, time=c.loc[okc, "pfs"].astype(float).values)
        auc, _ = cumulative_dynamic_auc(yc, yc, sc.loc[okc, col].values, [6, 12, 18])
        for t, a in zip(("6", "12", "18"), auc):
            out[t].append(float(a))
    res[Path(f).stem] = out
(D / "tdauc_21_5seed.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(len(res), "XONG")
