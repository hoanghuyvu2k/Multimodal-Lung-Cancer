"""Chi so song con (C-index, Cox HR cua lung_helpers, KM chi2) cho 4 mo hinh BM1 (DyAM, OvO, DyAM+NLP, OvO+NLP) o MOT phan hoach
(seed 42, thu tu tu nhien), doc diem da luu ref/single/scores/bm1_*.csv. Chay trong ref-linux-py38:
  run_ref.sh experiments/survival/bm1_single_survival_ref.py
Ra: experiments/results/ref/single/bm1_survival_single.json (tdAUC them bang compute_tdauc_bm1_single.py)."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from nlp_verify import verify_nlp_5seed as V                       # noqa: E402
from lung_helpers import compute_cindex, compute_multivariate_cox  # noqa: E402

clin = V.load_clinical()
D = ROOT / "experiments" / "results" / "ref" / "single"
res = {}
for m in ("dyam", "ovo", "dyam_nlp", "ovo_nlp"):
    summ = pd.read_csv(D / "scores" / f"bm1_{m}.csv", index_col=0)
    hr = float(compute_multivariate_cox(summ, clin).summary.loc["score", "exp(coef)"])
    res[m] = {"cindex": float(compute_cindex(summ, clin)), "cox_hr": hr, "km_chi2": V.km_chi2(summ, clin)}
    print(m, res[m], flush=True)
(D / "bm1_survival_single.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print("XONG")
