"""AUC theo từng fold của MỘT phân hoạch (seed 42, thứ tự tự nhiên) từ điểm đã lưu — KHÔNG huấn luyện lại.

Dựng lại fold bằng đúng bộ chia của train(): KFold(10, shuffle=True, random_state=0) trên thứ tự bệnh nhân tự nhiên
(labels của get_training_data).     run_ref.sh experiments/reference_env/perfold_single_ref.py
Ra: experiments/results/ref/single/perfold_auc.json   {tên: {"folds": [10 AUC], "mean", "sd", "pooled"}}
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1])); sys.path.insert(0, str(_HERE.parents[2]))

from sklearn.metrics import roc_auc_score                                    # noqa: E402
from sklearn.model_selection import KFold                                    # noqa: E402

from common.data_setup import load_all                                       # noqa: E402
from allcombo.run_paper_combos import PAPER_COMBOS                           # noqa: E402
from lung_helpers import get_training_data                                   # noqa: E402

R = _HERE.parents[1] / "results" / "ref"


def main():
    ctx = load_all()
    mods = next(m for lb, m, f in PAPER_COMBOS if lb == "Rad+IHC-G+Gen+PDL1")
    _, _, labels = get_training_data(mods, ctx.modality_dict, ctx.modality_MASK, ctx.df_outcomes)
    fold = {}
    for k, (_, te) in enumerate(KFold(10, shuffle=True, random_state=0).split(labels.index)):
        for p in labels.index[te]:
            fold[p] = k
    out = {}
    files = sorted((R / "single" / "scores").glob("bm1_*.csv")) + sorted((R / "single21" / "scores").glob("*.csv"))
    for f in files:
        d = pd.read_csv(f, index_col=0)
        d["fold"] = [fold[p] for p in d.index]
        aucs = [roc_auc_score(g["label"], g["score"]) for _, g in d.groupby("fold")]
        key = ("bm1_" + f.stem[4:]) if f.parent.parent.name == "single" else f.stem
        out[key] = {"folds": [float(a) for a in aucs], "mean": float(np.mean(aucs)), "sd": float(np.std(aucs, ddof=1)),
                    "pooled": float(roc_auc_score(d["label"], d["score"])), "n": int(len(d))}
    (R / "single" / "perfold_auc.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k in [x for x in out if x.startswith("bm1_")] + ["18__dyam", "18__ovo", "21__dyam", "21__ovo", "21__dyam_nlp", "21__ovo_nlp", "17__dyam"]:
        if k in out:
            v = out[k]
            print(f"{k:16s} pooled {v['pooled']:.4f}  per-fold mean {v['mean']:.3f} ± {v['sd']:.3f}  n {v['n']}")
    print("files", len(out))


if __name__ == "__main__":
    main()
