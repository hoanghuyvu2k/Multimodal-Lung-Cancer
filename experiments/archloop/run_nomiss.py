"""Bỏ/giữ `miss_pattern` trên các tổ hợp k >= 5 (quy tắc ở ledger.md). Mô hình chốt = uniform + NLP + C18, với (arm=miss) hoặc không (arm=nomiss) miss_pattern.

    PYTHONIOENCODING=utf-8 python experiments/archloop/run_nomiss.py nomiss 0,1 [--seeds=20]
Chỉ số tổ hợp: 0 Rad+IHC-A+Gen · 1 Rad+IHC-G+Gen · 2 Rad+IHC-A+MutAmp · 3 Rad+IHC-A+Gen+PDL1 · 4 Rad+IHC-A+Gen+TMB+PDL1
"""
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
_CODE = _EXP.parent
for _p in (str(_CODE), str(_EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
import torch                                                      # noqa: E402

from archloop import harness as H                                 # noqa: E402  (áp bản vá scipy)
from archloop.cands import train_pdl1fill                         # noqa: E402
from allcombo.run_paper_combos import PAPER_COMBOS, make_filter   # noqa: E402
from common import benchmarks as B                                # noqa: E402
from common.data_setup import load_all                            # noqa: E402
from missingness.run_miss import MISS, add_miss_modality          # noqa: E402
from nlp_verify.nlp_modality import add_nlp_modality              # noqa: E402
from lung_helpers import get_training_data                        # noqa: E402

COMBOS = ["Rad+IHC-A+Gen", "Rad+IHC-G+Gen", "Rad+IHC-A+MutAmp", "Rad+IHC-A+Gen+PDL1", "Rad+IHC-A+Gen+TMB+PDL1"]


def main():
    torch.set_num_threads(2)
    arm = sys.argv[1]
    idx = [int(i) for i in sys.argv[2].split(",")]
    arg = {a.split("=")[0]: a.split("=")[1] for a in sys.argv[3:] if a.startswith("--")}
    seeds = H.SEEDS20[:int(arg.get("--seeds", 20))]
    ctx = load_all()
    add_nlp_modality(ctx)
    for i in idx:
        label = COMBOS[i]
        mods, ftype = next((m, f) for lb, m, f in PAPER_COMBOS if lb == label)
        add_miss_modality(ctx, mods)
        full = mods + ["cnl_nlp"] + ([MISS] if arm == "miss" else [])
        data, mask, labels = get_training_data(full, ctx.modality_dict, ctx.modality_MASK, ctx.df_outcomes)
        no_scale = [full.index("cnl_nlp")] + ([full.index(MISS)] if arm == "miss" else [])
        params = {**B.MODEL_PARAMS, "no_scale": no_scale}
        f = H.OUT / f"nm_{arm}__{H.slug(label)}.csv"
        df = pd.read_csv(f, index_col=0) if f.exists() else pd.DataFrame({"label": labels["label"].astype(float)})
        for seed in seeds:
            if f"seed_{seed}" in df.columns:
                continue
            rng = np.random.default_rng(seed)
            summ, _ = train_pdl1fill(data, mask, labels.iloc[rng.permutation(len(labels))], make_filter(ftype, ctx),
                                     params, folds=B.FOLDS, seed=seed)
            df[f"seed_{seed}"] = pd.Series(summ["score"].astype(float).values, index=summ.index)
            df.to_csv(f)
        print(f"[nm_{arm}] {label}: {sum(k.startswith('seed_') for k in df.columns)} seed", flush=True)
    print("XONG")


if __name__ == "__main__":
    main()
