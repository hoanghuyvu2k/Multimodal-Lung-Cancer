"""Pha 2 — 6 tổ hợp k >= 5 x {DyAM, OvO} + NLP x 20 seed, lưu điểm từng bệnh nhân ở results/ovo_proof/nlp20_{model}__{slug}.csv.

    PYTHONIOENCODING=utf-8 python experiments/nlp_verify/run_nlp20.py dyam
    PYTHONIOENCODING=utf-8 python experiments/nlp_verify/run_nlp20.py ovo
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                        # noqa: E402
import pandas as pd                                       # noqa: E402
import torch                                              # noqa: E402

from common import benchmarks as B                        # noqa: E402
from common.data_setup import load_all                    # noqa: E402
from lung_helpers import get_training_data                # noqa: E402
from nlp_verify.nlp_modality import add_nlp_modality      # noqa: E402
from nlp_verify.nlp_run import build_l1                   # noqa: E402
from allcombo.run_combo import train as train_dyam        # noqa: E402
from allcombo.run_ovo_singlerun import train_ovo_safe     # noqa: E402
from allcombo.run_paper_combos import PAPER_COMBOS        # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "ovo_proof"
COMBOS = ["Rad+IHC-A+Gen", "Rad+IHC-G+Gen", "Rad+IHC-A+MutAmp", "Rad+IHC-A+Gen+PDL1", "Rad+IHC-G+Gen+PDL1", "Rad+IHC-A+Gen+TMB+PDL1"]
SEEDS = B.SEEDS + [11, 22, 33, 44, 55, 66, 77, 88, 99, 101, 202, 303, 404, 505, 606]
FN = {"dyam": train_dyam, "ovo": train_ovo_safe}


def slug(label):
    return label.replace("+", "_").replace("-", "").replace(" ", "")


def main():
    torch.set_num_threads(2)
    name = sys.argv[1]
    ctx = load_all()
    add_nlp_modality(ctx)
    for label in COMBOS:
        mods = next(m for lb, m, f in PAPER_COMBOS if lb == label) + ["cnl_nlp"]
        data, mask, labels = get_training_data(mods, ctx.modality_dict, ctx.modality_MASK, ctx.df_outcomes)
        l1 = build_l1(mods, ctx)
        params = {**B.MODEL_PARAMS, "no_scale": [mods.index("cnl_nlp")]}
        f = OUT / f"nlp20_{name}__{slug(label)}.csv"
        df = pd.read_csv(f, index_col=0) if f.exists() else pd.DataFrame({"label": labels["label"].astype(float)})
        for seed in SEEDS:
            if f"seed_{seed}" in df.columns:
                continue
            rng = np.random.default_rng(seed)
            summ, _ = FN[name](data, mask, labels.iloc[rng.permutation(len(labels))], l1, params, folds=B.FOLDS, seed=seed)
            df[f"seed_{seed}"] = pd.Series(summ["score"].astype(float).values, index=summ.index)
            df.to_csv(f)
        print(f"[nlp20_{name}] {label}: {sum(c.startswith('seed_') for c in df.columns)}/20 seed", flush=True)
    print("XONG")


if __name__ == "__main__":
    main()
