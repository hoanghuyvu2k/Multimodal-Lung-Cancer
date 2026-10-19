"""D6b chồng lên bộ tốt nhất: uniform_avg + NLP lâm sàng + miss_pattern + C18 + D6b (nhiệm vụ phụ PD-L1 trên
chuyên gia pathology IHC-A). So với c18stack__{slug}.csv cùng seed (bộ tốt nhất KHÔNG có D6b).

Điểm lưu ở experiments/results/ovo_proof/d6bstack__{slug}.csv.

    PYTHONIOENCODING=utf-8 python experiments/archloop/run_stack_d6b.py [--only=0,1]
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
from archloop import cands as C                                   # noqa: E402
from allcombo.run_paper_combos import PAPER_COMBOS, make_filter   # noqa: E402
from common import benchmarks as B                                # noqa: E402
from common.data_setup import load_all                            # noqa: E402
from missingness.run_miss import MISS, add_miss_modality          # noqa: E402
from nlp_verify.nlp_modality import add_nlp_modality              # noqa: E402
from lung_helpers import get_training_data                        # noqa: E402

COMBOS = ["Rad+IHC-A+Gen+TMB+PDL1", "Rad+IHC-A+Gen+PDL1", "Rad+IHC-A+MutAmp"]


def train_stack_d6b(data, mask, labels, l1, params, folds=10, seed=42):
    cols = list(mask.columns)
    idx = next(i for i, c in enumerate(cols) if c in C.PATH_CH)
    p = {**params, "hid_idx": idx, "aux_lam": 0.5}
    return C._with("MultiModalDynamicModelUncertainty", C.ModelAux, C.train_pdl1fill,
                   data, mask, labels, l1, p, folds=folds, seed=seed)


def main():
    torch.set_num_threads(2)
    ctx = load_all()
    add_nlp_modality(ctx)
    only = [a[len("--only="):] for a in sys.argv if a.startswith("--only=")]
    combos = [COMBOS[int(i)] for i in only[0].split(",")] if only else COMBOS
    for label in combos:
        mods, ftype = next((m, f) for lb, m, f in PAPER_COMBOS if lb == label)
        add_miss_modality(ctx, mods)
        full = mods + ["cnl_nlp", MISS]
        data, mask, labels = get_training_data(full, ctx.modality_dict, ctx.modality_MASK,
                                               ctx.df_outcomes)
        l1 = make_filter(ftype, ctx)
        params = {**B.MODEL_PARAMS, "no_scale": [full.index("cnl_nlp"), full.index(MISS)]}
        f = H.OUT / f"d6bstack__{H.slug(label)}.csv"
        df = (pd.read_csv(f, index_col=0) if f.exists()
              else pd.DataFrame({"label": labels["label"].astype(float)}))
        for seed in H.SEEDS20:
            col = f"seed_{seed}"
            if col in df.columns:
                continue
            rng = np.random.default_rng(seed)
            summ, _ = train_stack_d6b(data, mask, labels.iloc[rng.permutation(len(labels))], l1,
                                      params, folds=B.FOLDS, seed=seed)
            df[col] = pd.Series(summ["score"].astype(float).values, index=summ.index)
            df.to_csv(f)
        n = sum(c.startswith("seed_") for c in df.columns)
        print(f"[d6bstack] {label}: {n} seed", flush=True)
    print("XONG")


if __name__ == "__main__":
    main()
