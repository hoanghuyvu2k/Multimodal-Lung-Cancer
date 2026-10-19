"""W1 — bỏ từng nguồn ở BM1, OvO+NLP, 20 seed.  python experiments/ovo_final/run_loso_ovo.py rad path gen pdl1 nlp"""
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXP)); sys.path.insert(0, str(_EXP.parent))

import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402
import torch                                                  # noqa: E402

from archloop import harness as H                             # noqa: E402
from allcombo.run_paper_combos import PAPER_COMBOS             # noqa: E402
from allcombo.run_ovo_singlerun import train_ovo_safe          # noqa: E402
from common import benchmarks as B                            # noqa: E402
from common.data_setup import load_all                        # noqa: E402
from lung_helpers import get_training_data                    # noqa: E402
from nlp_verify.nlp_modality import add_nlp_modality          # noqa: E402
from nlp_verify.nlp_run import build_l1                       # noqa: E402

LABEL = "Rad+IHC-G+Gen+PDL1"
RAD = ["rad_lesion_pc", "rad_lesion_pl", "rad_lesion_ln"]
DROP = {"rad": RAD, "path": ["path_ihc_glcm"], "gen": ["gen_driver_mut_amp"], "pdl1": ["cnl_pdl1_score"], "nlp": ["cnl_nlp"]}

if __name__ == "__main__":
    torch.set_num_threads(2)
    ctx = load_all(); add_nlp_modality(ctx)
    base = next(m for lb, m, f in PAPER_COMBOS if lb == LABEL) + ["cnl_nlp"]
    for arm in sys.argv[1:]:
        mods = [m for m in base if m not in DROP[arm]]
        data, mask, labels = get_training_data(mods, ctx.modality_dict, ctx.modality_MASK, ctx.df_outcomes)
        l1 = build_l1(mods, ctx)
        params = {**B.MODEL_PARAMS, "no_scale": [mods.index("cnl_nlp")] if "cnl_nlp" in mods else []}
        f = H.OUT / f"loso_ovo_{arm}.csv"
        df = pd.read_csv(f, index_col=0) if f.exists() else pd.DataFrame({"label": labels["label"].astype(float)})
        for seed in H.SEEDS20:
            if f"seed_{seed}" in df.columns:
                continue
            rng = np.random.default_rng(seed)
            summ, _ = train_ovo_safe(data, mask, labels.iloc[rng.permutation(len(labels))], l1, params, folds=B.FOLDS, seed=seed)
            df[f"seed_{seed}"] = pd.Series(summ["score"].astype(float).values, index=summ.index)
            df.to_csv(f)
        print(f"[loso_ovo_{arm}] {sum(c.startswith('seed_') for c in df.columns)} seed: {mods}", flush=True)
    print("XONG")
