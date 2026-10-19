"""W2 — OvO + NLP + miss_pattern, 6 tổ hợp k>=5, 20 seed -> ovo_proof/nlp20miss_ovo__{slug}.csv.  python run_miss_ovo.py [idx ...]"""
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
from missingness.run_miss import MISS, add_miss_modality      # noqa: E402
from nlp_verify.nlp_modality import add_nlp_modality          # noqa: E402
from nlp_verify.nlp_run import build_l1                       # noqa: E402

if __name__ == "__main__":
    torch.set_num_threads(2)
    idx = [int(a) for a in sys.argv[1:]] or list(range(6))
    ctx = load_all(); add_nlp_modality(ctx)
    for i in idx:
        label = H.COMBOS[i]
        mods0 = next(m for lb, m, f in PAPER_COMBOS if lb == label)
        add_miss_modality(ctx, mods0)
        mods = mods0 + ["cnl_nlp", MISS]
        data, mask, labels = get_training_data(mods, ctx.modality_dict, ctx.modality_MASK, ctx.df_outcomes)
        l1 = build_l1(mods, ctx)
        params = {**B.MODEL_PARAMS, "no_scale": [mods.index("cnl_nlp"), mods.index(MISS)]}
        f = H.OUT / f"nlp20miss_ovo__{H.slug(label)}.csv"
        df = pd.read_csv(f, index_col=0) if f.exists() else pd.DataFrame({"label": labels["label"].astype(float)})
        for seed in H.SEEDS20:
            if f"seed_{seed}" in df.columns:
                continue
            rng = np.random.default_rng(seed)
            summ, _ = train_ovo_safe(data, mask, labels.iloc[rng.permutation(len(labels))], l1, params, folds=B.FOLDS, seed=seed)
            df[f"seed_{seed}"] = pd.Series(summ["score"].astype(float).values, index=summ.index)
            df.to_csv(f)
        print(f"[nlp20miss_ovo] {label}: {sum(c.startswith('seed_') for c in df.columns)} seed", flush=True)
    print("XONG")
