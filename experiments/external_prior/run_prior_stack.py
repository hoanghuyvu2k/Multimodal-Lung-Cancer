"""V2 — thêm điểm TIÊN NGHIỆM (mô hình PD-L1 + genomics huấn luyện CHỈ trên dữ liệu ngoài) làm một modality cố định vào bộ chốt c18stack.
Kế hoạch ở ledger.md. Mốc: c18stack__{slug}.csv cùng seed.

    PYTHONIOENCODING=utf-8 python experiments/external_prior/run_prior_stack.py --only=0 [--seeds=5]
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

COMBOS = ["Rad+IHC-G+Gen+PDL1", "Rad+IHC-A+Gen+TMB+PDL1", "Rad+IHC-A+MutAmp", "Rad+IHC-A+Gen+PDL1"]
PRIOR = "ext_prior"


def main():
    torch.set_num_threads(2)
    arg = {a.split("=")[0]: a.split("=")[1] for a in sys.argv[1:] if a.startswith("--")}
    seeds = H.SEEDS20[:int(arg.get("--seeds", 5))]
    combos = [COMBOS[int(i)] for i in arg["--only"].split(",")] if "--only" in arg else COMBOS[:3]
    ctx = load_all()
    add_nlp_modality(ctx)
    ps = pd.read_csv(_CODE / "experiments" / "results" / "external_prior_scores.csv", index_col=0).iloc[:, 0]
    ctx.modality_dict[PRIOR] = ps.reindex(ctx.df_outcomes.index).to_frame("prior")
    ctx.modality_MASK[PRIOR] = ctx.modality_dict[PRIOR]["prior"].notna().values
    print(f"{PRIOR}: {int(ctx.modality_MASK[PRIOR].sum())}/{len(ctx.df_outcomes)} bệnh nhân có điểm", flush=True)
    for label in combos:
        mods, ftype = next((m, f) for lb, m, f in PAPER_COMBOS if lb == label)
        add_miss_modality(ctx, mods)
        full = mods + ["cnl_nlp", MISS, PRIOR]
        data, mask, labels = get_training_data(full, ctx.modality_dict, ctx.modality_MASK, ctx.df_outcomes)
        params = {**B.MODEL_PARAMS, "no_scale": [full.index("cnl_nlp"), full.index(MISS)]}
        f = H.OUT / f"priorstack__{H.slug(label)}.csv"
        df = pd.read_csv(f, index_col=0) if f.exists() else pd.DataFrame({"label": labels["label"].astype(float)})
        for seed in seeds:
            if f"seed_{seed}" in df.columns:
                continue
            rng = np.random.default_rng(seed)
            summ, _ = train_pdl1fill(data, mask, labels.iloc[rng.permutation(len(labels))], make_filter(ftype, ctx),
                                     params, folds=B.FOLDS, seed=seed)
            df[f"seed_{seed}"] = pd.Series(summ["score"].astype(float).values, index=summ.index)
            df.to_csv(f)
        print(f"[priorstack] {label}: {sum(k.startswith('seed_') for k in df.columns)} seed", flush=True)
    print("XONG")


if __name__ == "__main__":
    main()
