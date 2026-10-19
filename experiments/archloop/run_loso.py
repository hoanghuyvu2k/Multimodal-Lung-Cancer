"""R3 — bỏ từng nguồn một (leave-one-source-out) ở BM1 trên mô hình chốt `c18stack`
(uniform + NLP + miss_pattern + C18; 20 seed). `miss_pattern` được ĐÓNG BĂNG từ danh sách nguồn ĐẦY ĐỦ, để "bỏ pathology" chỉ bỏ NỘI DUNG
chứ không bỏ luôn bit có-mặt (tránh lẫn nội dung với tín hiệu quy trình chăm sóc). Hai cánh `-path+avail`, `-pdl1+avail` bỏ cả bit sẵn-có.

    PYTHONIOENCODING=utf-8 python experiments/archloop/run_loso.py rad path gen pdl1 nlp miss path_avail pdl1_avail
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

LABEL = "Rad+IHC-G+Gen+PDL1"
RAD = ["rad_lesion_pc", "rad_lesion_pl", "rad_lesion_ln"]
PATH, GEN, PDL = "path_ihc_glcm", "gen_driver_mut_amp", "cnl_pdl1_score"
DROP = {"rad": RAD, "path": [PATH], "gen": [GEN], "pdl1": [PDL], "nlp": ["cnl_nlp"], "miss": [MISS],
        "path_avail": [PATH], "pdl1_avail": [PDL]}


def main():
    torch.set_num_threads(2)
    arms = sys.argv[1:]
    ctx = load_all()
    add_nlp_modality(ctx)
    mods, ftype = next((m, f) for lb, m, f in PAPER_COMBOS if lb == LABEL)
    add_miss_modality(ctx, mods)                                  # đóng băng từ danh sách ĐẦY ĐỦ
    miss_full = ctx.modality_dict[MISS].copy()
    for arm in arms:
        drop = DROP[arm]
        full = [m for m in mods + ["cnl_nlp", MISS] if m not in drop]
        if arm.endswith("_avail"):                                # bỏ cả bit có-mặt của nguồn đó
            ctx.modality_dict[MISS] = miss_full.drop(columns=[f"has_{drop[0]}"])
            ctx.modality_dict[MISS]["n_available"] = miss_full["n_available"] - ctx.modality_MASK[drop[0]].astype(float).reindex(miss_full.index).fillna(0.0)
        else:
            ctx.modality_dict[MISS] = miss_full
        data, mask, labels = get_training_data(full, ctx.modality_dict, ctx.modality_MASK, ctx.df_outcomes)
        l1 = {} if arm == "rad" else make_filter(ftype, ctx)      # radiomics đứng đầu danh sách, vị trí 0–2 không đổi khi bỏ nguồn khác
        no_scale = [full.index(x) for x in ("cnl_nlp", MISS) if x in full]
        params = {**B.MODEL_PARAMS, "no_scale": no_scale} if no_scale else dict(B.MODEL_PARAMS)
        f = H.OUT / f"loso_{arm}__{H.slug(LABEL)}.csv"
        df = (pd.read_csv(f, index_col=0) if f.exists()
              else pd.DataFrame({"label": labels["label"].astype(float)}))
        for seed in H.SEEDS20:
            if f"seed_{seed}" in df.columns:
                continue
            rng = np.random.default_rng(seed)
            summ, _ = train_pdl1fill(data, mask, labels.iloc[rng.permutation(len(labels))], l1, params,
                                     folds=B.FOLDS, seed=seed)
            df[f"seed_{seed}"] = pd.Series(summ["score"].astype(float).values, index=summ.index)
            df.to_csv(f)
        print(f"[loso_{arm}] {sum(k.startswith('seed_') for k in df.columns)} seed; nguồn: {full}", flush=True)
    print("XONG")


if __name__ == "__main__":
    main()
