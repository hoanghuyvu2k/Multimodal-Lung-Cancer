"""Mô hình ĐẦY ĐỦ (uniform_avg + NLP lâm sàng + miss_pattern + C18) trên cả 21 tổ hợp của bài báo — theo
"Full evaluation rule" của CLAUDE.md. Harness chuẩn 5 seed × 10-fold. Labs thô (cnl_dem_labs) được thay bằng NLP,
nên #20/#21 trùng #17/#18.

    PYTHONIOENCODING=utf-8 python experiments/archloop/run_full21.py all --part=0/2
    PYTHONIOENCODING=utf-8 python experiments/archloop/run_full21.py nlp20        # BM1, chỉ +NLP, 20 seed

Kết quả: experiments/results/full21/{slug}.json (mỗi tổ hợp một file -> chạy song song/tiếp tục được),
         experiments/results/ovo_proof/uninlp__Rad_IHCG_Gen_PDL1.csv (nlp20).
"""
import json
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
from sklearn.metrics import roc_auc_score                         # noqa: E402

from archloop import harness as H                                 # noqa: E402  (áp bản vá scipy)
from archloop.cands import train_pdl1fill                         # noqa: E402
from allcombo.run_paper_combos import PAPER_COMBOS, make_filter   # noqa: E402
from baselines.model_uniform_avg import train_uniform_avg         # noqa: E402
from common import benchmarks as B                                # noqa: E402
from common.data_setup import load_all                            # noqa: E402
from missingness.run_miss import MISS, add_miss_modality          # noqa: E402
from nlp_verify.nlp_modality import add_nlp_modality              # noqa: E402
from lung_helpers import get_training_data                        # noqa: E402

OUT21 = _CODE / "experiments" / "results" / "full21"


def run_all(ctx, part):
    OUT21.mkdir(parents=True, exist_ok=True)
    i, n = (int(x) for x in part.split("/"))
    for k, (label, mods, ftype) in enumerate(PAPER_COMBOS):
        f = OUT21 / f"{H.slug(label)}.json"
        if k % n != i or f.exists():
            continue
        base = [m for m in mods if m != "cnl_dem_labs"]
        add_miss_modality(ctx, base)
        full = base + ["cnl_nlp", MISS]
        data, mask, labels = get_training_data(full, ctx.modality_dict, ctx.modality_MASK,
                                               ctx.df_outcomes)
        params = {**B.MODEL_PARAMS, "no_scale": [full.index("cnl_nlp"), full.index(MISS)]}
        y = labels["label"].astype(float)
        aucs, cols = [], {}
        for seed in B.SEEDS:
            rng = np.random.default_rng(seed)
            summ, _ = train_pdl1fill(data, mask, labels.iloc[rng.permutation(len(labels))],
                                     make_filter(ftype, ctx), params, folds=B.FOLDS, seed=seed)
            s = pd.Series(summ["score"].astype(float).values, index=summ.index)
            cols[seed] = s
            aucs.append(float(roc_auc_score(y.loc[s.index], s)))
        z = pd.DataFrame(cols)
        ens = ((z - z.mean()) / z.std()).mean(axis=1)
        res = {"label": label, "n": int(len(y)), "n_mod": len(base), "per_seed": aucs,
               "mean": float(np.mean(aucs)), "sd": float(np.std(aucs, ddof=1)),
               "ens5": float(roc_auc_score(y.loc[ens.index], ens))}
        f.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[full21] {label}: {res['mean']:.4f} ± {res['sd']:.4f}  ens5 {res['ens5']:.4f}  n={res['n']}",
              flush=True)
    print("XONG")


def run_nlp20(ctx):
    label = "Rad+IHC-G+Gen+PDL1"
    mods, ftype = next((m, f) for lb, m, f in PAPER_COMBOS if lb == label)
    full = mods + ["cnl_nlp"]
    data, mask, labels = get_training_data(full, ctx.modality_dict, ctx.modality_MASK,
                                           ctx.df_outcomes)
    params = {**B.MODEL_PARAMS, "no_scale": [full.index("cnl_nlp")]}
    f = H.OUT / f"uninlp__{H.slug(label)}.csv"
    df = (pd.read_csv(f, index_col=0) if f.exists()
          else pd.DataFrame({"label": labels["label"].astype(float)}))
    for seed in H.SEEDS20:
        if f"seed_{seed}" in df.columns:
            continue
        rng = np.random.default_rng(seed)
        summ, _ = train_uniform_avg(data, mask, labels.iloc[rng.permutation(len(labels))],
                                    make_filter(ftype, ctx), params, folds=B.FOLDS, seed=seed)
        df[f"seed_{seed}"] = pd.Series(summ["score"].astype(float).values, index=summ.index)
        df.to_csv(f)
    print("XONG nlp20")


if __name__ == "__main__":
    torch.set_num_threads(2)
    ctx = load_all()
    add_nlp_modality(ctx)
    if sys.argv[1] == "nlp20":
        run_nlp20(ctx)
    else:
        part = next((a[len("--part="):] for a in sys.argv if a.startswith("--part=")), "0/1")
        run_all(ctx, part)
