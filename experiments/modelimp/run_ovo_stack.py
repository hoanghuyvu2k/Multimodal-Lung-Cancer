"""Mô hình đề xuất trên backbone OvO: OvO + NLP + miss_pattern (+ ensemble theo seed),
đánh giá trên các tổ hợp NHIỀU NGUỒN (k ≥ 5) — đúng phạm vi tiêu chí của người dùng.

    PYTHONIOENCODING=utf-8 python experiments/modelimp/run_ovo_stack.py
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
from sklearn.metrics import roc_auc_score                         # noqa: E402

from nlp_verify.nlp_run import build_l1                           # noqa: E402  (áp bản vá scipy)
from allcombo.run_ovo_singlerun import train_ovo_safe             # noqa: E402
from allcombo.run_paper_combos import PAPER_COMBOS                # noqa: E402
from common import benchmarks as B                                # noqa: E402
from common.data_setup import load_all                            # noqa: E402
from common.evaluate import paired_bootstrap, save_result         # noqa: E402
from missingness.run_miss import MISS, add_miss_modality          # noqa: E402
from nlp_verify.nlp_modality import add_nlp_modality              # noqa: E402
from lung_helpers import get_training_data                        # noqa: E402

SCORES = _CODE / "experiments" / "results" / "scores"
COMBOS = ["Rad+IHC-A+Gen", "Rad+IHC-G+Gen", "Rad+IHC-A+MutAmp", "Rad+IHC-A+Gen+PDL1",
          "Rad+IHC-G+Gen+PDL1", "Rad+IHC-A+Gen+TMB+PDL1"]


def main():
    torch.set_num_threads(2)
    ctx = load_all()
    add_nlp_modality(ctx)
    out = {}
    print(f"{'tổ hợp':<26}{'OvO+NLP+miss':>18}{'ensemble':>10}")
    for label in COMBOS:
        mods = next(m for lb, m, _f in PAPER_COMBOS if lb == label)
        add_miss_modality(ctx, mods)
        full = mods + ["cnl_nlp", MISS]
        data, mask, labels = get_training_data(full, ctx.modality_dict, ctx.modality_MASK,
                                               ctx.df_outcomes)
        l1 = build_l1(full, ctx)
        params = dict(B.MODEL_PARAMS)
        params["no_scale"] = [full.index("cnl_nlp"), full.index(MISS)]
        y = labels["label"].astype(float)
        cols, per = {}, []
        for seed in B.SEEDS:
            rng = np.random.default_rng(seed)
            summ, _ = train_ovo_safe(data, mask, labels.iloc[rng.permutation(len(labels))],
                                     l1, params, folds=B.FOLDS, seed=seed)
            s = pd.Series(summ["score"].astype(float).values, index=summ.index).dropna()
            cols[seed] = s
            per.append(float(roc_auc_score(y.loc[s.index], s)))
        sc = pd.DataFrame({f"seed_{k}": v for k, v in cols.items()})
        z = sc.apply(lambda c: (c - c.mean()) / c.std()).mean(1).dropna()
        out[label] = {"mean": float(np.mean(per)), "sd": float(np.std(per, ddof=1)),
                      "per_seed": per, "ensemble": float(roc_auc_score(y.loc[z.index], z))}
        slug = label.replace("+", "_").replace("-", "").replace(" ", "")
        sc.assign(label=y.loc[sc.index]).to_csv(SCORES / f"ovostack_{slug}__PAPER.csv")
        dy = pd.read_csv(SCORES / f"dyam_{slug}__PAPER.csv", index_col=0)
        d, lo, hi, pv = paired_bootstrap(
            {"_scores": sc, "_labels": y.loc[sc.index]},
            {"_scores": dy.drop(columns="label"), "_labels": dy["label"]})
        out[label]["vs_dyam"] = {"delta": d, "ci": [lo, hi], "p": pv}
        save_result("ovo_stack", out)
        v = out[label]["vs_dyam"]
        print(f"{label:<26}{out[label]['mean']:.4f}±{out[label]['sd']:.3f}".ljust(44)
              + f"{out[label]['ensemble']:>10.4f}   vs DyAM Δ={v['delta']:+.4f} "
              f"CI[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] p={v['p']:.3f}", flush=True)
    print("XONG")


if __name__ == "__main__":
    main()
