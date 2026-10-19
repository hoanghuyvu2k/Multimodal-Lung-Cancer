"""Con số chốt đo trên 20 seed, CÓ ĐỐI CHỨNG GHÉP CẶP: chồng đầy đủ có và không có tăng cường nhiễu loạn.

5 seed chuẩn hơi may ở BM1 (nền 0.7730 so với 0.7688 trên 20 seed), nên con số tiêu đề phải đo rộng hơn.

    PYTHONIOENCODING=utf-8 python experiments/modelimp/run_stack20.py
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
from scipy.stats import wilcoxon                                  # noqa: E402
from sklearn.metrics import roc_auc_score                         # noqa: E402

from modelimp import run_pertaug as P                             # noqa: E402
from modelimp.run_enscurve import SEEDS                           # noqa: E402
from allcombo.run_paper_combos import PAPER_COMBOS, make_filter   # noqa: E402
from common.data_setup import load_all                            # noqa: E402
from common.evaluate import save_result                           # noqa: E402
from missingness.run_miss import MISS, add_miss_modality          # noqa: E402
from nlp_verify.nlp_modality import add_nlp_modality              # noqa: E402
from lung_helpers import get_training_data                        # noqa: E402

LABEL = "Rad+IHC-G+Gen+PDL1"


def ens_curve(Z, y, ks=(5, 10, 20)):
    rng = np.random.default_rng(0)
    out = {}
    for k in ks:
        vals = []
        for _ in range(200 if k < Z.shape[1] else 1):
            sub = rng.choice(Z.columns, size=k, replace=False)
            m = Z[list(sub)].mean(1, skipna=True).dropna()
            vals.append(roc_auc_score(y.loc[m.index], m))
        out[k] = float(np.mean(vals))
    return out


def main():
    torch.set_num_threads(2)
    ctx = load_all()
    add_nlp_modality(ctx)
    pert = P.build_pert(ctx)
    mods, ftype = next((m, f) for lb, m, f in PAPER_COMBOS if lb == LABEL)
    add_miss_modality(ctx, mods)
    data, mask, labels = get_training_data(mods + ["cnl_nlp", MISS], ctx.modality_dict,
                                           ctx.modality_MASK, ctx.df_outcomes)
    l1 = make_filter(ftype, ctx)
    y = labels["label"].astype(float)
    ns = [i for i, c in enumerate(mask.columns) if c in ("cnl_nlp", MISS)]
    cols = {"aug": {}, "noaug": {}}
    for i, seed in enumerate(SEEDS, 1):
        lab = labels.iloc[np.random.default_rng(seed).permutation(len(labels))]
        cols["aug"][seed], _ = P.cv_once(data, mask, lab, l1, pert, seed, extra={"no_scale": ns})
        cols["noaug"][seed], _ = P.cv_once(data, mask, lab, l1, pert, seed, aug=False,
                                           extra={"no_scale": ns})
        a = roc_auc_score(y.loc[cols["aug"][seed].index], cols["aug"][seed])
        b = roc_auc_score(y.loc[cols["noaug"][seed].index], cols["noaug"][seed])
        print(f"seed {i:>2}/20 ({seed}): không aug {b:.4f} | aug {a:.4f} | Δ {a - b:+.4f}",
              flush=True)
    res = {}
    for name in ("noaug", "aug"):
        aucs = np.array([roc_auc_score(y.loc[s.index], s) for s in cols[name].values()])
        Z = pd.DataFrame(cols[name]).apply(lambda c: (c - c.mean()) / c.std())
        res[name] = {"mean": float(aucs.mean()), "sd": float(aucs.std(ddof=1)),
                     "per_seed": aucs.tolist(), "ens": ens_curve(Z, y)}
    d = np.array(res["aug"]["per_seed"]) - np.array(res["noaug"]["per_seed"])
    w = wilcoxon(d)
    res["paired"] = {"mean_delta": float(d.mean()), "sd_delta": float(d.std(ddof=1)),
                     "n_pos": int((d > 0).sum()), "wilcoxon_p": float(w.pvalue)}
    save_result("final_stack20", res)
    print(f"\nNLP+miss         : {res['noaug']['mean']:.4f} ± {res['noaug']['sd']:.3f} | "
          f"ensemble K5 {res['noaug']['ens'][5]:.4f} K10 {res['noaug']['ens'][10]:.4f} "
          f"K20 {res['noaug']['ens'][20]:.4f}")
    print(f"NLP+miss + aug   : {res['aug']['mean']:.4f} ± {res['aug']['sd']:.3f} | "
          f"ensemble K5 {res['aug']['ens'][5]:.4f} K10 {res['aug']['ens'][10]:.4f} "
          f"K20 {res['aug']['ens'][20]:.4f}")
    print(f"ghép cặp 20 seed : Δ = {d.mean():+.4f} ± {d.std(ddof=1):.4f}, "
          f"aug thắng {int((d > 0).sum())}/20, Wilcoxon p = {w.pvalue:.4f}")
    print("XONG")


if __name__ == "__main__":
    main()
