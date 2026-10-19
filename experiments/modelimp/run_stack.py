"""Chồng đầy đủ: NLP + miss_pattern + tăng cường nhiễu loạn radiomics + ensemble theo seed.

    PYTHONIOENCODING=utf-8 python experiments/modelimp/run_stack.py
"""
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
_CODE = _EXP.parent
for _p in (str(_CODE), str(_EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402

from modelimp import run_pertaug as P                             # noqa: E402
from allcombo.run_paper_combos import PAPER_COMBOS, make_filter   # noqa: E402
from common import benchmarks as B                                # noqa: E402
from common.data_setup import load_all                            # noqa: E402
from common.evaluate import save_result                           # noqa: E402
from missingness.run_miss import MISS, add_miss_modality          # noqa: E402
from nlp_verify.nlp_modality import add_nlp_modality              # noqa: E402
from lung_helpers import get_training_data                        # noqa: E402

REF = {"Rad+IHC-G+Gen+PDL1": (0.7896, 0.7970), "Rad+IHC-G+Gen": (0.7871, 0.7956)}   # NLP+miss, không aug


def main():
    torch.set_num_threads(2)
    ctx = load_all()
    add_nlp_modality(ctx)
    pert = P.build_pert(ctx)
    out = {}
    print(f"{'tổ hợp':<22}{'biến thể':<28}{'AUC':>15}{'ensemble':>10}")
    for label, (r_mean, r_ens) in REF.items():
        mods, ftype = next((m, f) for lb, m, f in PAPER_COMBOS if lb == label)
        add_miss_modality(ctx, mods)
        full = mods + ["cnl_nlp", MISS]
        data, mask, labels = get_training_data(full, ctx.modality_dict, ctx.modality_MASK,
                                               ctx.df_outcomes)
        l1 = make_filter(ftype, ctx)
        y = labels["label"].astype(float)
        ns = [i for i, c in enumerate(mask.columns) if c in ("cnl_nlp", MISS)]
        runs = {"NLP+miss+aug": {}, "NLP+miss+aug+TTA": {}}
        for seed in B.SEEDS:
            lab = labels.iloc[np.random.default_rng(seed).permutation(len(labels))]
            a, t = P.cv_once(data, mask, lab, l1, pert, seed, extra={"no_scale": ns})
            runs["NLP+miss+aug"][seed], runs["NLP+miss+aug+TTA"][seed] = a, t
        print(f"{label:<22}{'NLP+miss (không aug)':<28}{r_mean:>15.4f}{r_ens:>10.4f}")
        out[label] = {}
        for name, cols in runs.items():
            r = P.summarize(cols, y)
            out[label][name] = r
            save_result("final_stack", out)
            print(f"{'':<22}{name:<28}{r['mean']:.4f}±{r['sd']:.3f}{r['ensemble']:>10.4f}",
                  flush=True)
    print("XONG")


if __name__ == "__main__":
    main()
