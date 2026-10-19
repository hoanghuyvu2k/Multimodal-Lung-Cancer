"""H2 — độ bền của tín hiệu sẵn-có (kế hoạch ở ledger.md). BM1, mô hình chốt CÓ (A) hoặc KHÔNG (B) miss_pattern.
Ở tập kiểm tra mỗi bệnh nhân có pathology bị đặt "không có" với xác suất p (A: cập nhật cả bit has_path và n_available).

    PYTHONIOENCODING=utf-8 python experiments/archloop/run_shift.py verify
    PYTHONIOENCODING=utf-8 python experiments/archloop/run_shift.py A 0.25 [--seeds=20]
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

from archloop import harness as H                                 # noqa: E402  (áp bản vá scipy)
from archloop import cands as C                                   # noqa: E402
import baselines.model_uniform_avg as mua                         # noqa: E402
from allcombo.run_paper_combos import PAPER_COMBOS, make_filter   # noqa: E402
from common import benchmarks as B                                # noqa: E402
from common.data_setup import load_all                            # noqa: E402
from missingness.run_miss import MISS, add_miss_modality          # noqa: E402
from nlp_verify.nlp_modality import add_nlp_modality              # noqa: E402
from lung_helpers import get_training_data                        # noqa: E402

LABEL = "Rad+IHC-G+Gen+PDL1"
PATH = "path_ihc_glcm"


class ModelShift(mua.MultiModalDynamicModelUncertainty):
    def __init__(self, *a, pert_p=0.0, path_idx=None, miss_idx=None, has_col=None, nav_col=None, **kw):
        super().__init__(*a, **kw)
        self.pert_p, self.path_idx, self.miss_idx = float(pert_p), path_idx, miss_idx
        self.has_col, self.nav_col = has_col, nav_col

    def get_coefs(self, modality_list, modality_mask):
        return pd.DataFrame()

    def _forward_eval(self, l_X_INPUTS, arr_MASK):
        if self.pert_p > 0:
            X = [np.array(x.values if isinstance(x, pd.DataFrame) else x, dtype=float, copy=True) for x in l_X_INPUTS]
            mask = np.array(arr_MASK, copy=True)
            rng = np.random.default_rng(int(abs(np.nansum(X[self.path_idx])) * 1000) % (2 ** 32))
            cand = np.where(mask[:, self.path_idx] == 1)[0]
            drop = cand[rng.random(len(cand)) < self.pert_p]
            mask[drop, self.path_idx] = 0
            if self.miss_idx is not None:
                X[self.miss_idx][drop, self.has_col] = 0.0
                X[self.miss_idx][drop, self.nav_col] -= 1.0
            l_X_INPUTS, arr_MASK = X, mask
        return super()._forward_eval(l_X_INPUTS, arr_MASK)


def build(ctx, arm):
    mods, ftype = next((m, f) for lb, m, f in PAPER_COMBOS if lb == LABEL)
    add_miss_modality(ctx, mods)
    miss_cols = list(ctx.modality_dict[MISS].columns)
    full = mods + ["cnl_nlp"] + ([MISS] if arm == "A" else [])
    data, mask, labels = get_training_data(full, ctx.modality_dict, ctx.modality_MASK, ctx.df_outcomes)
    no_scale = [full.index("cnl_nlp")] + ([full.index(MISS)] if arm == "A" else [])
    info = dict(path_idx=full.index(PATH), miss_idx=full.index(MISS) if arm == "A" else None,
                has_col=miss_cols.index(f"has_{PATH}"), nav_col=miss_cols.index("n_available"))
    return data, mask, labels, make_filter(ftype, ctx), {**B.MODEL_PARAMS, "no_scale": no_scale, **info}


def one(data, mask, labels, l1, params, p, seed):
    rng = np.random.default_rng(seed)
    summ, _ = C._with("MultiModalDynamicModelUncertainty", ModelShift, C.train_pdl1fill, data, mask,
                      labels.iloc[rng.permutation(len(labels))], l1, {**params, "pert_p": p},
                      folds=B.FOLDS, seed=seed)
    return pd.Series(summ["score"].astype(float).values, index=summ.index)


def auc(y, s):
    m = s.notna()
    return roc_auc_score(y[m], s[m])


def main():
    torch.set_num_threads(2)
    arm = sys.argv[1]
    ctx = load_all()
    add_nlp_modality(ctx)
    if arm == "verify":
        for a, ref in (("A", "c18stack__" + H.slug(LABEL)), ("B", "loso_miss__" + H.slug(LABEL))):
            data, mask, labels, l1, params = build(ctx, a)
            s = one(data, mask, labels, l1, params, 0.0, 42)
            r = pd.read_csv(H.OUT / f"{ref}.csv", index_col=0)
            y = labels["label"].astype(float).reindex(s.index)
            print(f"VERIFY {a}: seed 42 = {auc(y, s):.4f} (tham chiếu {ref} = {auc(r['label'], r['seed_42']):.4f})", flush=True)
        return
    p = float(sys.argv[2])
    arg = {a.split("=")[0]: a.split("=")[1] for a in sys.argv[3:] if a.startswith("--")}
    seeds = H.SEEDS20[:int(arg.get("--seeds", 20))]
    data, mask, labels, l1, params = build(ctx, arm)
    f = H.OUT / f"shift_{arm}_p{int(p * 100)}__{H.slug(LABEL)}.csv"
    df = pd.read_csv(f, index_col=0) if f.exists() else pd.DataFrame({"label": labels["label"].astype(float)})
    for seed in seeds:
        if f"seed_{seed}" in df.columns:
            continue
        df[f"seed_{seed}"] = one(data, mask, labels, l1, params, p, seed)
        df.to_csv(f)
    print(f"[shift_{arm}_p{int(p * 100)}] {sum(k.startswith('seed_') for k in df.columns)} seed", flush=True)
    print("XONG")


if __name__ == "__main__":
    main()
