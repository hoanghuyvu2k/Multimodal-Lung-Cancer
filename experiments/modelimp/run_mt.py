"""Đánh giá học đa nhiệm đáp ứng + sống còn (model_mt.py).

    PYTHONIOENCODING=utf-8 python experiments/modelimp/run_mt.py [--smoke]
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
from lifelines.utils import concordance_index                     # noqa: E402
from sklearn.metrics import roc_auc_score                         # noqa: E402

from modelimp import run_all as R                                 # noqa: E402  (vá scipy)
from modelimp import model_mt as MT                               # noqa: E402
from common import benchmarks as B                                # noqa: E402
from common import data_setup as ds                               # noqa: E402
from common.evaluate import load_result, save_result              # noqa: E402
from lung_helpers import get_clinical_table_v2                    # noqa: E402

COMBOS = ["Rad+IHC-G+Gen+PDL1", "Rad+IHC-G+Gen", "Rad+IHC-A+Gen+PDL1", "Rad+Gen",
          "PDL1+Gen", "IHC-G+Gen"]
VARIANTS = {"+Cox": dict(lam_cox=1.0), "+Ord": dict(lam_ord=1.0),
            "+Cox+Ord": dict(lam_cox=1.0, lam_ord=1.0)}
SCORES = _CODE / "experiments" / "results" / "scores"


def clinical():
    coh = pd.read_csv(f"{ds.BASE_DB_DIR}/final_cohort_listing.csv").set_index("main_index")
    return get_clinical_table_v2(
        path=f"{ds.BASE_DB_DIR}/18193MSKMINDProjectM-OmnibusInventory_DATA_2021-12-20_1540"
             f"-WITH-TB-and-SCANNER.csv",
        main_index_col="dmp_pt_id", cohort=coh[coh["cohort"] == "discovery"])


def metrics(cols, y, clin):
    """AUC + C-index từng seed, và AUC/C-index của ensemble (trung bình z-score qua seed)."""
    aucs, cis = [], []
    for s in cols.values():
        aucs.append(roc_auc_score(y.loc[s.index], s))
        c = clin.loc[s.index]
        cis.append(concordance_index(c["pfs"], -s.values, c["pfs_censor"]))
    z = pd.DataFrame(cols).apply(lambda c: (c - c.mean()) / c.std()).mean(1, skipna=True)
    z = z.dropna(); c = clin.loc[z.index]
    return {"auc": float(np.mean(aucs)), "auc_sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
            "cindex": float(np.mean(cis)), "auc_ens": float(roc_auc_score(y.loc[z.index], z)),
            "cindex_ens": float(concordance_index(c["pfs"], -z.values, c["pfs_censor"])),
            "auc_per_seed": [float(a) for a in aucs]}


def run(ctx, clin, label, seeds, **kw):
    data, mask, labels, l1 = R.combo(ctx, label)
    y = labels["label"].astype(float)
    cols = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        summ, _ = MT.train_mt(data, mask, labels.iloc[rng.permutation(len(labels))], l1,
                              dict(B.MODEL_PARAMS), folds=B.FOLDS, seed=seed, **kw)
        cols[seed] = pd.Series(summ["score"].astype(float).values, index=summ.index).dropna()
    return metrics(cols, y, clin)


def baseline(label, y, clin):
    slug = label.replace("+", "_").replace("-", "").replace(" ", "")
    df = pd.read_csv(SCORES / f"uni_{slug}__PAPER.csv", index_col=0)
    cols = {c: df[c].dropna() for c in df.columns if c.startswith("seed_")}
    return metrics(cols, y, clin)


def main(argv):
    smoke = "--smoke" in argv
    ctx = ds.load_all()
    clin = clinical()
    MT.AUX["df"] = clin[["pfs", "pfs_censor", "bor"]]
    combos = COMBOS[:1] if smoke else COMBOS
    seeds = B.SEEDS[:1] if smoke else B.SEEDS
    variants = ({"λ=0 (phải = uniform_avg)": {}, **VARIANTS} if smoke else VARIANTS)
    try:
        payload = load_result("multitask")
    except Exception:
        payload = {"combos": {}}
    print(f"{'tổ hợp':<22}{'biến thể':<26}{'AUC':>15}{'Δ':>9}{'AUC ens':>9}{'Δ':>9}"
          f"{'C-index':>9}{'Δ':>9}")
    for label in combos:
        _d, _m, labels, _l = R.combo(ctx, label)
        y = labels["label"].astype(float)
        b = baseline(label, y, clin)
        print(f"{label:<22}{'uniform_avg':<26}{b['auc']:>15.4f}{0:>+9.4f}{b['auc_ens']:>9.4f}"
              f"{0:>+9.4f}{b['cindex']:>9.4f}{0:>+9.4f}")
        if not smoke:
            payload["combos"].setdefault(label, {})["base"] = b
        for name, kw in variants.items():
            r = run(ctx, clin, label, seeds, **kw)
            if not smoke:
                payload["combos"][label][name] = r
                save_result("multitask", payload)
            print(f"{'':<22}{name:<26}{r['auc']:.4f}±{r['auc_sd']:.3f}{r['auc'] - b['auc']:>+9.4f}"
                  f"{r['auc_ens']:>9.4f}{r['auc_ens'] - b['auc_ens']:>+9.4f}"
                  f"{r['cindex']:>9.4f}{r['cindex'] - b['cindex']:>+9.4f}", flush=True)
    print("XONG")


if __name__ == "__main__":
    import torch
    torch.set_num_threads(2)
    main(sys.argv)
