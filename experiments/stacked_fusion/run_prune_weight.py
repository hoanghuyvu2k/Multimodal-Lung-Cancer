"""Thử nghiệm: prune modality nhiễu (ý #2) + trung bình có trọng số độ-tin-cậy (ý #1), trên BM1.

Mục tiêu: có nhích được AUC nội bộ trên backbone uniform_avg (BM1 khoá 0.775, ens 0.785) không.
Không sửa lung_helpers.py. Prune dùng chính train_uniform_avg (backbone thật).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_CODE_DIR = Path(__file__).resolve().parents[2]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from lung_helpers import get_training_data                      # noqa: E402
from baselines.model_uniform_avg import train_uniform_avg       # noqa: E402
from common import benchmarks as B                              # noqa: E402
from common.data_setup import load_all                          # noqa: E402
from common.evaluate import (                                   # noqa: E402
    load_result, load_scores, paired_bootstrap, save_result,
)
from stacked_fusion.model_stack import _modality_pred, _select_features  # noqa: E402
from sklearn.model_selection import KFold                       # noqa: E402
from lung_helpers import get_summary_df, set_global_seed        # noqa: E402

# map tên radiomics -> filter tương ứng (rad_filters khoá theo vị trí pc=0/pl=1/ln=2)
RAD_NAMES = ["rad_lesion_pc", "rad_lesion_pl", "rad_lesion_ln"]


def build_l1(modality_list, ctx):
    """Dựng l1_filter theo vị trí modality trong list (chỉ radiomics)."""
    name2filt = {RAD_NAMES[i]: ctx.rad_filters[i] for i in range(3)}
    return {modality_list.index(n): name2filt[n] for n in modality_list if n in name2filt}


def run_uniform(modality_list, ctx, seeds):
    data, mask, labels = get_training_data(
        modality_list, ctx.modality_dict, ctx.modality_MASK, ctx.df_outcomes)
    l1 = build_l1(modality_list, ctx)
    params = dict(B.MODEL_PARAMS)
    cols = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        shuffled = labels.iloc[rng.permutation(len(labels))]
        summ, _ = train_uniform_avg(data, mask, shuffled, l1, params, folds=B.FOLDS, seed=seed)
        cols[seed] = pd.Series(summ["score"].astype(float).values, index=summ.index)
        y = pd.Series(summ["label"].astype(float).values, index=summ.index)
    sc = pd.DataFrame(cols)
    per_seed = [roc_auc_score(y.loc[sc.index], sc[s]) for s in seeds]
    ens = roc_auc_score(y.loc[sc.index], sc.mean(axis=1))
    return {"mean": float(np.mean(per_seed)), "sd": float(np.std(per_seed, ddof=1)),
            "ens": float(ens), "_scores": sc, "_labels": y.loc[sc.index]}


def train_lr_weighted(modality_list, mask_df, outcomes, l1, params, folds, seed, weights):
    """LR-late numpy: score = Σ wᵢ · zᵢ(modality). weights=None -> đều."""
    set_global_seed(seed)
    names = list(mask_df.columns)
    y = outcomes["label"]
    w = np.array([weights.get(n, 1.0) for n in names]) if weights else np.ones(len(names))
    d = {}
    kf = KFold(n_splits=folds, random_state=0, shuffle=True)
    for fold, (tr, te) in enumerate(kf.split(outcomes.index)):
        train_px, valid_px = outcomes.index[tr], outcomes.index[te]
        mlist = _select_features(modality_list, l1, outcomes, train_px)
        if any(len(df.columns) == 0 for df in mlist):
            continue
        val = np.zeros((len(valid_px), len(names)))
        for m in range(len(names)):
            val[:, m] = _modality_pred(mlist[m], names[m], mask_df, y, train_px, valid_px)
        scores = val @ w
        yv = y.loc[valid_px].values
        for i, px in enumerate(valid_px):
            d[px] = {"label": yv[i], "score": scores[i], "fold": fold}
    return get_summary_df(d), None


def run_lrw(modality_list, ctx, seeds, weights):
    data, mask, labels = get_training_data(
        modality_list, ctx.modality_dict, ctx.modality_MASK, ctx.df_outcomes)
    l1 = build_l1(modality_list, ctx)
    cols = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        shuffled = labels.iloc[rng.permutation(len(labels))]
        summ, _ = train_lr_weighted(data, mask, shuffled, l1, {}, B.FOLDS, seed, weights)
        cols[seed] = pd.Series(summ["score"].astype(float).values, index=summ.index)
        y = pd.Series(summ["label"].astype(float).values, index=summ.index)
    sc = pd.DataFrame(cols)
    per_seed = [roc_auc_score(y.loc[sc.index], sc[s]) for s in seeds]
    ens = roc_auc_score(y.loc[sc.index], sc.mean(axis=1))
    return {"mean": float(np.mean(per_seed)), "sd": float(np.std(per_seed, ddof=1)),
            "ens": float(ens), "_scores": sc, "_labels": y.loc[sc.index]}


def main():
    ctx = load_all()
    seeds = B.SEEDS
    audit = load_result("signal_audit")["uniform_avg"]
    ref = load_scores("uniform_avg", "BM1")  # backbone khoá

    full = B.BENCHMARKS["BM1"]["modalities"]           # pc,pl,ln,path_ihc_glcm,gen,pdl1
    prune_cfgs = {
        "full (ref)": full,
        "drop_pl": [m for m in full if m != "rad_lesion_pl"],
        "drop_pl_ln": [m for m in full if m not in ("rad_lesion_pl", "rad_lesion_ln")],
        "drop_all_rad": [m for m in full if not m.startswith("rad_")],
    }

    print(f"\n{'=' * 82}\nÝ #2 — PRUNE modality nhiễu (backbone uniform_avg, BM1)\n{'=' * 82}")
    print(f"{'config':<16}{'#mod':>6}{'mean±sd':>16}{'ens':>9}{'Δmean vs full':>15}{'p vs ref':>10}")
    payload = {"prune": {}, "weight": {}}
    base = None
    for name, mods in prune_cfgs.items():
        r = run_uniform(mods, ctx, seeds)
        if base is None:
            base = r["mean"]
        _, _, _, p = paired_bootstrap(r, ref)
        payload["prune"][name] = {k: r[k] for k in ("mean", "sd", "ens")}
        ms = f"{r['mean']:.4f}±{r['sd']:.4f}"
        print(f"{name:<16}{len(mods):>6}{ms:>16}"
              f"{r['ens']:>9.4f}{r['mean'] - base:>+15.4f}{p:>10.3f}")

    # Ý #1 — trọng số độ tin cậy trên full BM1 (LR-late)
    w_auc = {n: max(0.0, audit[n]["mean_auc"] - 0.5) for n in full}
    print(f"\n{'=' * 82}\nÝ #1 — LR-late: trọng số ĐỀU vs trọng số theo AUC (full BM1)\n{'=' * 82}")
    wstr = ", ".join(f"{n.split('_')[-1]}:{v:.2f}" for n, v in w_auc.items())
    print(f"  weights_auc = {{{wstr}}}")
    print(f"{'variant':<20}{'mean±sd':>16}{'ens':>9}")
    for name, w in [("lr_late uniform", None), ("lr_late auc-weighted", w_auc)]:
        r = run_lrw(full, ctx, seeds, w)
        payload["weight"][name] = {k: r[k] for k in ("mean", "sd", "ens")}
        ms = f"{r['mean']:.4f}±{r['sd']:.4f}"
        print(f"{name:<20}{ms:>16}{r['ens']:>9.4f}")

    save_result("prune_weight", payload)
    print(f"\n{'=' * 82}")
    print("Đối chiếu: BM1 uniform_avg khoá = 0.7746 (mean) / 0.7850 (ens).")
    print("=" * 82)


if __name__ == "__main__":
    main()
