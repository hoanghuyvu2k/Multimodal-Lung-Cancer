"""Chạy ĐÚNG 21 tổ hợp nguồn của bài báo gốc (Figures-Finalized.ipynb cell 18), so uniform_avg vs DyAM.

Bài báo phân biệt IHC-A(path_ihc_pdl1) vs IHC-G(path_ihc_glcm), TMB vs Gen(mut_amp) vs non_tmb, có Rad-LU,
và đơn modality — mịn hơn allcombo (5 domain) nên allcombo chỉ phủ 7/21. Đây chạy đủ 21 tổ hợp giống hệt
danh sách modality của notebook, 5-seed 10-fold, cùng harness để so sánh nhất quán.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lung_helpers import get_training_data                      # noqa: E402
from baselines.model_uniform_avg import train_uniform_avg       # noqa: E402
from common import benchmarks as B                              # noqa: E402
from common.data_setup import load_all                          # noqa: E402
from common.evaluate import paired_bootstrap, save_result, save_scores  # noqa: E402
from allcombo.run_combo import train as train_dyam              # noqa: E402  (DyAM có guard 0-feature)

RAD = ["rad_lesion_pc", "rad_lesion_pl", "rad_lesion_ln"]

# (label, modalities, filter_type). filter: none | rad | rad_lu. Modality Rad đặt đầu để filter khớp vị trí.
PAPER_COMBOS = [
    ("TMB",                       ["gen_driver_tmb"],                          "none"),
    ("PDL1",                      ["cnl_pdl1_score"],                          "none"),
    ("IHC-A",                     ["path_ihc_pdl1"],                           "none"),
    ("Gen",                       ["gen_driver_mut_amp"],                      "none"),
    ("Rad",                       RAD,                                         "rad"),
    ("Rad-LU",                    RAD + ["rad_lesion_lu"],                     "rad_lu"),
    ("TMB+PDL1",                  ["gen_driver_tmb", "cnl_pdl1_score"],        "none"),
    ("PDL1+Gen",                  ["cnl_pdl1_score", "gen_driver_mut_amp"],    "none"),
    ("Rad+IHC-A",                 RAD + ["path_ihc_pdl1"],                     "rad"),
    ("Rad+IHC-G",                 RAD + ["path_ihc_glcm"],                     "rad"),
    ("Rad+Gen",                   RAD + ["gen_driver_mut_amp"],                "rad"),
    ("IHC-A+Gen",                 ["path_ihc_pdl1", "gen_driver_mut_amp"],     "none"),
    ("IHC-G+Gen",                 ["path_ihc_glcm", "gen_driver_mut_amp"],     "none"),
    ("Rad+IHC-A+Gen",            RAD + ["path_ihc_pdl1", "gen_driver_mut_amp"], "rad"),
    ("Rad+IHC-G+Gen",            RAD + ["path_ihc_glcm", "gen_driver_mut_amp"], "rad"),
    ("Rad+IHC-A+MutAmp",         RAD + ["path_ihc_pdl1", "gen_driver_non_tmb", "cnl_pdl1_score"], "rad"),
    ("Rad+IHC-A+Gen+PDL1",       RAD + ["path_ihc_pdl1", "gen_driver_mut_amp", "cnl_pdl1_score"], "rad"),
    ("Rad+IHC-G+Gen+PDL1",       RAD + ["path_ihc_glcm", "gen_driver_mut_amp", "cnl_pdl1_score"], "rad"),
    ("Rad+IHC-A+Gen+TMB+PDL1",   RAD + ["path_ihc_pdl1", "gen_driver_non_tmb", "gen_driver_tmb", "cnl_pdl1_score"], "rad"),
    ("Rad+IHC-A+Gen+PDL1+Labs",  RAD + ["path_ihc_pdl1", "gen_driver_mut_amp", "cnl_pdl1_score", "cnl_dem_labs"], "rad"),
    ("Rad+IHC-G+Gen+PDL1+Labs",  RAD + ["path_ihc_glcm", "gen_driver_mut_amp", "cnl_pdl1_score", "cnl_dem_labs"], "rad"),
]


def make_filter(ftype, ctx):
    if ftype == "rad":
        return ctx.rad_filters             # {0:pc,1:pl,2:ln}
    if ftype == "rad_lu":
        return ctx.rad_filters_lu          # {0:pc,1:pl,2:ln,3:lu}
    return {}


def run_model(train_fn, modalities, l1_filter, ctx, seeds):
    data, mask, labels = get_training_data(modalities, ctx.modality_dict, ctx.modality_MASK, ctx.df_outcomes)
    params = dict(B.MODEL_PARAMS)
    y_full = labels["label"].astype(float)   # nhãn đầy đủ (patient->label, bất biến qua seed)
    cols = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        shuffled = labels.iloc[rng.permutation(len(labels))]
        summ, _ = train_fn(data, mask, shuffled, l1_filter, params, folds=B.FOLDS, seed=seed)
        cols[f"seed_{seed}"] = pd.Series(summ["score"].astype(float).values, index=summ.index)
    scores = pd.DataFrame(cols)                       # union index; NaN nơi seed bỏ fold 0-feature
    ylab = y_full.loc[scores.index]                   # nhãn cho mọi bệnh nhân từng được chấm
    per_seed = []
    for c in scores:
        col = scores[c].dropna()                      # chỉ bệnh nhân seed này chấm
        per_seed.append(roc_auc_score(ylab.loc[col.index], col))
    return {"mean_auc": float(np.mean(per_seed)), "sd_auc": float(np.std(per_seed, ddof=1)),
            "per_seed_auc": [float(x) for x in per_seed],
            "n": int(len(ylab)), "_scores": scores, "_labels": ylab}


def slug(label):
    return label.replace("+", "_").replace("-", "").replace(" ", "")


def main():
    ctx = load_all()
    payload = {"combos": {}}
    print(f"\n{'=' * 96}\n21 TỔ HỢP BÀI BÁO — uniform_avg vs DyAM (5 seed x 10-fold)\n{'=' * 96}")
    print(f"{'#':>3} {'combo':<28}{'#mod':>5}{'n':>5}{'uniform':>11}{'DyAM':>11}{'Δ(u-d)':>10}{'p':>8}{'win':>6}")
    for i, (label, mods, ftype) in enumerate(PAPER_COMBOS, 1):
        l1 = make_filter(ftype, ctx)
        ru = run_model(train_uniform_avg, mods, l1, ctx, B.SEEDS)
        rd = run_model(train_dyam, mods, l1, ctx, B.SEEDS)
        delta, lo, hi, p = paired_bootstrap(ru, rd)
        save_scores(f"uni_{slug(label)}", "PAPER", ru)
        save_scores(f"dyam_{slug(label)}", "PAPER", rd)
        win = "" if p >= 0.05 else ("UNI" if delta > 0 else "DYAM")
        payload["combos"][label] = {
            "modalities": mods, "n_mod": len(mods), "n": ru["n"], "filter": ftype,
            "uniform": {k: ru[k] for k in ("mean_auc", "sd_auc", "per_seed_auc")},
            "dyam": {k: rd[k] for k in ("mean_auc", "sd_auc", "per_seed_auc")},
            "delta_mean": ru["mean_auc"] - rd["mean_auc"],
            "paired": {"delta_auc": delta, "ci_low": lo, "ci_high": hi, "p_value": p},
        }
        save_result("paper_combos", payload)
        print(f"{i:>3} {label:<28}{len(mods):>5}{ru['n']:>5}"
              f"{ru['mean_auc']:>11.4f}{rd['mean_auc']:>11.4f}"
              f"{ru['mean_auc'] - rd['mean_auc']:>+10.4f}{p:>8.3f}{win:>6}", flush=True)
    print("=" * 96)


if __name__ == "__main__":
    main()
