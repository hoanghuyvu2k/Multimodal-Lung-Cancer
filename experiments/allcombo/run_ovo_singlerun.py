"""OvO single-run (1-lần, thứ tự tự nhiên) cho BM1/BM2 — mảnh còn thiếu để so đỉnh 3 model.

train_ovo gốc không guard 0-feature -> sao y vòng fold + guard như safe DyAM, dùng MultiModalDynamicModelOvO.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lung_helpers import (                                      # noqa: E402
    MultiModalDynamicModelOvO, get_summary_df, get_training_data,
    l1_filter_features_list, set_global_seed,
)
from common import benchmarks as B                              # noqa: E402
from common.data_setup import load_all                          # noqa: E402

RAD = ["rad_lesion_pc", "rad_lesion_pl", "rad_lesion_ln"]
COMBOS = {
    "BM1": (RAD + ["path_ihc_glcm", "gen_driver_mut_amp", "cnl_pdl1_score"], True),
    "BM2": (RAD + ["path_ihc_glcm", "gen_driver_mut_amp", "cnl_pdl1_score", "cnl_dem_labs"], True),
}


def train_ovo_safe(modality_list_in, modality_mask, outcomes, l1_filter, params, folds, seed):
    set_global_seed(seed)
    p = {k: v for k, v in params.items() if k != "cross_modality_enabled"}
    d = {}
    kf = KFold(n_splits=folds, random_state=0, shuffle=True)
    for fold, (tr, te) in enumerate(kf.split(outcomes.index)):
        mlist = [df.copy(deep=True) for df in modality_list_in]
        train_px, valid_px = outcomes.index[tr], outcomes.index[te]
        for pos, filt in l1_filter.items():
            l1_filter_features_list(mlist, filt["l1_selection_df"], outcomes, valid_px, pos, **filt["kwargs"])
        if any(len(df.columns) == 0 for df in mlist):
            continue
        clf = MultiModalDynamicModelOvO(**p)
        clf.fit([df.loc[train_px].values for df in mlist],
                modality_mask.loc[train_px].astype(int).values,
                outcomes.loc[train_px, "label"])
        vs = clf.predict_proba([df.loc[valid_px].values for df in mlist],
                               modality_mask.loc[valid_px].astype(int).values)
        vy = outcomes.loc[valid_px, "label"].values
        for i, px in enumerate(valid_px):
            d[px] = {"label": vy[i], "score": vs[i], "fold": fold}
    return get_summary_df(d), None


def auc_of(summ):
    return float(roc_auc_score(summ["label"].astype(float).values, summ["score"].astype(float).values))


def main():
    ctx = load_all()
    print(f"\n{'=' * 70}\nOvO single-run (1-lần, tự nhiên) — mảnh còn thiếu\n{'=' * 70}")
    for bm, (mods, filt) in COMBOS.items():
        data, mask, labels = get_training_data(mods, ctx.modality_dict, ctx.modality_MASK, ctx.df_outcomes)
        l1 = ctx.rad_filters if filt else {}
        summ, _ = train_ovo_safe(data, mask, labels, l1, dict(B.MODEL_PARAMS), B.FOLDS, 42)
        print(f"  {bm}: OvO 1-run = {auc_of(summ):.4f}", flush=True)
    print("=" * 70)


if __name__ == "__main__":
    main()
