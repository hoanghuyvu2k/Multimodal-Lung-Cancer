"""Mô hình mới — Stacked Late-Fusion (thay attention).

Mỗi modality một base learner LR -> dự đoán OUT-OF-FOLD (inner CV) -> meta-LR học trọng số modality
từ OOF. Vì meta học trên OOF (không in-fold), nó hạ trọng số modality không generalize (radiomics OOF
kém nhất quán) và tin modality ổn định. Trọng số meta là diễn giải trung thực thay attention.

Chống rò rỉ:
  - L1 filter feature radiomics fit trên OUTER-TRAIN (không nhìn outer-val). (Chọn feature ổn định nên
    fit 1 lần/outer-fold thay vì mỗi inner-fold — đánh đổi tốc độ, rò rỉ không đáng kể.)
  - Base-pred OOF cho outer-train sinh bằng INNER 5-fold: mỗi inner base LR chỉ thấy inner-train.
  - Base refit trên full outer-train mới chạm outer-val.
  - Modality vắng mặt -> base-pred = 0 (sau z-score theo train), meta-LR có intercept nên xử lý được.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

_CODE_DIR = Path(__file__).resolve().parents[2]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from lung_helpers import (  # noqa: E402
    get_summary_df, l1_filter_features_list, set_global_seed,
)

BASE_LR = dict(penalty="l2", C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs")
META_LR = dict(penalty="l2", C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs")


def _select_features(modality_list_in, l1_dfs_filter, outcomes, fit_px):
    """L1-select feature radiomics fit trên fit_px (exclude phần còn lại)."""
    mlist = [df.copy(deep=True) for df in modality_list_in]
    exclude = outcomes.index.difference(fit_px)
    for pos, filt in l1_dfs_filter.items():
        l1_filter_features_list(mlist, filt["l1_selection_df"], outcomes, exclude, pos, **filt["kwargs"])
    return mlist


def _modality_pred(df, mask_col, mmask, y, fit_px, target_px):
    """Fit base LR trên hàng CÓ modality của fit_px; trả logit z-score cho target_px (vắng mặt -> 0)."""
    scaler = RobustScaler()
    A_fit = np.nan_to_num(scaler.fit_transform(df.loc[fit_px].values))
    yfit = y.loc[fit_px].values
    pf = mmask.loc[fit_px, mask_col].values.astype(bool)
    out = np.zeros(len(target_px))
    if pf.sum() < 10 or len(np.unique(yfit[pf])) < 2:
        return out
    clf = LogisticRegression(**BASE_LR).fit(A_fit[pf], yfit[pf])
    z_fit = clf.decision_function(A_fit[pf])
    mu, sd = z_fit.mean(), z_fit.std()
    sd = sd if sd > 1e-8 else 1.0
    A_t = np.nan_to_num(scaler.transform(df.loc[target_px].values))
    z = (clf.decision_function(A_t) - mu) / sd
    pt = mmask.loc[target_px, mask_col].values.astype(bool)
    z[~pt] = 0.0
    return z


def train_stack(modality_list_in, modality_mask, outcomes, l1_dfs_filter,
                model_params, folds=10, seed=42):
    set_global_seed(seed)
    d_summarys_all = {}
    n_mod = len(modality_list_in)
    names = list(modality_mask.columns)
    coef_rows = []
    y = outcomes["label"]

    outer = KFold(n_splits=folds, random_state=0, shuffle=True)
    for fold, (tr, te) in enumerate(tqdm(list(outer.split(outcomes.index)), file=sys.stdout)):
        train_px, valid_px = outcomes.index[tr], outcomes.index[te]
        mlist = _select_features(modality_list_in, l1_dfs_filter, outcomes, train_px)
        if any(len(df.columns) == 0 for df in mlist):
            continue

        # OOF base-preds trên outer-train qua inner 5-fold
        tr_arr = np.asarray(train_px)
        oof = np.zeros((len(train_px), n_mod))
        inner = KFold(n_splits=5, random_state=0, shuffle=True)
        for itr, iva in inner.split(tr_arr):
            i_tr, i_va = pd.Index(tr_arr[itr]), pd.Index(tr_arr[iva])
            for m in range(n_mod):
                oof[iva, m] = _modality_pred(mlist[m], names[m], modality_mask, y, i_tr, i_va)

        # base-preds cho outer-val (refit trên full outer-train)
        val = np.zeros((len(valid_px), n_mod))
        for m in range(n_mod):
            val[:, m] = _modality_pred(mlist[m], names[m], modality_mask, y, train_px, valid_px)

        # meta-LR học trọng số modality từ OOF
        meta = LogisticRegression(**META_LR).fit(oof, y.loc[train_px].values)
        scores = meta.decision_function(val)

        row = {names[m]: float(meta.coef_[0][m]) for m in range(n_mod)}
        row["intercept"] = float(meta.intercept_[0])
        row["fold"] = fold
        coef_rows.append(row)

        y_te = y.loc[valid_px].values
        for idx, px in enumerate(valid_px):
            d_summarys_all[px] = {"label": y_te[idx], "score": scores[idx], "fold": fold}

    return get_summary_df(d_summarys_all), pd.DataFrame(coef_rows)
