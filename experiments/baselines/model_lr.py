"""
Step 6 — Baseline trung thực bằng hồi quy logistic.

Hai biến thể, dùng ĐÚNG cấu trúc fold và L1 filter như các mô hình neural, để so sánh
trực tiếp được:

  `lr_concat`    : ghép toàn bộ feature của mọi modality thành một ma trận, một LR duy nhất.
  `lr_late`      : mỗi modality một LR riêng, rồi trung bình logit trên các modality có sẵn.
                   Đây là bản LR tương ứng của `uniform_avg` — thay head neural bằng LR.

Xử lý modality thiếu: sau khi scale, feature của modality vắng mặt bị đặt về 0 — tương đương
cách mask triệt tiêu đóng góp trong mô hình neural.

Chuẩn hoá logit trong `lr_late`: logit của mỗi modality được z-score theo phân bố trên tập
train trước khi lấy trung bình. Nếu không làm vậy, modality có logit thang lớn sẽ lấn át —
vai trò này trong mô hình neural do `tanh` đảm nhiệm.
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
    get_summary_df,
    l1_filter_features_list,
    set_global_seed,
)

LR_KWARGS = dict(penalty="l2", C=1.0, class_weight="balanced",
                 max_iter=2000, solver="lbfgs")


def _fold_data(modality_list_in, modality_mask, outcomes, l1_dfs_filter, folds):
    """Sinh ra từng fold với đúng phân hoạch và L1 filter như train()."""
    kf = KFold(n_splits=folds, random_state=0, shuffle=True)
    for fold, (tr, te) in enumerate(tqdm(list(kf.split(outcomes.index)), file=sys.stdout)):
        modality_list = [df.copy(deep=True) for df in modality_list_in]
        train_px, valid_px = outcomes.index[tr], outcomes.index[te]

        for pos, filt in l1_dfs_filter.items():
            l1_filter_features_list(modality_list, filt["l1_selection_df"], outcomes,
                                    valid_px, pos, **filt["kwargs"])
        if any(len(df.columns) == 0 for df in modality_list):
            continue

        yield (fold, modality_list, train_px, valid_px,
               modality_mask.loc[train_px].astype(int).values,
               modality_mask.loc[valid_px].astype(int).values,
               outcomes.loc[train_px, "label"].values,
               outcomes.loc[valid_px, "label"].values)


def _scaled(df, train_px, valid_px):
    scaler = RobustScaler()
    tr = np.nan_to_num(scaler.fit_transform(df.loc[train_px].values))
    va = np.nan_to_num(scaler.transform(df.loc[valid_px].values))
    return tr, va


def train_lr_concat(modality_list_in, modality_mask, outcomes, l1_dfs_filter,
                    model_params, folds=10, seed=42):
    set_global_seed(seed)
    d_summarys_all = {}

    for (fold, mlist, train_px, valid_px, tr_mask, va_mask,
         tr_y, va_y) in _fold_data(modality_list_in, modality_mask, outcomes,
                                   l1_dfs_filter, folds):
        tr_blocks, va_blocks = [], []
        for i, df in enumerate(mlist):
            tr, va = _scaled(df, train_px, valid_px)
            # Modality vắng mặt -> toàn 0, giống hệt tác dụng của mask trong mô hình neural
            tr_blocks.append(tr * tr_mask[:, [i]])
            va_blocks.append(va * va_mask[:, [i]])

        clf = LogisticRegression(**LR_KWARGS)
        clf.fit(np.hstack(tr_blocks), tr_y)
        scores = clf.decision_function(np.hstack(va_blocks))

        for idx, px in enumerate(valid_px):
            d_summarys_all[px] = {"label": va_y[idx], "score": scores[idx], "fold": fold}

    return get_summary_df(d_summarys_all), pd.DataFrame()


def train_lr_late(modality_list_in, modality_mask, outcomes, l1_dfs_filter,
                  model_params, folds=10, seed=42):
    set_global_seed(seed)
    d_summarys_all = {}

    for (fold, mlist, train_px, valid_px, tr_mask, va_mask,
         tr_y, va_y) in _fold_data(modality_list_in, modality_mask, outcomes,
                                   l1_dfs_filter, folds):
        n_mod = len(mlist)
        z_valid = np.zeros((len(valid_px), n_mod))

        for i, df in enumerate(mlist):
            tr, va = _scaled(df, train_px, valid_px)
            present = tr_mask[:, i] == 1
            # Cần cả hai lớp trong tập train của modality này mới fit được
            if present.sum() < 10 or len(np.unique(tr_y[present])) < 2:
                continue

            clf = LogisticRegression(**LR_KWARGS)
            clf.fit(tr[present], tr_y[present])

            z_tr = clf.decision_function(tr[present])
            mu, sd = z_tr.mean(), z_tr.std()
            sd = sd if sd > 1e-8 else 1.0
            z_valid[:, i] = (clf.decision_function(va) - mu) / sd

        # Trung bình trên các modality có sẵn — đúng dạng của uniform_avg
        denom = np.clip(va_mask.sum(axis=1), 1, None)
        scores = (z_valid * va_mask).sum(axis=1) / denom

        for idx, px in enumerate(valid_px):
            d_summarys_all[px] = {"label": va_y[idx], "score": scores[idx], "fold": fold}

    return get_summary_df(d_summarys_all), pd.DataFrame()
