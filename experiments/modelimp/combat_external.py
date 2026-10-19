"""Phép thử quyết định của hướng 1: ComBat có làm radiomics khái quát ra ngoài mẫu không?

Động cơ: radiomics quyết định hiệu năng NỘI BỘ nhưng ra ngoài mẫu thì gần như ngẫu nhiên
(AUC rad_valid 0.44-0.48). Nếu phần lớn tín hiệu nội bộ là hiệu ứng máy chụp/cohort thì hài hoà
hoá phải làm ngoại kiểm khá lên — kể cả khi AUC nội bộ giảm (đã đo: -0.023 BM1, -0.029 BM3).

Ba biến thể, cùng một quy trình huấn luyện (SelectKBest k=30 + LR, như run_validation.py):
    goc          không hài hoà
    combat_sc    ComBat theo MÁY CHỤP (chỉ áp được cho discovery; validation để nguyên)
    combat_coh   ComBat theo COHORT (discovery vs rad_valid) — hài hoà chung hai cohort

Lưu ý: combat_coh dùng cả ma trận đặc trưng của tập test khi hiệu chỉnh (không dùng nhãn).
Đây là cách làm phổ biến trong tài liệu radiomics nhưng là transductive — phải khai báo rõ.

    PYTHONIOENCODING=utf-8 python experiments/modelimp/combat_external.py
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

from common.data_setup import load_all                            # noqa: E402
from common.evaluate import save_result                           # noqa: E402
from modelimp.combat_rad import combat_frame, scanner_batch       # noqa: E402
from validation.run_validation import (                           # noqa: E402
    eval_pair, largest_lesion, numeric_features, read_any, select_k, valid_labels,
)

RAD = "lung_radiomics_spacing1.0_mirpon_window1350.250_allimagetypes_bw20.parquet"
RAD_VAL = ("lung_radiomics_spacing1.0_mirpon_window1350.250_allimagetypes_bw20"
           "_validation.parquet")


def prep():
    ctx = load_all()
    rad_lab, _ = valid_labels()
    disc = numeric_features(largest_lesion(read_any(RAD)))
    val = numeric_features(largest_lesion(read_any(RAD_VAL)))
    ytr = ctx.df_outcomes.reindex(disc.index)["label"]
    keep = ytr.notna()
    disc, ytr = disc[keep.values], ytr[keep].values
    common = val.index.intersection(rad_lab.index)
    val, yte = val.loc[common], rad_lab.loc[common].values
    cols = disc.columns.intersection(val.columns)
    return disc[cols], ytr, val[cols], yte


def run(Xtr, ytr, Xte, yte, name, out):
    sel_tr, sel_te = select_k(Xtr, ytr, Xte, k=30)
    out[name] = {"sel": eval_pair(sel_tr, ytr, sel_te, yte),
                 "all": eval_pair(Xtr, ytr, Xte, yte)}
    s, a = out[name]["sel"], out[name]["all"]
    g = lambda d: d.get("auc", d) if isinstance(d, dict) else d          # noqa: E731
    print(f"{name:<14} k=30: {g(s)}\n{'':<14} full: {g(a)}", flush=True)


def main():
    Xtr, ytr, Xte, yte = prep()
    print(f"discovery {Xtr.shape} · rad_valid {Xte.shape} · "
          f"đáp ứng ngoài mẫu {int((yte == 0).sum())}/{len(yte)}\n")
    out = {}

    run(Xtr, ytr, Xte, yte, "goc", out)

    sc = scanner_batch()
    tr_sc, n1 = combat_frame(Xtr, sc)
    print(f"   [ComBat máy chụp] {n1} đặc trưng")
    run(tr_sc, ytr, Xte, yte, "combat_scanner", out)

    both = pd.concat([Xtr, Xte])
    coh = pd.Series(["discovery"] * len(Xtr) + ["rad_valid"] * len(Xte), index=both.index)
    h, n2 = combat_frame(both, coh)
    print(f"   [ComBat cohort] {n2} đặc trưng")
    run(h.iloc[:len(Xtr)], ytr, h.iloc[len(Xtr):], yte, "combat_cohort", out)

    save_result("combat_external", out)
    print("\nXONG — experiments/results/combat_external.json")


if __name__ == "__main__":
    main()
