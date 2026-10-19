"""Số đặc trưng radiomics giữ lại sau chọn lọc theo fold (bảng Supp. 'Feature counts'), trong môi trường tham chiếu.

Cùng cấu hình như train(): ctx.rad_filters (l1_strength 0.1, robustness 0.15, outlier 6), KFold(10, shuffle, random_state=0),
bộ lọc fit trên phần huấn luyện (loại fold giữ-ngoài), tổ hợp BM1 (Rad+IHC-G+Gen+PDL1).
    run_ref.sh experiments/reference_env/count_rad_features_ref.py
Kết quả: experiments/results/ref/rad_feature_counts.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1])); sys.path.insert(0, str(_HERE.parents[2]))

from common import benchmarks as B                                    # noqa: E402,F401
from common.data_setup import load_all                                # noqa: E402
from lung_helpers import get_training_data, select_radiomics_features_elastic   # noqa: E402
from allcombo.run_paper_combos import PAPER_COMBOS                    # noqa: E402

SITES = {"rad_lesion_pc": 0, "rad_lesion_pl": 1, "rad_lesion_ln": 2}

if __name__ == "__main__":
    ctx = load_all()
    mods = next(m for lb, m, f in PAPER_COMBOS if lb == "Rad+IHC-G+Gen+PDL1")
    data, mask, labels = get_training_data(mods, ctx.modality_dict, ctx.modality_MASK, ctx.df_outcomes)
    out = {}
    for name, pos in SITES.items():
        i = mods.index(name)
        filt = ctx.rad_filters[pos]
        raw = data[i].shape[1]
        counts = []
        for tr, te in KFold(10, shuffle=True, random_state=0).split(labels.index):
            feats, _ = select_radiomics_features_elastic(filt["l1_selection_df"].drop(index=labels.index[te], errors="ignore"),
                                                         labels, **filt["kwargs"])
            counts.append(len(feats))
        out[name] = {"raw_columns_in_modality": int(raw), "n_patients": int(mask[name].sum()), "per_fold": counts,
                     "median": float(np.median(counts)), "min": int(min(counts)), "max": int(max(counts))}
        print(name, out[name], flush=True)
    (_HERE.parents[1] / "results" / "ref" / "rad_feature_counts.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("XONG")
