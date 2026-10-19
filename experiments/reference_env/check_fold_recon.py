"""Kiểm tra cách dựng fold của perfold_single_ref.py: chạy lại BM1 DyAM (seed 42, thứ tự tự nhiên), so cột 'fold' thật với fold dựng lại."""
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1])); sys.path.insert(0, str(_HERE.parents[2]))

from sklearn.model_selection import KFold                                    # noqa: E402
from common import benchmarks as B                                           # noqa: E402
from common.data_setup import load_all                                       # noqa: E402
from allcombo.run_paper_combos import PAPER_COMBOS, make_filter              # noqa: E402
from allcombo.run_combo import train as train_dyam                           # noqa: E402
from lung_helpers import get_training_data, auc_roc_ci                       # noqa: E402

ctx = load_all()
mods, ftype = next((m, f) for lb, m, f in PAPER_COMBOS if lb == "Rad+IHC-G+Gen+PDL1")
data, mask, labels = get_training_data(mods, ctx.modality_dict, ctx.modality_MASK, ctx.df_outcomes)
summ, _ = train_dyam(data, mask, labels, make_filter(ftype, ctx), dict(B.MODEL_PARAMS), folds=B.FOLDS, seed=42)
fold = {}
for k, (_, te) in enumerate(KFold(10, shuffle=True, random_state=0).split(labels.index)):
    for p in labels.index[te]:
        fold[p] = k
rec = np.array([fold[p] for p in summ.index])
print("columns", list(summ.columns)[:8])
print("AUC", auc_roc_ci(summ["label"].values, summ["score"].values, 0.95)[0])
print("fold column present:", "fold" in summ.columns)
if "fold" in summ.columns:
    print("reconstructed == real fold for all patients:", bool((rec == summ["fold"].values).all()))
