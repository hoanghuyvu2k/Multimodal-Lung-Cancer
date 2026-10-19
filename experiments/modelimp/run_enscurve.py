"""Đường cong kích thước ensemble: AUC theo số mô hình K (subagging qua các phân hoạch CV).

Ensemble theo seed là thay đổi DUY NHẤT ở tầng mô hình từng thắng (+0.009 với K=5). Câu hỏi: K lớn
hơn có cho thêm không? Mỗi seed = một phân hoạch 10-fold khác, nên với mỗi bệnh nhân, mỗi seed cho
đúng một mô hình CHƯA từng thấy bệnh nhân đó -> trung bình K điểm là ước lượng out-of-bag hợp lệ.

    PYTHONIOENCODING=utf-8 python experiments/modelimp/run_enscurve.py
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
from sklearn.metrics import roc_auc_score                         # noqa: E402

from modelimp import run_all as R                                 # noqa: E402  (vá scipy)
from baselines.model_uniform_avg import train_uniform_avg         # noqa: E402
from common import benchmarks as B                                # noqa: E402
from common.data_setup import load_all                            # noqa: E402
from common.evaluate import save_result                           # noqa: E402

SEEDS = B.SEEDS + [11, 22, 33, 44, 55, 66, 77, 88, 99, 101, 202, 303, 404, 505, 606]   # 20
KS = [1, 3, 5, 10, 15, 20]


def main():
    ctx = load_all()
    out = {}
    for label in ("Rad+IHC-G+Gen+PDL1", "Rad+IHC-G+Gen", "Rad+Gen"):
        data, mask, labels, l1 = R.combo(ctx, label)
        y = labels["label"].astype(float)
        cols = {}
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            summ, _ = train_uniform_avg(data, mask, labels.iloc[rng.permutation(len(labels))],
                                        l1, dict(B.MODEL_PARAMS), folds=B.FOLDS, seed=seed)
            cols[seed] = pd.Series(summ["score"].astype(float).values, index=summ.index)
        Z = pd.DataFrame(cols).apply(lambda c: (c - c.mean()) / c.std())
        single = [roc_auc_score(y.loc[Z[s].dropna().index], Z[s].dropna()) for s in SEEDS]
        rng = np.random.default_rng(0)
        row = {"single_mean": float(np.mean(single)), "single_sd": float(np.std(single, ddof=1))}
        line = f"{label:<22} 1 mô hình: {np.mean(single):.4f}±{np.std(single, ddof=1):.3f} |"
        for k in KS[1:]:
            vals = []
            for _ in range(200 if k < len(SEEDS) else 1):          # trung bình trên các tập con cỡ k
                sub = rng.choice(SEEDS, size=k, replace=False)
                m = Z[list(sub)].mean(1, skipna=True).dropna()
                vals.append(roc_auc_score(y.loc[m.index], m))
            row[f"K{k}"] = float(np.mean(vals)); row[f"K{k}_sd"] = float(np.std(vals))
            line += f" K={k}: {np.mean(vals):.4f}"
        out[label] = row
        save_result("ensemble_curve", out)
        print(line, flush=True)
    print("XONG")


if __name__ == "__main__":
    import torch
    torch.set_num_threads(2)
    main()
