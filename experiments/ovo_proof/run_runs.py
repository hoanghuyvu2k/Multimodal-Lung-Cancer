"""OvO có thực sự tốt hơn DyAM ở các tổ hợp nhiều nguồn? — phần CHẠY (20 seed, lưu điểm từng bệnh nhân).

Phạm vi cố định trước: 6 tổ hợp k ≥ 5 nguồn, so sánh THUẦN cơ chế hợp nhất (không NLP, không miss).
Ba mô hình trên CÙNG phân hoạch cho mỗi seed: DyAM · OvO · uniform_avg.
Lưu tăng dần từng (tổ hợp, mô hình, seed) -> chạy lại sẽ tiếp tục chỗ dở.

    PYTHONIOENCODING=utf-8 python experiments/ovo_proof/run_runs.py [dyam ovo uniform]
"""
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
_CODE = _EXP.parent
for _p in (str(_CODE), str(_EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sklearn.preprocessing._data as _skd                        # noqa: E402

_orig_yj = _skd.PowerTransformer._yeo_johnson_optimize


def _safe_yj(self, x):                                            # vá scipy >= 1.14
    try:
        return _orig_yj(self, x)
    except Exception:
        return 1.0


_skd.PowerTransformer._yeo_johnson_optimize = _safe_yj

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
import torch                                                      # noqa: E402

from allcombo.run_combo import train as train_dyam                # noqa: E402
from allcombo.run_ovo_singlerun import train_ovo_safe             # noqa: E402
from allcombo.run_paper_combos import PAPER_COMBOS, make_filter   # noqa: E402
from baselines.model_uniform_avg import train_uniform_avg         # noqa: E402
from common import benchmarks as B                                # noqa: E402
from common.data_setup import load_all                            # noqa: E402
from lung_helpers import get_training_data                        # noqa: E402

OUT = _CODE / "experiments" / "results" / "ovo_proof"
COMBOS = ["Rad+IHC-A+Gen", "Rad+IHC-G+Gen", "Rad+IHC-A+MutAmp", "Rad+IHC-A+Gen+PDL1",
          "Rad+IHC-G+Gen+PDL1", "Rad+IHC-A+Gen+TMB+PDL1"]
SEEDS = B.SEEDS + [11, 22, 33, 44, 55, 66, 77, 88, 99, 101, 202, 303, 404, 505, 606]   # 20
MODELS = {"dyam": train_dyam, "ovo": train_ovo_safe, "uniform": train_uniform_avg}


def slug(label):
    return label.replace("+", "_").replace("-", "").replace(" ", "")


def main(argv):
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    wanted = [m for m in argv[1:] if m in MODELS] or list(MODELS)
    ctx = load_all()
    for name in wanted:
        fn = MODELS[name]
        for label in COMBOS:
            mods, ftype = next((m, f) for lb, m, f in PAPER_COMBOS if lb == label)
            data, mask, labels = get_training_data(mods, ctx.modality_dict, ctx.modality_MASK,
                                                   ctx.df_outcomes)
            l1 = make_filter(ftype, ctx)
            f = OUT / f"{name}__{slug(label)}.csv"
            df = (pd.read_csv(f, index_col=0) if f.exists()
                  else pd.DataFrame({"label": labels["label"].astype(float)}))
            for seed in SEEDS:
                col = f"seed_{seed}"
                if col in df.columns:
                    continue
                rng = np.random.default_rng(seed)
                summ, _ = fn(data, mask, labels.iloc[rng.permutation(len(labels))], l1,
                             dict(B.MODEL_PARAMS), folds=B.FOLDS, seed=seed)
                df[col] = pd.Series(summ["score"].astype(float).values, index=summ.index)
                df.to_csv(f)
            n = sum(c.startswith("seed_") for c in df.columns)
            print(f"[{name}] {label}: {n}/20 seed", flush=True)
    print("XONG")


if __name__ == "__main__":
    main(sys.argv)
