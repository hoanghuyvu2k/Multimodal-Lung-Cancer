"""Audit: bài báo chạy train() 1 LẦN, thứ tự tự nhiên (không hoán vị nhãn). Tôi trung bình 5 seed hoán vị.
Tái lập đúng cấu hình bài báo (train gốc, model_params y hệt, không permute) để xem có ra ~0.79 không,
đồng thời in phổ 5-seed để thấy 1-lần chỉ là 1 mẫu trong phân bố.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lung_helpers import get_training_data                      # noqa: E402
from lung_helpers import train as train_paper                   # noqa: E402  (train gốc, không guard)
from baselines.model_uniform_avg import train_uniform_avg       # noqa: E402
from common import benchmarks as B                              # noqa: E402
from common.data_setup import load_all                          # noqa: E402
from allcombo.run_combo import train as train_dyam_safe         # noqa: E402

RAD = ["rad_lesion_pc", "rad_lesion_pl", "rad_lesion_ln"]
COMBOS = {
    "BM1 Rad+IHC-G+Gen+PDL1": (RAD + ["path_ihc_glcm", "gen_driver_mut_amp", "cnl_pdl1_score"], True, 0.7839),
    "BM3 Rad+Gen":            (RAD + ["gen_driver_mut_amp"], True, 0.7384),
    "BM4 PDL1+Gen":           (["cnl_pdl1_score", "gen_driver_mut_amp"], False, 0.6931),
}


def auc_of(summ):
    return roc_auc_score(summ["label"].astype(float).values, summ["score"].astype(float).values)


def main():
    ctx = load_all()
    print(f"\n{'=' * 96}")
    print("AUDIT — 1-lần không hoán vị (như bài báo) vs 5-seed hoán vị (của tôi). model_params y hệt.")
    print(f"{'=' * 96}")
    print(f"{'combo':<26}{'model':<10}{'paper_ref':>10}{'1-run(nat)':>12}{'5seed mean±sd':>18}{'5seed span':>16}")

    for name, (mods, use_filt, ref) in COMBOS.items():
        data, mask, labels = get_training_data(mods, ctx.modality_dict, ctx.modality_MASK, ctx.df_outcomes)
        l1 = ctx.rad_filters if use_filt else {}
        params = dict(B.MODEL_PARAMS)

        for mlabel, fn in [("DyAM", train_dyam_safe), ("uniform", train_uniform_avg)]:
            # 1-run: thứ tự tự nhiên (không permute) — đúng như notebook
            summ1, _ = fn(data, mask, labels, l1, params, folds=B.FOLDS, seed=42)
            auc1 = auc_of(summ1)
            # 5-seed hoán vị
            per = []
            for seed in B.SEEDS:
                rng = np.random.default_rng(seed)
                sh = labels.iloc[rng.permutation(len(labels))]
                s, _ = fn(data, mask, sh, l1, params, folds=B.FOLDS, seed=seed)
                per.append(auc_of(s))
            per = np.array(per)
            refstr = f"{ref:.4f}" if mlabel == "DyAM" else "-"
            print(f"{name:<26}{mlabel:<10}{refstr:>10}{auc1:>12.4f}"
                  f"{f'{per.mean():.4f}±{per.std(ddof=1):.4f}':>18}"
                  f"{f'[{per.min():.3f},{per.max():.3f}]':>16}", flush=True)
    print("=" * 96)


if __name__ == "__main__":
    main()
