"""Step 1 — Audit tín hiệu từng modality.

Fusion đã cạn (dự án trước) → tín hiệu nằm ở từng modality. Câu hỏi: modality nào là tín hiệu,
modality nào là nhiễu?

Với MỖI modality đơn lẻ:
  - Chỉ lấy các bệnh nhân THỰC SỰ có modality đó (mask=True) — đo đúng sức phân biệt của nó,
    không bị loãng bởi bệnh nhân vắng mặt (score = 0).
  - Chạy backbone `uniform_avg` (đơn-modality → chính là head tanh(risk) của modality đó) và
    baseline LR đơn-modality, 5 seeds × 10-fold (dùng mẹo hoán vị hàng để lấy phương sai CV).
  - Ghi AUC ± sd + n.

Radiomics cần L1 filter tương ứng ở vị trí 0; modality khác không có filter.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import benchmarks as B          # noqa: E402
from common.data_setup import load_all      # noqa: E402
from common.evaluate import save_result     # noqa: E402

from lung_helpers import get_training_data   # noqa: E402
from baselines.model_uniform_avg import train_uniform_avg  # noqa: E402
from baselines.model_lr import train_lr_concat             # noqa: E402


def rad_filter_for(name, ctx):
    """Map modality radiomics -> l1_selection_df của chính nó, đặt ở vị trí 0."""
    mapping = {
        "rad_lesion_pc": ctx.rad_filters[0],
        "rad_lesion_pl": ctx.rad_filters[1],
        "rad_lesion_ln": ctx.rad_filters[2],
        "rad_lesion_lu": ctx.rad_filters_lu[3],
    }
    if name in mapping:
        return {0: mapping[name]}
    return {}


def run_single(train_fn, name, ctx, seeds=None, folds=None):
    seeds = seeds if seeds is not None else B.SEEDS
    folds = folds if folds is not None else B.FOLDS

    # Chỉ giữ bệnh nhân có modality này
    present = ctx.modality_MASK[ctx.modality_MASK[name]].index
    outcomes = ctx.df_outcomes.loc[ctx.df_outcomes.index.intersection(present)]

    data, mask, labels = get_training_data(
        [name], ctx.modality_dict, ctx.modality_MASK, outcomes)
    l1_filter = rad_filter_for(name, ctx)

    aucs = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        shuffled = labels.iloc[rng.permutation(len(labels))]
        summary_df, _ = train_fn(data, mask, shuffled, l1_filter, dict(B.MODEL_PARAMS),
                                 folds=folds, seed=seed)
        y = summary_df["label"].astype(float).values
        s = summary_df["score"].astype(float).values
        aucs.append(float(roc_auc_score(y, s)))
    return {
        "n": int(len(labels)),
        "pos": int(labels["label"].sum()),
        "mean_auc": float(np.mean(aucs)),
        "sd_auc": float(np.std(aucs, ddof=1)),
        "per_seed_auc": aucs,
    }


def main():
    ctx = load_all()
    names = sorted(ctx.modality_dict.keys())
    payload = {"uniform_avg": {}, "lr": {}}

    for name in names:
        print(f"\n[{name}]", flush=True)
        try:
            u = run_single(train_uniform_avg, name, ctx)
            l = run_single(train_lr_concat, name, ctx)
        except Exception as e:  # noqa: BLE001
            print(f"  LỖI {name}: {e}", flush=True)
            continue
        payload["uniform_avg"][name] = u
        payload["lr"][name] = l
        print(f"  n={u['n']:>3} pos={u['pos']:>3} | uniform_avg {u['mean_auc']:.4f}±{u['sd_auc']:.4f}"
              f" | LR {l['mean_auc']:.4f}±{l['sd_auc']:.4f}", flush=True)
        save_result("signal_audit", payload)

    # Bảng xếp hạng theo uniform_avg
    print(f"\n{'=' * 72}\nXẾP HẠNG TÍN HIỆU PER-MODALITY (uniform_avg, 5 seeds × 10-fold)\n{'=' * 72}")
    print(f"{'modality':<22}{'n':>5}{'pos':>5}{'uniform_avg':>16}{'LR':>16}  cờ")
    print("-" * 72)
    ranked = sorted(payload["uniform_avg"].items(),
                    key=lambda kv: kv[1]["mean_auc"], reverse=True)
    for name, u in ranked:
        l = payload["lr"][name]
        flag = "NHIỄU" if u["mean_auc"] <= 0.53 else ""
        print(f"{name:<22}{u['n']:>5}{u['pos']:>5}"
              f"{u['mean_auc']:>8.4f}±{u['sd_auc']:<6.4f}"
              f"{l['mean_auc']:>8.4f}±{l['sd_auc']:<6.4f}  {flag}")
    print("=" * 72)
    save_result("signal_audit", payload)


if __name__ == "__main__":
    main()
