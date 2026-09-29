"""Bổ sung cột single-run (1-lần, thứ tự tự nhiên = cách bài báo) cho 21 tổ hợp, gộp vào paper_combos.json.

Chạy uniform_avg và DyAM MỘT lần (không hoán vị nhãn, seed 42) cho mỗi combo -> AUC single-run.
Ghép với 5-seed đã có để in bảng đối chiếu trực tiếp với bài báo.
"""

import sys
from pathlib import Path

from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lung_helpers import get_training_data                      # noqa: E402
from baselines.model_uniform_avg import train_uniform_avg       # noqa: E402
from common import benchmarks as B                              # noqa: E402
from common.data_setup import load_all                          # noqa: E402
from common.evaluate import load_result, save_result           # noqa: E402
from allcombo.run_combo import train as train_dyam             # noqa: E402
from allcombo.run_paper_combos import PAPER_COMBOS, make_filter  # noqa: E402

# ref bài báo (chỉ cho các combo có trong benchmarks); None nếu không có
PAPER_REF = {
    "Rad+IHC-G+Gen+PDL1": 0.7839, "Rad+IHC-G+Gen+PDL1+Labs": 0.7879,
    "Rad+Gen": 0.7384, "PDL1+Gen": 0.6931,
}


def single_run_auc(train_fn, mods, l1, ctx):
    data, mask, labels = get_training_data(mods, ctx.modality_dict, ctx.modality_MASK, ctx.df_outcomes)
    summ, _ = train_fn(data, mask, labels, l1, dict(B.MODEL_PARAMS), folds=B.FOLDS, seed=42)
    return float(roc_auc_score(summ["label"].astype(float).values, summ["score"].astype(float).values))


def main():
    ctx = load_all()
    payload = load_result("paper_combos")
    c = payload["combos"]

    print(f"\n{'=' * 104}")
    print("21 TỔ HỢP — 5-seed (chính) vs single-run (như bài báo). DyAM đối chiếu ref bài báo.")
    print(f"{'=' * 104}")
    print(f"{'#':>3} {'combo':<26}{'#mod':>5}"
          f"{'uni 5seed':>11}{'uni 1run':>10}{'DyAM 5seed':>12}{'DyAM 1run':>11}{'paper_ref':>11}")
    for i, (label, mods, ftype) in enumerate(PAPER_COMBOS, 1):
        l1 = make_filter(ftype, ctx)
        u1 = single_run_auc(train_uniform_avg, mods, l1, ctx)
        d1 = single_run_auc(train_dyam, mods, l1, ctx)
        c[label]["uniform"]["single_run_auc"] = u1
        c[label]["dyam"]["single_run_auc"] = d1
        save_result("paper_combos", payload)
        u5 = c[label]["uniform"]["mean_auc"]
        d5 = c[label]["dyam"]["mean_auc"]
        ref = PAPER_REF.get(label)
        refstr = f"{ref:.4f}" if ref else "-"
        print(f"{i:>3} {label:<26}{len(mods):>5}"
              f"{u5:>11.4f}{u1:>10.4f}{d5:>12.4f}{d1:>11.4f}{refstr:>11}", flush=True)
    print("=" * 104)


if __name__ == "__main__":
    main()
