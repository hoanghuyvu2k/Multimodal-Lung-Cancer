"""Chạy OvO qua ĐÚNG pipeline đã dùng cho uniform & DyAM (run_model 5-seed + single-run, cùng guard)
trên cả 21 tổ hợp bài báo. Gộp vào paper_combos.json để có bảng 3 model đồng nhất cấu hình.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.data_setup import load_all                          # noqa: E402
from common.evaluate import load_result, paired_bootstrap, save_result, save_scores  # noqa: E402
from allcombo.run_paper_combos import (                         # noqa: E402
    PAPER_COMBOS, make_filter, run_model, slug,
)
from allcombo.run_paper_singlerun import single_run_auc         # noqa: E402
from allcombo.run_ovo_singlerun import train_ovo_safe           # noqa: E402


def main():
    ctx = load_all()
    payload = load_result("paper_combos")
    c = payload["combos"]

    print(f"\n{'=' * 110}")
    print("21 TỔ HỢP — OvO cùng cấu hình uniform/DyAM. So 5-seed và single-run cả 3 model.")
    print(f"{'=' * 110}")
    print(f"{'#':>3} {'combo':<26}{'#mod':>5}"
          f"{'uni5s':>8}{'DyAM5s':>8}{'OvO5s':>8}  |{'uni1r':>8}{'DyAM1r':>8}{'OvO1r':>8}")
    for i, (label, mods, ftype) in enumerate(PAPER_COMBOS, 1):
        l1 = make_filter(ftype, ctx)
        ro = run_model(train_ovo_safe, mods, l1, ctx, __import__("common.benchmarks", fromlist=["SEEDS"]).SEEDS)
        o1 = single_run_auc(train_ovo_safe, mods, l1, ctx)
        # paired bootstrap OvO vs uniform (5-seed)
        ru = {"_scores": None}
        c[label]["ovo"] = {
            "mean_auc": ro["mean_auc"], "sd_auc": ro["sd_auc"],
            "per_seed_auc": ro["per_seed_auc"], "single_run_auc": o1,
        }
        save_scores(f"ovo_{slug(label)}", "PAPER", ro)
        save_result("paper_combos", payload)
        e = c[label]
        print(f"{i:>3} {label:<26}{len(mods):>5}"
              f"{e['uniform']['mean_auc']:>8.4f}{e['dyam']['mean_auc']:>8.4f}{ro['mean_auc']:>8.4f}  |"
              f"{e['uniform']['single_run_auc']:>8.4f}{e['dyam']['single_run_auc']:>8.4f}{o1:>8.4f}",
              flush=True)
    print("=" * 110)


if __name__ == "__main__":
    main()
