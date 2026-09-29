"""Step 2 — Đánh giá nội bộ đầy đủ Stacked Late-Fusion.

Chạy train_stack trên BM1-BM4 x 5 seeds x 10-fold. Lưu results/stacked.json + scores CSV.
Paired bootstrap vs uniform_avg (nền), original, ovo, lr_late. Kiểm tra điều kiện GIỮ:
stacked KHÔNG thua có ý nghĩa uniform_avg trên cả 4 BM (không BM nào p<0.05 theo hướng stacked thua).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import benchmarks as B                              # noqa: E402
from common.data_setup import load_all                          # noqa: E402
from common.evaluate import (                                   # noqa: E402
    load_scores, paired_bootstrap, run_repeated_cv, save_result,
    save_scores, strip_internals,
)
from stacked_fusion.model_stack import train_stack              # noqa: E402

COMPARATORS = ["uniform_avg", "original", "ovo", "lr_late"]


def main():
    ctx = load_all()
    payload = {"method": "stacked", "benchmarks": {}}

    for bm in B.BENCHMARKS:
        print(f"\n{'=' * 70}\n{bm}: {B.BENCHMARKS[bm]['name']}\n{'=' * 70}")
        res = run_repeated_cv(train_stack, bm, ctx, seeds=B.SEEDS)
        save_scores("stacked", bm, res)

        entry = strip_internals(res)
        entry["compare"] = {}
        for other in COMPARATORS:
            try:
                ref = load_scores(other, bm)
            except FileNotFoundError:
                continue
            delta, lo, hi, p = paired_bootstrap(res, ref)
            entry["compare"][other] = {
                "delta_auc": delta, "ci_low": lo, "ci_high": hi, "p_value": p}
        payload["benchmarks"][bm] = entry

    save_result("stacked", payload)

    # Bảng tổng hợp + kiểm tra điều kiện GIỮ
    print(f"\n{'=' * 78}\nSTACKED — AUC nội bộ (5 seed) và Δ vs uniform_avg\n{'=' * 78}")
    print(f"{'BM':<5}{'stacked':>16}{'uniform_avg':>14}{'Δ':>9}{'95% CI':>20}{'p':>8}")
    loses = []
    for bm in B.BENCHMARKS:
        e = payload["benchmarks"][bm]
        u = e["compare"].get("uniform_avg")
        stk = f"{e['mean_auc']:.4f}±{e['sd_auc']:.4f}"
        if u:
            d, lo, hi, p = u["delta_auc"], u["ci_low"], u["ci_high"], u["p_value"]
            print(f"{bm:<5}{stk:>16}{B.BENCHMARKS[bm]['name'][:12]:>14}"
                  f"{d:>+9.4f}{f'[{lo:+.3f},{hi:+.3f}]':>20}{p:>8.3f}")
            if p < 0.05 and d < 0:
                loses.append(bm)
        else:
            print(f"{bm:<5}{stk:>16}{'(no ref)':>14}")
    print("=" * 78)

    if loses:
        print(f"\n[XOA?] stacked thua uniform_avg có ý nghĩa ở: {loses} "
              f"({len(loses)}/4). Nếu >=2/4 -> XÓA theo tiêu chí 1.2.")
    else:
        print("\n[GIU-a] stacked KHÔNG thua uniform_avg có ý nghĩa ở BM nào -> đạt điều kiện (a). "
              "Sang Step 3 kiểm tra trọng số meta (điều kiện b).")

    # Δ so với các phương pháp attention
    print(f"\n{'=' * 78}\nΔ stacked vs các phương pháp (paired bootstrap)\n{'=' * 78}")
    print(f"{'BM':<5}" + "".join(f"{m:>16}" for m in COMPARATORS))
    for bm in B.BENCHMARKS:
        cmp = payload["benchmarks"][bm]["compare"]
        cells = []
        for m in COMPARATORS:
            c = cmp.get(m)
            cells.append(f"{c['delta_auc']:+.4f}({c['p_value']:.2f})" if c else "-")
        print(f"{bm:<5}" + "".join(f"{c:>16}" for c in cells))
    print("=" * 78)


if __name__ == "__main__":
    main()
