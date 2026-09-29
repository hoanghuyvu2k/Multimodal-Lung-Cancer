"""
Step 6 — Chạy baseline LR trên BM1-BM4, 5 seeds x 10-fold, ghép cặp với các mô hình đã lưu.

Câu hỏi cần trả lời: mô hình neural đa modality có vượt được hồi quy logistic thường không?
Nếu không, thì toàn bộ kiến trúc chưa chứng minh được giá trị.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import benchmarks as B          # noqa: E402
from common.data_setup import load_all      # noqa: E402
from common.evaluate import (               # noqa: E402
    load_result, load_scores, paired_bootstrap, run_repeated_cv,
    save_result, save_scores, strip_internals,
)

from baselines.model_lr import train_lr_concat, train_lr_late  # noqa: E402

METHODS = {"lr_concat": train_lr_concat, "lr_late": train_lr_late}
REFS = ("original", "ovo", "uniform_avg")


def main():
    ctx = load_all()
    payload = {"methods": {}, "paired": {}}

    for bm_key in B.BENCHMARKS:
        print(f"\n{'=' * 70}\n{bm_key} — {B.BENCHMARKS[bm_key]['name']}\n{'=' * 70}",
              flush=True)
        ref = {m: load_scores(m, bm_key) for m in REFS}

        for mname, fn in METHODS.items():
            print(f"  [{mname}]", flush=True)
            res = run_repeated_cv(fn, bm_key, ctx)
            save_scores(mname, bm_key, res)
            payload["methods"].setdefault(mname, {})[bm_key] = strip_internals(res)

            for ref_name, ref_res in ref.items():
                payload["paired"].setdefault(f"{mname}_vs_{ref_name}", {})[bm_key] = dict(
                    zip(("delta_auc", "ci_low", "ci_high", "p_value"),
                        paired_bootstrap(res, ref_res)))

            print(f"  {bm_key} {mname:<10}: {res['mean_auc']:.4f} ± {res['sd_auc']:.4f}",
                  flush=True)

        save_result("baseline_lr", payload)

    # Bảng tổng hợp
    prev = load_result("method_B")["methods"]
    base = load_result("method_A")["methods"]
    print(f"\n\n{'=' * 100}\nBASELINE LR vs MÔ HÌNH NEURAL (mean ± sd, 5 seeds × 10-fold)\n"
          f"{'=' * 100}")
    print(f"{'BM':<5} {'Original':>16} {'OvO':>16} {'uniform_avg':>16} "
          f"{'lr_concat':>16} {'lr_late':>16}")
    print("-" * 100)
    for bm_key in B.BENCHMARKS:
        row = [
            base["original"][bm_key], base["ovo"][bm_key], prev["uniform_avg"][bm_key],
            payload["methods"]["lr_concat"][bm_key], payload["methods"]["lr_late"][bm_key],
        ]
        print(f"{bm_key:<5} " + " ".join(
            f"{r['mean_auc']:>7.4f}±{r['sd_auc']:<7.4f}" for r in row))
    print("=" * 100)

    print("\nCâu hỏi then chốt — neural có hơn LR không?")
    for bm_key in B.BENCHMARKS:
        d = payload["paired"]["lr_late_vs_uniform_avg"][bm_key]
        print(f"  {bm_key} lr_late − uniform_avg = {d['delta_auc']:+.4f} "
              f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] p={d['p_value']:.3f}")
    for bm_key in B.BENCHMARKS:
        d = payload["paired"]["lr_late_vs_original"][bm_key]
        print(f"  {bm_key} lr_late − original    = {d['delta_auc']:+.4f} "
              f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] p={d['p_value']:.3f}")

    save_result("baseline_lr", payload)


if __name__ == "__main__":
    main()
