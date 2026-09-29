"""Step 0 — Khóa backbone uniform_avg.

- Trích số nền uniform_avg (BM1–BM4) từ results/method_B.json → results/backbone_lock.json.
- Chạy lại uniform_avg trên BM1 (5 seeds × 10-fold) và xác nhận khớp số nền 0.7746.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import benchmarks as B          # noqa: E402
from common.data_setup import load_all      # noqa: E402
from common.evaluate import (               # noqa: E402
    load_result, run_repeated_cv, save_result, strip_internals,
)

from baselines.model_uniform_avg import train_uniform_avg  # noqa: E402


def main():
    mB = load_result("method_B")["methods"]["uniform_avg"]
    locked = {bm: {"mean_auc": mB[bm]["mean_auc"], "sd_auc": mB[bm]["sd_auc"]}
              for bm in B.BENCHMARKS}
    print("Số nền uniform_avg (từ method_B.json):")
    for bm, v in locked.items():
        print(f"  {bm}: {v['mean_auc']:.4f} ± {v['sd_auc']:.4f}")

    ctx = load_all()
    print("\nChạy lại uniform_avg BM1 (5 seeds × 10-fold) để kiểm chứng khớp...", flush=True)
    res = run_repeated_cv(train_uniform_avg, "BM1", ctx)
    repro = res["mean_auc"]
    ref = locked["BM1"]["mean_auc"]
    delta = repro - ref
    print(f"\nBM1 tái lập = {repro:.4f} | nền = {ref:.4f} | Δ = {delta:+.5f}")
    match = abs(delta) < 1e-6

    payload = {
        "backbone": "uniform_avg",
        "source": "experiments/baselines/model_uniform_avg.py",
        "locked_auc": locked,
        "reproduce_check": {"bm": "BM1", "value": repro, "ref": ref,
                            "delta": delta, "exact_match": match},
    }
    save_result("backbone_lock", payload)
    print(f"\nĐã ghi results/backbone_lock.json — khớp tuyệt đối: {match}")


if __name__ == "__main__":
    main()
