"""
Step 3 — Khóa baseline: Original vs OvO trên BM1-BM4, 5 seeds x 10-fold.

Đây là bảng tham chiếu cho mọi phương pháp về sau. Ghi kết quả tăng dần sau mỗi
benchmark để nếu bị ngắt giữa chừng vẫn giữ được phần đã chạy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import benchmarks as B          # noqa: E402
from common.data_setup import load_all      # noqa: E402
from common.evaluate import (               # noqa: E402
    paired_bootstrap, run_repeated_cv, save_result, strip_internals,
)

from lung_helpers import train, train_ovo   # noqa: E402


def train_ovo_adapter(data, mask, labels, l1_filter, params, folds, seed):
    """train_ovo không nhận cross_modality_enabled."""
    p = {k: v for k, v in params.items() if k != "cross_modality_enabled"}
    return train_ovo(data, mask, labels, l1_filter, p, folds=folds, seed=seed)


def main():
    ctx = load_all()
    payload = {"methods": {"original": {}, "ovo": {}}, "paired": {}}

    for bm_key in B.BENCHMARKS:
        name = B.BENCHMARKS[bm_key]["name"]
        print(f"\n{'=' * 70}\n{bm_key} — {name}\n{'=' * 70}", flush=True)

        print("  [original]", flush=True)
        res_org = run_repeated_cv(train, bm_key, ctx)
        print("  [ovo]", flush=True)
        res_ovo = run_repeated_cv(train_ovo_adapter, bm_key, ctx)

        delta, lo, hi, p = paired_bootstrap(res_ovo, res_org)

        payload["methods"]["original"][bm_key] = strip_internals(res_org)
        payload["methods"]["ovo"][bm_key] = strip_internals(res_ovo)
        payload["paired"][bm_key] = {
            "comparison": "ovo - original",
            "delta_auc": delta, "ci_low": lo, "ci_high": hi, "p_value": p,
        }

        print(f"\n  {bm_key} Original : {res_org['mean_auc']:.4f} ± {res_org['sd_auc']:.4f}")
        print(f"  {bm_key} OvO      : {res_ovo['mean_auc']:.4f} ± {res_ovo['sd_auc']:.4f}")
        print(f"  {bm_key} Δ(OvO-Org): {delta:+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.3f}",
              flush=True)

        save_result("baseline", payload)

    print(f"\n\n{'=' * 70}\nBẢNG TỔNG HỢP (mean ± sd, 5 seeds x 10-fold)\n{'=' * 70}")
    print(f"{'BM':<5} {'config':<26} {'Original':>16} {'OvO':>16} {'Δ':>9} {'p':>7}")
    print("-" * 70)
    for bm_key in payload["methods"]["original"]:
        o = payload["methods"]["original"][bm_key]
        v = payload["methods"]["ovo"][bm_key]
        pr = payload["paired"][bm_key]
        print(f"{bm_key:<5} {o['benchmark_name']:<26} "
              f"{o['mean_auc']:>7.4f}±{o['sd_auc']:<7.4f} "
              f"{v['mean_auc']:>7.4f}±{v['sd_auc']:<7.4f} "
              f"{pr['delta_auc']:>+9.4f} {pr['p_value']:>7.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
