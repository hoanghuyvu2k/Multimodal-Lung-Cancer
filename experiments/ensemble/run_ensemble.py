"""Step 6 — Ensemble seed + calibration (robustness rẻ tiền, không GIỮ/XÓA).

sd giữa seed ≈ 0.02. Trung bình điểm dự đoán per-patient qua 5 seed có cho AUC ổn định hơn/nhỉnh
hơn một seed đơn không? Và điểm của backbone có calibrate được không (Brier)?

Chỉ hậu xử lý các score đã lưu ở results/scores/ — không train lại.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import benchmarks as B          # noqa: E402
from common.evaluate import load_scores, save_result  # noqa: E402

METHODS = ["uniform_avg", "original", "ovo", "lr_late"]


def calib_brier(y, ens):
    """Platt (LogisticRegression) trên chính điểm ensemble -> xác suất -> Brier. In-sample, mô tả."""
    p = LogisticRegression().fit(ens.reshape(-1, 1), y).predict_proba(ens.reshape(-1, 1))[:, 1]
    return float(brier_score_loss(y, p))


def main():
    payload = {"methods": {}}
    for m in METHODS:
        payload["methods"][m] = {}
        for bm in B.BENCHMARKS:
            try:
                d = load_scores(m, bm)
            except FileNotFoundError:
                continue
            y = d["_labels"].values.astype(float)
            seed_cols = d["_scores"].values                     # (n_patient, n_seed)
            per_seed = [roc_auc_score(y, seed_cols[:, j]) for j in range(seed_cols.shape[1])]
            ens = seed_cols.mean(axis=1)
            payload["methods"][m][bm] = {
                "mean_seed_auc": float(np.mean(per_seed)),
                "sd_seed_auc": float(np.std(per_seed, ddof=1)),
                "min_seed_auc": float(np.min(per_seed)),
                "max_seed_auc": float(np.max(per_seed)),
                "ensemble_auc": float(roc_auc_score(y, ens)),
                "brier_ensemble": calib_brier(y, ens),
            }

    save_result("ensemble", payload)

    print(f"\n{'=' * 92}\nENSEMBLE SEED — AUC ổn định hơn? (mean per-seed vs ensemble)\n{'=' * 92}")
    print(f"{'method':<14}{'BM':<6}{'mean seed':>11}{'min':>9}{'max':>9}"
          f"{'ensemble':>11}{'Δens':>9}{'Brier':>9}")
    for m in METHODS:
        for bm in B.BENCHMARKS:
            r = payload["methods"].get(m, {}).get(bm)
            if not r:
                continue
            print(f"{m:<14}{bm:<6}{r['mean_seed_auc']:>11.4f}{r['min_seed_auc']:>9.4f}"
                  f"{r['max_seed_auc']:>9.4f}{r['ensemble_auc']:>11.4f}"
                  f"{r['ensemble_auc'] - r['mean_seed_auc']:>+9.4f}{r['brier_ensemble']:>9.4f}")
    print("=" * 92)

    # Tóm tắt lợi ích ensemble trên backbone
    print("\nLợi ích ensemble (ensemble − mean per-seed), uniform_avg:")
    for bm in B.BENCHMARKS:
        r = payload["methods"]["uniform_avg"][bm]
        print(f"  {bm}: {r['ensemble_auc'] - r['mean_seed_auc']:+.4f} "
              f"(mean {r['mean_seed_auc']:.4f} -> ens {r['ensemble_auc']:.4f}, "
              f"span seed {r['max_seed_auc'] - r['min_seed_auc']:.4f})")
    save_result("ensemble", payload)


if __name__ == "__main__":
    main()
