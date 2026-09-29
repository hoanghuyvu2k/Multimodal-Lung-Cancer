"""Step 4 — Seed-ensemble + calibration cho stacked (hậu xử lý, không train lại).

Đọc results/scores/stacked__BM*.csv (5 seed) + uniform_avg. Tính:
- mean per-seed AUC vs ensemble AUC (trung bình điểm qua 5 seed)
- Brier (Platt in-sample, mô tả) cho stacked vs uniform_avg
Cập nhật results/stacked.json phần "ensemble".
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import benchmarks as B                              # noqa: E402
from common.evaluate import (                                   # noqa: E402
    load_result, load_scores, save_result,
)


def calib_brier(y, ens):
    p = LogisticRegression().fit(ens.reshape(-1, 1), y).predict_proba(ens.reshape(-1, 1))[:, 1]
    return float(brier_score_loss(y, p))


def summarize(method, bm):
    d = load_scores(method, bm)
    y = d["_labels"].values.astype(float)
    cols = d["_scores"].values
    per_seed = [roc_auc_score(y, cols[:, j]) for j in range(cols.shape[1])]
    ens = cols.mean(axis=1)
    return {
        "mean_seed_auc": float(np.mean(per_seed)),
        "sd_seed_auc": float(np.std(per_seed, ddof=1)),
        "ensemble_auc": float(roc_auc_score(y, ens)),
        "brier_ensemble": calib_brier(y, ens),
    }


def main():
    payload = load_result("stacked")
    payload.setdefault("ensemble", {})

    print(f"\n{'=' * 84}\nSTACKED — seed-ensemble + Brier (so với uniform_avg)\n{'=' * 84}")
    print(f"{'BM':<5}{'stk mean':>10}{'stk ens':>10}{'Δens':>8}{'stk Brier':>11}"
          f"{'unif ens':>10}{'unif Brier':>12}")
    for bm in B.BENCHMARKS:
        stk = summarize("stacked", bm)
        unif = summarize("uniform_avg", bm)
        payload["ensemble"][bm] = {"stacked": stk, "uniform_avg": unif}
        print(f"{bm:<5}{stk['mean_seed_auc']:>10.4f}{stk['ensemble_auc']:>10.4f}"
              f"{stk['ensemble_auc'] - stk['mean_seed_auc']:>+8.4f}{stk['brier_ensemble']:>11.4f}"
              f"{unif['ensemble_auc']:>10.4f}{unif['brier_ensemble']:>12.4f}")
    print("=" * 84)

    # Nhận xét ngắn
    print("\nΔ ensemble AUC (stacked − uniform_avg) và Δ Brier (âm = stacked calibrate tốt hơn):")
    for bm in B.BENCHMARKS:
        e = payload["ensemble"][bm]
        dauc = e["stacked"]["ensemble_auc"] - e["uniform_avg"]["ensemble_auc"]
        dbri = e["stacked"]["brier_ensemble"] - e["uniform_avg"]["brier_ensemble"]
        print(f"  {bm}: ΔAUC_ens = {dauc:+.4f} | ΔBrier = {dbri:+.4f}")

    save_result("stacked", payload)


if __name__ == "__main__":
    main()
