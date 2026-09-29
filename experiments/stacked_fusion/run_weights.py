"""Step 3 — Trọng số meta-LR per modality (điểm bán paper: diễn giải thay attention).

Gọi thẳng train_stack (nó trả coef_df: trọng số meta per modality mỗi fold) qua 5 seed x 10 fold,
gộp mean±sd cho BM1 và BM2. Kiểm chứng story: radiomics bị hạ trọng số so với pathology/pdl1/gen?

Chuẩn hoá base-pred z-score theo train nên các cột meta coef so sánh được với nhau về độ lớn.
Dùng đúng cơ chế đổi phân hoạch CV như harness: hoán vị hàng outcomes theo seed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_CODE_DIR = Path(__file__).resolve().parents[2]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from lung_helpers import get_training_data                      # noqa: E402

from common import benchmarks as B                              # noqa: E402
from common.data_setup import load_all                          # noqa: E402
from common.evaluate import save_result                         # noqa: E402
from stacked_fusion.model_stack import train_stack              # noqa: E402

BMS = ["BM1", "BM2"]


def collect_weights(bm, ctx):
    data, mask, labels = get_training_data(
        B.BENCHMARKS[bm]["modalities"], ctx.modality_dict, ctx.modality_MASK, ctx.df_outcomes)
    l1_filter = B.get_filters(bm, ctx)
    mod_names = list(mask.columns)

    all_coefs = []
    for seed in B.SEEDS:
        rng = np.random.default_rng(seed)
        shuffled = labels.iloc[rng.permutation(len(labels))]
        _, coef_df = train_stack(data, mask, shuffled, l1_filter,
                                 dict(B.MODEL_PARAMS), folds=B.FOLDS, seed=seed)
        coef_df["seed"] = seed
        all_coefs.append(coef_df)

    coefs = pd.concat(all_coefs, ignore_index=True)
    # Gộp mean±sd qua seed x fold cho từng modality
    stats = {}
    for m in mod_names + ["intercept"]:
        stats[m] = {"mean": float(coefs[m].mean()), "sd": float(coefs[m].std(ddof=1)),
                    "mean_abs": float(coefs[m].abs().mean())}
    return mod_names, stats, coefs


def main():
    ctx = load_all()
    payload = {"note": "meta-LR coef per modality, gộp 5 seed x 10 fold. "
                       "base-pred z-score theo train nên coef so sánh được.",
               "benchmarks": {}}

    for bm in BMS:
        print(f"\n{'=' * 74}\n{bm}: {B.BENCHMARKS[bm]['name']} — trọng số meta\n{'=' * 74}")
        mod_names, stats, _ = collect_weights(bm, ctx)
        payload["benchmarks"][bm] = {"modalities": mod_names, "weights": stats}

        print(f"{'modality':<22}{'mean coef':>12}{'sd':>10}{'mean|coef|':>13}")
        for m in mod_names:
            s = stats[m]
            print(f"{m:<22}{s['mean']:>12.4f}{s['sd']:>10.4f}{s['mean_abs']:>13.4f}")
        s = stats["intercept"]
        print(f"{'(intercept)':<22}{s['mean']:>12.4f}{s['sd']:>10.4f}")

        # Story: radiomics vs non-radiomics (dùng mean|coef| = độ ảnh hưởng)
        rad = [m for m in mod_names if m.startswith("rad_")]
        non = [m for m in mod_names if not m.startswith("rad_")]
        rad_infl = np.mean([stats[m]["mean_abs"] for m in rad]) if rad else float("nan")
        non_infl = np.mean([stats[m]["mean_abs"] for m in non]) if non else float("nan")
        payload["benchmarks"][bm]["story"] = {
            "rad_mean_abs": float(rad_infl), "nonrad_mean_abs": float(non_infl),
            "ratio_nonrad_over_rad": float(non_infl / rad_infl) if rad_infl else None}
        print(f"\n  radiomics mean|coef| = {rad_infl:.4f} ({len(rad)} mod) | "
              f"non-rad mean|coef| = {non_infl:.4f} ({len(non)} mod) | "
              f"tỉ lệ non/rad = {non_infl / rad_infl:.2f}x")

    save_result("stacked_weights", payload)
    print(f"\n{'=' * 74}")
    for bm in BMS:
        st = payload["benchmarks"][bm]["story"]
        verdict = "HẠ radiomics" if st["ratio_nonrad_over_rad"] > 1.15 else "KHÔNG rõ hạ"
        print(f"{bm}: non/rad = {st['ratio_nonrad_over_rad']:.2f}x -> meta {verdict}")
    print("=" * 74)


if __name__ == "__main__":
    main()
