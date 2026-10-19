"""Một phân hoạch (giao thức gốc: seed 42, thứ tự bệnh nhân tự nhiên) trong ref-linux-py38, cho bản bài 1-phân-hoạch.

  1) 11 biến thể lâm sàng (nhánh IHC-A: 7, IHC-G: 4) bằng DyAM `train`, bộ lọc ctx.rad_filters: AUC (+DeLong CI), C-index, Cox HR, KM chi2 (trung vị).
  2) BM1 (Rad+IHC-G+Gen+PDL1) x {DyAM, OvO, NLP+DyAM, NLP+OvO}: AUC + DeLong CI, lưu điểm.

    run_ref.sh experiments/reference_env/run_single_partition_ref.py
Kết quả: experiments/results/ref/single/{single_partition.json, scores/*.csv}
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1])); sys.path.insert(0, str(_HERE.parents[2]))

from common import benchmarks as B                                          # noqa: E402
from common.data_setup import load_all                                      # noqa: E402
from nlp_verify import verify_nlp_5seed as V                                # noqa: E402
from nlp_verify.verify_nlp_variants import ARMS, VARIANTS, add_pca16, slug  # noqa: E402
from nlp_verify.nlp_modality import add_nlp_modality                        # noqa: E402
from allcombo.run_paper_combos import PAPER_COMBOS, make_filter             # noqa: E402
from allcombo.run_combo import train as train_dyam                          # noqa: E402
from allcombo.run_ovo_singlerun import train_ovo_safe                       # noqa: E402
from lung_helpers import (auc_roc_ci, compute_cindex, compute_multivariate_cox, get_training_data, train)   # noqa: E402

OUT = _HERE.parents[1] / "results" / "win_doanenv" / "single"
(OUT / "scores").mkdir(parents=True, exist_ok=True)
BIO = Path("D:/repro_linux/cache/bioclin_embed_v2.parquet")


def add_bio(ctx, name="cnl_bioclin"):
    df = pd.read_parquet(BIO)
    ctx.modality_MASK[name] = False
    ctx.modality_MASK.loc[df.index, name] = True
    ctx.modality_dict[name] = df


def main():
    ctx = load_all()
    add_nlp_modality(ctx)
    add_pca16(ctx, "cnl_nlp", "cnl_nlp_pca16")
    add_bio(ctx)
    add_pca16(ctx, "cnl_bioclin", "cnl_bioclin_pca16")
    clin = V.load_clinical()
    res = {"clinical_variants": {}, "bm1": {}}

    for arm, arm_mod in ARMS.items():
        for name, extra, no_scale in VARIANTS:
            if (arm, name) not in V.PAPER_AUC:
                continue
            mods = V.BASE + [arm_mod] + extra
            data, mask, labels = get_training_data(mods, ctx.modality_dict, ctx.modality_MASK, ctx.df_outcomes)
            params = dict(V.MODEL_PARAMS)
            idx = [list(mask.columns).index(m) for m in no_scale if m in mask.columns]
            if idx:
                params["no_scale"] = idx
            summ, _ = train(data, mask, labels, ctx.rad_filters, params, folds=V.FOLDS, seed=42)
            y, s = summ["label"].astype(float).values, summ["score"].astype(float).values
            auc, ci = auc_roc_ci(y, s, 0.95)
            try:
                hr = float(compute_multivariate_cox(summ, clin).summary.loc["score", "exp(coef)"])
            except Exception:
                hr = float("nan")
            r = {"auc": float(auc), "auc_ci": [float(ci[0]), float(ci[1])], "cindex": float(compute_cindex(summ, clin)), "cox_hr": hr,
                 "km_chi2": V.km_chi2(summ, clin)}
            res["clinical_variants"][f"{arm}__{slug(name)}"] = r
            summ[["label", "score"]].to_csv(OUT / "scores" / f"{arm}__{slug(name)}.csv")
            print(f"[single] {arm} {name}: AUC {auc:.4f} C {r['cindex']:.3f} HR {hr:.2f} KM {r['km_chi2']:.1f}", flush=True)
            (OUT / "single_partition.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    print("XONG")


if __name__ == "__main__":
    main()
