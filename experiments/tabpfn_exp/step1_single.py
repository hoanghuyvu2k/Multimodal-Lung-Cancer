"""Step 1 — Per-modality: TabPFN vs LR (discovery-CV 5-seed)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data_setup import load_all                          # noqa: E402
from common.evaluate import save_result                         # noqa: E402
from validation.run_validation import largest_lesion, numeric_features, read_any  # noqa: E402
from tabpfn_exp.tabpfn_eval import cv_auc                        # noqa: E402


def main():
    ctx = load_all()
    y = ctx.df_outcomes["label"]

    mods = {}
    for name in ["cnl_pdl1_score", "gen_driver_mut_amp", "path_ihc_glcm", "cnl_dem_labs"]:
        X = ctx.modality_dict[name]
        present = ctx.modality_MASK.loc[X.index, name].values
        mods[name] = (X[present], y.loc[X[present].index])
    # radiomics largest-lesion (như validation)
    rad = numeric_features(largest_lesion(read_any(
        "lung_radiomics_spacing1.0_mirpon_window1350.250_allimagetypes_bw20.parquet")))
    yr = y.reindex(rad.index); keep = yr.notna()
    mods["radiomics"] = (rad[keep.values], yr[keep])

    out = {}
    print(f"\n{'=' * 62}\nPER-MODALITY: TabPFN vs LR (discovery-CV 5-seed)\n{'=' * 62}")
    print(f"{'modality':<20}{'n':>5}{'#feat':>7}{'LR':>10}{'TabPFN':>10}")
    for name, (X, yy) in mods.items():
        lm, ls = cv_auc(X, yy, "lr", k=50)
        tm, ts = cv_auc(X, yy, "tabpfn", k=50)
        out[name] = {"n": int(len(yy)), "n_feat": int(X.shape[1]),
                     "lr": [lm, ls], "tabpfn": [tm, ts]}
        print(f"{name:<20}{len(yy):>5}{X.shape[1]:>7}{lm:>10.4f}{tm:>10.4f}", flush=True)
    save_result("tabpfn_single", out)
    print("=" * 62)
    n_win = sum(1 for v in out.values() if v["tabpfn"][0] > v["lr"][0] + 0.005)
    print(f"TabPFN > LR (Δ>0.005) ở {n_win}/{len(out)} modality")


if __name__ == "__main__":
    main()
