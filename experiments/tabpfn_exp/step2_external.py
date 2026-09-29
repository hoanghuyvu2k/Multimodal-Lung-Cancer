"""Step 2 — External: TabPFN vs LR (train discovery -> test path_valid / rad_valid)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data_setup import load_all                          # noqa: E402
from common.evaluate import save_result                         # noqa: E402
from validation.run_validation import (valid_labels, largest_lesion,  # noqa: E402
                                        numeric_features, read_any)
from tabpfn_exp.tabpfn_eval import ext_auc                       # noqa: E402


def main():
    ctx = load_all()
    y = ctx.df_outcomes["label"]
    rad_lab, path_lab = valid_labels()
    out = {}

    # ---- PATHOLOGY glcm ----
    Xtr = ctx.modality_dict["path_ihc_glcm"]
    present = ctx.modality_MASK.loc[Xtr.index, "path_ihc_glcm"].values
    Xtr = Xtr[present]; ytr = y.loc[Xtr.index]
    val = numeric_features(read_any("lung_pathology_pdl1_glcm_v3_validation.parquet"))
    val.index = val.index.astype(str)
    cp = val.index.intersection(path_lab.index.astype(str))
    Xte, yte = val.loc[cp], path_lab.loc[cp].values
    cols = Xtr.columns.intersection(Xte.columns)
    for mdl in ("lr", "tabpfn"):
        a, ci, n = ext_auc(Xtr[cols], ytr.values, Xte[cols], yte, mdl, k=50)
        out.setdefault("pathology", {})[mdl] = {"auc": a, "ci": ci, "n": n}

    # ---- RADIOMICS largest-lesion ----
    rd = numeric_features(largest_lesion(read_any(
        "lung_radiomics_spacing1.0_mirpon_window1350.250_allimagetypes_bw20.parquet")))
    yr = y.reindex(rd.index); keep = yr.notna(); rd, yr = rd[keep.values], yr[keep]
    rv = numeric_features(largest_lesion(read_any(
        "lung_radiomics_spacing1.0_mirpon_window1350.250_allimagetypes_bw20_validation.parquet")))
    cr = rv.index.intersection(rad_lab.index.astype(str))
    rv_c, yrv = rv.loc[cr], rad_lab.loc[cr].values
    cols = rd.columns.intersection(rv_c.columns)
    for mdl in ("lr", "tabpfn"):
        a, ci, n = ext_auc(rd[cols], yr.values, rv_c[cols], yrv, mdl, k=50)
        out.setdefault("radiomics", {})[mdl] = {"auc": a, "ci": ci, "n": n}

    save_result("tabpfn_external", out)
    print(f"\n{'=' * 60}\nEXTERNAL: TabPFN vs LR\n{'=' * 60}")
    print(f"{'modality':<12}{'LR':>22}{'TabPFN':>22}")
    for m in ("pathology", "radiomics"):
        lr, tp = out[m]["lr"], out[m]["tabpfn"]
        ls = f"{lr['auc']:.4f} [{lr['ci'][0]:.2f},{lr['ci'][1]:.2f}]"
        ts = f"{tp['auc']:.4f} [{tp['ci'][0]:.2f},{tp['ci'][1]:.2f}]"
        print(f"{m:<12}{ls:>22}{ts:>22}")
    print("=" * 60)


if __name__ == "__main__":
    main()
