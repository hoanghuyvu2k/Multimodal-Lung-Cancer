"""Eval fmcib (thật): single-modality (vs radiomics/BiomedCLIP/MedicalNet) + fusion đa nguồn."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data_setup import load_all                          # noqa: E402
from common.evaluate import RESULTS_DIR, save_result            # noqa: E402
from validation.run_validation import (valid_labels, largest_lesion,  # noqa: E402
                                        numeric_features, read_any)
from foundation_embed.ct_step45_eval import cv_auc, ext_auc     # noqa: E402
from foundation_embed.fm_fusion_compare import add_embed_modality, run_cfg  # noqa: E402
from baselines.model_uniform_avg import train_uniform_avg       # noqa: E402
from allcombo.run_combo import train as train_dyam             # noqa: E402

DS = Path(__file__).resolve().parents[2].parent / "datasets"


def main():
    ctx = load_all()
    y = ctx.df_outcomes["label"]
    rad_lab, _ = valid_labels()

    # ---------- SINGLE MODALITY ----------
    fd = pd.read_parquet(DS / "ct_fmcib_embed_discovery.parquet")
    fv = pd.read_parquet(DS / "ct_fmcib_embed_rad_valid.parquet")
    fd = fd[fd.index.isin(y.index)]
    yfd = y.loc[fd.index]
    cvf = fv.index.intersection(rad_lab.index.astype(str))
    fv_c, yfv = fv.loc[cvf], rad_lab.loc[cvf].values

    prev = json.load(open(RESULTS_DIR / "fm_ct.json", encoding="utf-8"))
    single = {"fmcib": {}}
    single["fmcib"]["discovery_cv"] = dict(zip(("mean", "sd"), cv_auc(fd, yfd)))
    a, ci, n = ext_auc(fd, yfd.values, fv_c, yfv)
    single["fmcib"]["external"] = {"auc": a, "ci": ci, "n": n}

    print(f"\n{'=' * 74}\nSINGLE-MODALITY: fmcib vs khác (discovery-CV | external)\n{'=' * 74}")
    print(f"{'model':<20}{'discovery-CV':>16}{'external':>26}")
    rows = [
        ("radiomics", prev["discovery_cv"]["radiomics"], prev["external"]["radiomics"]),
        ("BiomedCLIP", prev["discovery_cv"]["ct_embed"], prev["external"]["ct_embed"]),
        ("fmcib (thật)", single["fmcib"]["discovery_cv"], single["fmcib"]["external"]),
    ]
    for nm, dc, ex in rows:
        dcv = f"{dc['mean']:.4f}±{dc['sd']:.4f}"
        ext = f"{ex['auc']:.4f} [{ex['ci'][0]:.2f},{ex['ci'][1]:.2f}] n={ex['n']}"
        print(f"{nm:<20}{dcv:>16}{ext:>26}")
    print("=" * 74)

    # ---------- FUSION ----------
    n_r = add_embed_modality(ctx, "rad_fmcib", "ct_fmcib_embed_discovery.parquet")
    n_p = add_embed_modality(ctx, "path_emb", "path_fm_embed_discovery.parquet")
    print(f"\nfusion modalities: rad_fmcib n={n_r}, path_emb n={n_p}")
    G, D = "gen_driver_mut_amp", "cnl_pdl1_score"
    RAD, PHC = ["rad_lesion_pc", "rad_lesion_pl", "rad_lesion_ln"], ["path_ihc_glcm"]
    configs = {
        "A hand-crafted (BM1)":        RAD + PHC + [G, D],
        "B fmcib + Phikon":            ["rad_fmcib", "path_emb", G, D],
        "D fmcib rad + path HC":       ["rad_fmcib"] + PHC + [G, D],
        "E BM1 + fmcib (thêm)":        RAD + PHC + [G, D, "rad_fmcib"],
    }
    fus = {}
    print(f"\n{'=' * 74}\nFUSION (discovery-CV 5-seed)\n{'=' * 74}")
    print(f"{'config':<26}{'uniform_avg':>16}{'DyAM':>16}")
    for name, mods in configs.items():
        um, us = run_cfg(train_uniform_avg, mods, ctx)
        dm, ds = run_cfg(train_dyam, mods, ctx)
        fus[name] = {"uniform": [um, us], "dyam": [dm, ds], "modalities": mods}
        uc = f"{um:.4f}±{us:.4f}"; dc2 = f"{dm:.4f}±{ds:.4f}"
        print(f"{name:<26}{uc:>16}{dc2:>16}", flush=True)
    print("=" * 74)

    save_result("fmcib_eval", {"single": single, "fusion": fus})
    fa = single["fmcib"]["external"]["auc"]
    print(f"\nĐIỂM CHỐT single external: fmcib {fa:.4f} vs radiomics "
          f"{prev['external']['radiomics']['auc']:.4f}")
    a0 = fus["A hand-crafted (BM1)"]["uniform"][0]
    e0 = fus["E BM1 + fmcib (thêm)"]["uniform"][0]
    print(f"ĐIỂM CHỐT fusion: BM1+fmcib {e0:.4f} vs BM1 {a0:.4f} -> Δ={e0-a0:+.4f}")


if __name__ == "__main__":
    main()
