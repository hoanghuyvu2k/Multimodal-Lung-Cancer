"""Step 4+5 Phase 3 — 3D MedicalNet embed vs radiomics vs BiomedCLIP: discovery-CV + external."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data_setup import load_all                          # noqa: E402
from common.evaluate import RESULTS_DIR, save_result            # noqa: E402
from validation.run_validation import valid_labels             # noqa: E402
from foundation_embed.ct_step45_eval import cv_auc, ext_auc    # noqa: E402

DS = Path(__file__).resolve().parents[2].parent / "datasets"


def main():
    ctx = load_all()
    y = ctx.df_outcomes["label"]
    rad_lab, _ = valid_labels()

    d = pd.read_parquet(DS / "ct3d_fm_embed_discovery.parquet")
    v = pd.read_parquet(DS / "ct3d_fm_embed_rad_valid.parquet")
    d = d[d.index.isin(y.index)]
    yd = y.loc[d.index]
    cv = v.index.intersection(rad_lab.index.astype(str))
    v_c, yv = v.loc[cv], rad_lab.loc[cv].values

    out = {}
    out["discovery_cv"] = dict(zip(("mean", "sd"), cv_auc(d, yd)))
    a, ci, n = ext_auc(d, yd.values, v_c, yv)
    out["external"] = {"auc": a, "ci": ci, "n": n}
    save_result("fm_ct3d", out)

    # đối chiếu 3 model (radiomics + BiomedCLIP từ fm_ct.json)
    prev = json.load(open(RESULTS_DIR / "fm_ct.json", encoding="utf-8"))
    it = prev["external_intersect"]
    print(f"\n{'=' * 78}\n3 MODEL — discovery-CV & external (rad_valid)\n{'=' * 78}")
    print(f"{'model':<22}{'discovery-CV':>16}{'external':>26}")
    rows = [
        ("radiomics thủ công", prev["discovery_cv"]["radiomics"], prev["external"]["radiomics"]),
        ("BiomedCLIP 2.5D", prev["discovery_cv"]["ct_embed"], prev["external"]["ct_embed"]),
        ("MedicalNet 3D (mới)", out["discovery_cv"], out["external"]),
    ]
    for name, dc, ex in rows:
        dcv = f"{dc['mean']:.4f}±{dc['sd']:.4f}"
        ext = f"{ex['auc']:.4f} [{ex['ci'][0]:.2f},{ex['ci'][1]:.2f}]"
        print(f"{name:<22}{dcv:>16}{ext:>26}")
    print("=" * 78)
    print(f"\nĐIỂM CHỐT external: MedicalNet 3D {out['external']['auc']:.4f} vs "
          f"radiomics {prev['external']['radiomics']['auc']:.4f} vs BiomedCLIP {prev['external']['ct_embed']['auc']:.4f}")
    print(f"discovery-CV: MedicalNet 3D {out['discovery_cv']['mean']:.4f} vs radiomics "
          f"{prev['discovery_cv']['radiomics']['mean']:.4f} (>0.51 BiomedCLIP?)")


if __name__ == "__main__":
    main()
