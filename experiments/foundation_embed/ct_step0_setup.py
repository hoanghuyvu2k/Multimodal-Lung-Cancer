"""Step 0 CT — mapping volume->patient->cohort, verify BiomedCLIP, chọn series khớp mask geometry."""

import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.evaluate import RESULTS_DIR                         # noqa: E402

DS = Path(__file__).resolve().parents[2].parent / "datasets"
VOLS = DS / "radiology" / "LUNG_18-193" / "volumes"
SEGS = DS / "radiology" / "LUNG_18-193" / "segmentations"


def build_map():
    omni = pd.read_csv(
        DS / "18193mskmindprojectm-omnibusinventory_data_2021-12-20_1540-with-tb-and-scanner.csv",
        low_memory=False, dtype=str)
    coh = pd.read_csv(DS / "final_cohort_listing.csv")
    vols = [d for d in os.listdir(VOLS) if (VOLS / d).is_dir()]
    omni["acc"] = omni["did_acc"].astype(str).str.replace(r"\.0$", "", regex=True)
    s2dmp = omni.dropna(subset=["dmp_pt_id"]).drop_duplicates("acc").set_index("acc")["dmp_pt_id"]
    disc = set(coh[coh.cohort == "discovery"]["main_index"].astype(str))
    rv = set(coh[coh.cohort == "rad_valid"]["main_index"].astype(str))
    rows = []
    for v in vols:
        if v in rv:
            rows.append({"acc": v, "patient": v, "cohort": "rad_valid"})
        elif v in s2dmp.index and s2dmp[v] in disc:
            rows.append({"acc": v, "patient": s2dmp[v], "cohort": "discovery"})
        else:
            rows.append({"acc": v, "patient": s2dmp.get(v, ""), "cohort": "unmapped"})
    return pd.DataFrame(rows)


def seg_path(acc):
    for suf in ("nh", "amp"):
        p = SEGS / f"{acc}_{suf}.mha"
        if p.exists():
            return p
    c = list(SEGS.glob(f"{acc}_*.mha"))
    return c[0] if c else None


def match_series(acc):
    """Chọn series .nii có size trùng mask."""
    import SimpleITK as sitk
    sp = seg_path(acc)
    if sp is None:
        return {"error": "no_seg"}
    mask = sitk.ReadImage(str(sp))
    msize = mask.GetSize()
    import numpy as np
    marr = sitk.GetArrayFromImage(mask)
    niis = list((VOLS / acc).glob("SCANS/*/*_volumetric_image.nii"))
    matched = None
    for n in niis:
        img = sitk.ReadImage(str(n))
        if img.GetSize() == msize:
            matched = n.parent.name
            break
    return {"seg": sp.name, "mask_size": list(msize), "mask_spacing": list(mask.GetSpacing()),
            "n_series": len(niis), "matched_series": matched,
            "lesion_voxels": int((marr > 0).sum()), "mask_labels": [int(x) for x in np.unique(marr)[:6]]}


def verify_biomedclip():
    import open_clip
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, prep = open_clip.create_model_from_pretrained(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    model = model.to(dev).eval()
    import torch as t
    with t.no_grad():
        dummy = t.zeros(1, 3, 224, 224).to(dev)
        d = model.encode_image(dummy).shape[-1]
    return {"device": dev, "embed_dim": int(d)}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mp = build_map()
    mp.to_csv(RESULTS_DIR / "ct_slide_map.csv", index=False)
    counts = mp["cohort"].value_counts().to_dict()
    print("volume theo cohort:", counts)
    setup = {"counts": counts}
    for name, fn in [("biomedclip", verify_biomedclip),
                     ("series_match", lambda: match_series(mp[mp.cohort == "discovery"]["acc"].iloc[0]))]:
        try:
            setup[name] = fn()
            print(f"{name} OK:", setup[name])
        except Exception as e:
            setup[name] = {"error": str(e)[:300]}
            print(f"{name} FAIL:", str(e)[:300])
    with open(RESULTS_DIR / "ct_setup.json", "w", encoding="utf-8") as fh:
        json.dump(setup, fh, indent=2, ensure_ascii=False)
    print("-> ghi ct_setup.json + ct_slide_map.csv")


if __name__ == "__main__":
    main()
