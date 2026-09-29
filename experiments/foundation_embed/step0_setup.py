"""Step 0 — Khung foundation-embedding: mapping slide->patient->cohort, verify Phikon/openslide/annotation.

Lưu mapping ra results/fm_slide_map.csv và setup ra results/fm_setup.json.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.evaluate import RESULTS_DIR                         # noqa: E402

BASE = Path(__file__).resolve().parents[2].parent / "datasets"
SLIDES = BASE / "pathology" / "LUNG_18-193" / "slides"
HALO = BASE / "pathology" / "LUNG_18-193" / "halo"


def build_map():
    omni = pd.read_csv(
        BASE / "18193mskmindprojectm-omnibusinventory_data_2021-12-20_1540-with-tb-and-scanner.csv",
        low_memory=False)
    coh = pd.read_csv(BASE / "final_cohort_listing.csv")
    slides = [f[:-4] for f in os.listdir(SLIDES) if f.endswith(".svs")]

    omni["sid"] = omni["slide_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    s2dmp = omni.dropna(subset=["dmp_pt_id"]).drop_duplicates("sid").set_index("sid")["dmp_pt_id"]
    disc = set(coh[coh["cohort"] == "discovery"]["main_index"].astype(str))
    pval = set(coh[coh["cohort"] == "path_valid"]["main_index"].astype(str))

    rows = []
    for s in slides:
        if s in pval:                       # path_valid: slide_id CHÍNH là patient key
            rows.append({"slide": s, "patient": s, "cohort": "path_valid"})
        elif s in s2dmp.index and s2dmp[s] in disc:   # discovery: qua dmp_pt_id
            rows.append({"slide": s, "patient": s2dmp[s], "cohort": "discovery"})
        else:
            rows.append({"slide": s, "patient": s2dmp.get(s, ""), "cohort": "unmapped"})
    return pd.DataFrame(rows)


def verify_phikon():
    import torch
    from transformers import AutoImageProcessor, AutoModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained("owkin/phikon")
    model = AutoModel.from_pretrained("owkin/phikon").to(dev).eval()
    dim = model.config.hidden_size
    return {"device": dev, "embed_dim": int(dim), "processor": type(proc).__name__}


def verify_slide(mp):
    import openslide
    s = mp[mp["cohort"] == "discovery"]["slide"].iloc[0]
    sl = openslide.OpenSlide(str(SLIDES / f"{s}.svs"))
    props = dict(sl.properties)
    return {
        "slide": s, "n_levels": sl.level_count,
        "level0_dims": sl.level_dimensions[0],
        "objective_power": props.get("openslide.objective-power", "NA"),
        "mpp_x": props.get("openslide.mpp-x", "NA"),
    }


def peek_annotation(mp):
    s = mp[mp["cohort"] == "discovery"]["slide"].iloc[0]
    cand = list(HALO.glob(f"{s}_*.annotations"))
    if not cand:
        return {"found": False}
    txt = cand[0].read_text(errors="ignore")[:400]
    return {"found": True, "file": cand[0].name, "head": txt}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mp = build_map()
    mp.to_csv(RESULTS_DIR / "fm_slide_map.csv", index=False)
    counts = mp["cohort"].value_counts().to_dict()
    print("slide theo cohort:", counts)

    setup = {"slide_counts": counts,
             "n_discovery_patients": int(mp[mp.cohort == "discovery"]["patient"].nunique()),
             "n_pathval_patients": int(mp[mp.cohort == "path_valid"]["patient"].nunique())}
    try:
        setup["phikon"] = verify_phikon()
        print("Phikon OK:", setup["phikon"])
    except Exception as e:
        setup["phikon"] = {"error": str(e)[:300]}
        print("Phikon FAIL:", str(e)[:300])
    try:
        setup["slide"] = verify_slide(mp)
        print("Slide OK:", setup["slide"])
    except Exception as e:
        setup["slide"] = {"error": str(e)[:300]}
        print("Slide FAIL:", str(e)[:300])
    try:
        setup["annotation"] = peek_annotation(mp)
        print("Annotation:", setup["annotation"].get("found"),
              setup["annotation"].get("head", "")[:150])
    except Exception as e:
        setup["annotation"] = {"error": str(e)[:300]}

    with open(RESULTS_DIR / "fm_setup.json", "w", encoding="utf-8") as fh:
        json.dump(setup, fh, indent=2, ensure_ascii=False)
    print("-> ghi fm_setup.json + fm_slide_map.csv")


if __name__ == "__main__":
    main()
