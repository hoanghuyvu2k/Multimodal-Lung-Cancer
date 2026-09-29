"""Step 1 — 21 tổ hợp bài báo × {base, +Labs, +NLP} × uniform_avg (5-seed + single-run). Lưu tăng dần."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.evaluate import load_result, save_result           # noqa: E402
from common.data_setup import load_all                          # noqa: E402
from nlp_verify.nlp_modality import add_nlp_modality            # noqa: E402
from nlp_verify.nlp_run import run_variant                      # noqa: E402

RAD = ["rad_lesion_pc", "rad_lesion_pl", "rad_lesion_ln"]
# 21 tổ hợp y như bang-so-sanh-paper-combos.md (label, modalities)
PAPER = [
    ("TMB", ["gen_driver_tmb"]),
    ("PDL1", ["cnl_pdl1_score"]),
    ("IHC-A", ["path_ihc_pdl1"]),
    ("Gen", ["gen_driver_mut_amp"]),
    ("Rad", RAD),
    ("Rad-LU", RAD + ["rad_lesion_lu"]),
    ("TMB+PDL1", ["gen_driver_tmb", "cnl_pdl1_score"]),
    ("PDL1+Gen", ["cnl_pdl1_score", "gen_driver_mut_amp"]),
    ("Rad+IHC-A", RAD + ["path_ihc_pdl1"]),
    ("Rad+IHC-G", RAD + ["path_ihc_glcm"]),
    ("Rad+Gen", RAD + ["gen_driver_mut_amp"]),
    ("IHC-A+Gen", ["path_ihc_pdl1", "gen_driver_mut_amp"]),
    ("IHC-G+Gen", ["path_ihc_glcm", "gen_driver_mut_amp"]),
    ("Rad+IHC-A+Gen", RAD + ["path_ihc_pdl1", "gen_driver_mut_amp"]),
    ("Rad+IHC-G+Gen", RAD + ["path_ihc_glcm", "gen_driver_mut_amp"]),
    ("Rad+IHC-A+MutAmp", RAD + ["path_ihc_pdl1", "gen_driver_non_tmb", "cnl_pdl1_score"]),
    ("Rad+IHC-A+Gen+PDL1", RAD + ["path_ihc_pdl1", "gen_driver_mut_amp", "cnl_pdl1_score"]),
    ("Rad+IHC-G+Gen+PDL1", RAD + ["path_ihc_glcm", "gen_driver_mut_amp", "cnl_pdl1_score"]),
    ("Rad+IHC-A+Gen+TMB+PDL1", RAD + ["path_ihc_pdl1", "gen_driver_non_tmb", "gen_driver_tmb", "cnl_pdl1_score"]),
    ("Rad+IHC-A+Gen+PDL1+Labs", RAD + ["path_ihc_pdl1", "gen_driver_mut_amp", "cnl_pdl1_score", "cnl_dem_labs"]),
    ("Rad+IHC-G+Gen+PDL1+Labs", RAD + ["path_ihc_glcm", "gen_driver_mut_amp", "cnl_pdl1_score", "cnl_dem_labs"]),
]


def main():
    ctx = load_all()
    add_nlp_modality(ctx)
    try:
        payload = load_result("nlp_combos")
    except Exception:
        payload = {"combos": {}}
    done = payload["combos"]

    print(f"{'#':>3} {'combo':<26}{'base':>16}{'+Labs':>16}{'+NLP':>16}  (5-seed; 1r trong json)")
    for i, (label, mods) in enumerate(PAPER, 1):
        if label in done:
            e = done[label]
        else:
            base = [m for m in mods if m != "cnl_dem_labs"]
            variants = {"base": base, "labs": base + ["cnl_dem_labs"], "nlp": base + ["cnl_nlp"]}
            e = {v: run_variant(m, ctx) for v, m in variants.items()}
            done[label] = e
            save_result("nlp_combos", payload)   # lưu tăng dần
        b, l, n = e["base"], e["labs"], e["nlp"]
        print(f"{i:>3} {label:<26}"
              f"{b['mean']:.4f}±{b['sd']:.3f}"[:15].rjust(16) +
              f"{l['mean']:.4f}±{l['sd']:.3f}"[:15].rjust(16) +
              f"{n['mean']:.4f}±{n['sd']:.3f}"[:15].rjust(16), flush=True)
    save_result("nlp_combos", payload)
    print("XONG Step 1")


if __name__ == "__main__":
    main()
