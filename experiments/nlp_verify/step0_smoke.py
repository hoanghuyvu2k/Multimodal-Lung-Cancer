"""Step 0 — build NLP cache + smoke BM1 (base / +Labs / +NLP) + đo thời gian."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data_setup import load_all                          # noqa: E402
from nlp_verify.nlp_modality import add_nlp_modality            # noqa: E402
from nlp_verify.nlp_run import run_variant                      # noqa: E402

RAD = ["rad_lesion_pc", "rad_lesion_pl", "rad_lesion_ln"]
BASE = RAD + ["path_ihc_glcm", "gen_driver_mut_amp", "cnl_pdl1_score"]  # BM1


def main():
    ctx = load_all()
    n = add_nlp_modality(ctx)
    print(f"NLP modality cnl_nlp: n={n}, dim={ctx.modality_dict['cnl_nlp'].shape[1]}")
    variants = {"base (BM1)": BASE, "+Labs": BASE + ["cnl_dem_labs"], "+NLP": BASE + ["cnl_nlp"]}
    print(f"\n{'variant':<14}{'5-seed':>16}{'single-run':>12}{'sec':>8}")
    for name, mods in variants.items():
        t = time.time()
        r = run_variant(mods, ctx)
        dt = time.time() - t
        fivs = f"{r['mean']:.4f}±{r['sd']:.4f}"
        print(f"{name:<14}{fivs:>16}{r['single_run']:>12.4f}{dt:>8.1f}", flush=True)


if __name__ == "__main__":
    main()
