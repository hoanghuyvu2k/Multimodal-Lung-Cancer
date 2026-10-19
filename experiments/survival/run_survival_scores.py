"""W4 bước 1 — lưu điểm nguy cơ ngoài-fold của 11 biến thể x 5 seed (DyAM, cùng harness verify_nlp_5seed) để tính tdAUC ở venv numpy>=2."""
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXP)); sys.path.insert(0, str(_EXP.parent))

import pandas as pd                                           # noqa: E402
import torch                                                  # noqa: E402

from nlp_verify import verify_nlp_5seed as V                  # noqa: E402
from nlp_verify.verify_nlp_variants import ARMS, VARIANTS, add_bioclinbert, add_pca16, slug   # noqa: E402
from common.data_setup import load_all                        # noqa: E402
from nlp_verify.nlp_modality import add_nlp_modality          # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "survival_scores"

if __name__ == "__main__":
    torch.set_num_threads(3)
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = load_all(); add_nlp_modality(ctx)
    add_pca16(ctx, "cnl_nlp", "cnl_nlp_pca16"); add_bioclinbert(ctx); add_pca16(ctx, "cnl_bioclin", "cnl_bioclin_pca16")
    clin = V.load_clinical()
    for arm, arm_mod in ARMS.items():
        for name, extra, no_scale in VARIANTS:
            if (arm, name) not in V.PAPER_AUC:
                continue
            f = OUT / f"{arm}__{slug(name)}.csv"
            if f.exists():
                continue
            r = V.run_5seed(ctx, clin, arm_mod, extra, no_scale)
            df = r["_scores"].copy(); df["label"] = r["_labels"]
            df.to_csv(f)
            print(f"[surv_scores] {arm} {name}: saved", flush=True)
    print("XONG")
