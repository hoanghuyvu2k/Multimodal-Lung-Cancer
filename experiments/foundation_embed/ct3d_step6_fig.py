"""Step 6 Phase 3 — hình 3 model x (discovery-CV, external)."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.evaluate import RESULTS_DIR                         # noqa: E402

DOC = Path(__file__).resolve().parents[2] / "document" / "2026-07-22_ct-foundation-embedding"


def main():
    p = json.load(open(RESULTS_DIR / "fm_ct.json", encoding="utf-8"))
    p3 = json.load(open(RESULTS_DIR / "fm_ct3d.json", encoding="utf-8"))
    models = ["radiomics", "BiomedCLIP", "MedicalNet 3D"]
    disc = [p["discovery_cv"]["radiomics"]["mean"], p["discovery_cv"]["ct_embed"]["mean"],
            p3["discovery_cv"]["mean"]]
    ext = [p["external"]["radiomics"]["auc"], p["external"]["ct_embed"]["auc"], p3["external"]["auc"]]
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    x = np.arange(3); w = 0.38
    ax.bar(x - w / 2, disc, w, color="#227c9d", label="discovery-CV (nội bộ)")
    ax.bar(x + w / 2, ext, w, color="#d1751d", label="external (rad_valid)")
    for xi, (a, b) in enumerate(zip(disc, ext)):
        ax.text(xi - w / 2, a + 0.006, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, b + 0.006, f"{b:.3f}", ha="center", fontsize=8)
    ax.axhline(0.5, color="#999", lw=1, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("AUC", fontsize=9); ax.set_ylim(0.45, 0.70)
    ax.set_title("Foundation embed CT: nội bộ ≈ ngẫu nhiên → external chỉ là nhiễu", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(DOC / "fig-ct3-models.svg")
    print(f"-> {DOC / 'fig-ct3-models.svg'}")


if __name__ == "__main__":
    main()
