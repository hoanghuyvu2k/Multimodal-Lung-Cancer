"""Step 6 CT — hình discovery-CV vs external, CT-embed vs radiomics."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.evaluate import RESULTS_DIR                         # noqa: E402

DOC = Path(__file__).resolve().parents[2] / "document" / "2026-07-22_ct-foundation-embedding"
C_CT, C_RAD = "#d1751d", "#227c9d"


def main():
    d = json.load(open(RESULTS_DIR / "fm_ct.json", encoding="utf-8"))
    it = d["external_intersect"]
    ct = [d["discovery_cv"]["ct_embed"]["mean"], it["ct_embed"]["auc"]]
    ra = [d["discovery_cv"]["radiomics"]["mean"], it["radiomics"]["auc"]]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    x = np.arange(2); w = 0.36
    ax.bar(x - w / 2, ct, w, color=C_CT, label="CT-embed (BiomedCLIP)")
    ax.bar(x + w / 2, ra, w, color=C_RAD, label="radiomics thủ công")
    for xi, (c, r) in enumerate(zip(ct, ra)):
        ax.text(xi - w / 2, c + 0.006, f"{c:.3f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, r + 0.006, f"{r:.3f}", ha="center", fontsize=8)
    ax.axhline(0.5, color="#999", lw=1, ls="--")
    ax.text(1.5, 0.505, "ngẫu nhiên", fontsize=7, color="#999")
    ax.set_xticks(x); ax.set_xticklabels(["discovery-CV", f"external (n={it['n']})"], fontsize=9)
    ax.set_ylabel("AUC", fontsize=9); ax.set_ylim(0.45, 0.65)
    ax.set_title("CT-embed (BiomedCLIP) ≈ ngẫu nhiên nội bộ — KHÔNG dùng được", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(DOC / "fig-ct-vs-radiomics.svg")
    print(f"-> {DOC / 'fig-ct-vs-radiomics.svg'}")


if __name__ == "__main__":
    main()
