"""Step 6 — hình: discovery-CV vs external, Phikon vs GLCM."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.evaluate import RESULTS_DIR                         # noqa: E402

DOC = Path(__file__).resolve().parents[2] / "document" / "2026-07-21_foundation-embedding"
C_PH, C_GL = "#7b4ea3", "#227c9d"


def main():
    d = json.load(open(RESULTS_DIR / "fm_pathology.json", encoding="utf-8"))
    it = d["external_intersect"]
    ph = [d["discovery_cv"]["phikon"]["mean"], it["phikon"]["auc"]]
    gl = [d["discovery_cv"]["glcm"]["mean"], it["glcm"]["auc"]]

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    x = np.arange(2)
    w = 0.36
    ax.bar(x - w / 2, ph, w, color=C_PH, label="Phikon embed")
    ax.bar(x + w / 2, gl, w, color=C_GL, label="GLCM thủ công")
    for xi, (p, g) in enumerate(zip(ph, gl)):
        ax.text(xi - w / 2, p + 0.008, f"{p:.3f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, g + 0.008, f"{g:.3f}", ha="center", fontsize=8)
    ax.axhline(0.5, color="#bbb", lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(["discovery-CV\n(nội bộ)", f"external\n(n={it['n']})"], fontsize=9)
    ax.set_ylabel("AUC", fontsize=9)
    ax.set_ylim(0.5, 0.85)
    ax.set_title("Phikon giàu tín hiệu nội bộ hơn, nhưng KHÔNG generalize hơn GLCM", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(DOC / "fig-phikon-vs-glcm.svg")
    print(f"-> {DOC / 'fig-phikon-vs-glcm.svg'}")


if __name__ == "__main__":
    main()
