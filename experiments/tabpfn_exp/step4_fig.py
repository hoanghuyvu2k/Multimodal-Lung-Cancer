"""Step 4 — hình tổng hợp TabPFN vs LR/uniform (per-modality + fusion)."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.evaluate import RESULTS_DIR                         # noqa: E402

DOC = Path(__file__).resolve().parents[2] / "document" / "2026-07-22_tabpfn"


def main():
    single = json.load(open(RESULTS_DIR / "tabpfn_single.json", encoding="utf-8"))
    fus = json.load(open(RESULTS_DIR / "tabpfn_fusion.json", encoding="utf-8"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    # panel 1: per-modality LR vs TabPFN
    mods = list(single); x = np.arange(len(mods)); w = 0.38
    ax1.bar(x - w / 2, [single[m]["lr"][0] for m in mods], w, color="#227c9d", label="LR")
    ax1.bar(x + w / 2, [single[m]["tabpfn"][0] for m in mods], w, color="#d1751d", label="TabPFN")
    ax1.set_xticks(x); ax1.set_xticklabels([m.replace("cnl_", "").replace("_", "\n")[:10] for m in mods],
                                           fontsize=7)
    ax1.axhline(0.5, color="#999", lw=1, ls="--")
    ax1.set_ylim(0.45, 0.75); ax1.set_ylabel("AUC", fontsize=9)
    ax1.set_title("Per-modality: TabPFN ≤ LR (0/5)", fontsize=9)
    ax1.legend(fontsize=8, frameon=False); ax1.spines[["top", "right"]].set_visible(False)
    # panel 2: fusion vs uniform_avg
    bms = list(fus); x2 = np.arange(len(bms))
    ax2.bar(x2 - w / 2, [fus[b]["uniform_avg_locked"] for b in bms], w, color="#227c9d", label="uniform_avg")
    ax2.bar(x2 + w / 2, [fus[b]["tabpfn_fusion"][0] for b in bms], w, color="#d1751d", label="TabPFN-fusion")
    ax2.set_xticks(x2); ax2.set_xticklabels(bms, fontsize=8)
    ax2.set_ylim(0.6, 0.80); ax2.set_ylabel("AUC", fontsize=9)
    ax2.set_title("Fusion: TabPFN thắng 1/4 (chỉ BM4)", fontsize=9)
    ax2.legend(fontsize=8, frameon=False); ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(DOC / "fig-tabpfn.svg")
    print(f"-> {DOC / 'fig-tabpfn.svg'}")


if __name__ == "__main__":
    main()
