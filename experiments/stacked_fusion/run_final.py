"""Step 6 — Tổng hợp: bảng cuối + 2 hình + quyết định GIỮ/XÓA.

Bảng: uniform_avg | unif-ens | original | ovo | lr_late | stacked | stacked-ens (BM1-4).
Hình a: forest ΔAUC(stacked - uniform_avg) với 95% CI (paired bootstrap, từ stacked.json).
Hình b: bar trọng số meta mean|coef| per modality BM1 (radiomics vs non-rad).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import benchmarks as B                              # noqa: E402
from common.evaluate import load_result, load_scores           # noqa: E402

DOC = Path(__file__).resolve().parents[2] / "document" / "2026-07-21_stacked-fusion"
BMS = list(B.BENCHMARKS)

# CVD-safe: radiomics = cam (#d1751d), non-rad = xanh mòng két (#227c9d), stacked = tím (#7b4ea3)
C_RAD, C_NON, C_STK, C_NEUT = "#d1751d", "#227c9d", "#7b4ea3", "#9aa0a6"


def per_seed_stats(method, bm):
    d = load_scores(method, bm)
    y = d["_labels"].values.astype(float)
    cols = d["_scores"].values
    aucs = [roc_auc_score(y, cols[:, j]) for j in range(cols.shape[1])]
    return float(np.mean(aucs)), float(np.std(aucs, ddof=1))


def final_table():
    ens = load_result("ensemble")["methods"]
    stk = load_result("stacked")
    methods = ["uniform_avg", "original", "ovo", "lr_late", "stacked"]
    print(f"\n{'=' * 96}\nBẢNG CUỐI — AUC nội bộ (mean±sd 5 seed) và ensemble\n{'=' * 96}")
    hdr = f"{'method':<14}" + "".join(f"{bm:>10}" for bm in BMS) + "   |" + \
          "".join(f"{bm + '·ens':>10}" for bm in BMS)
    print(hdr)
    for m in methods:
        cells, ecells = [], []
        for bm in BMS:
            mu, sd = per_seed_stats(m, bm)
            cells.append(f"{mu:.3f}±{sd:.3f}")
            if m == "stacked":
                e = stk["ensemble"][bm]["stacked"]["ensemble_auc"]
            else:
                e = ens[m][bm]["ensemble_auc"]
            ecells.append(f"{e:.4f}")
        print(f"{m:<14}" + "".join(f"{c:>10}" for c in cells) + "   |" +
              "".join(f"{c:>10}" for c in ecells))
    print("=" * 96)


def fig_forest():
    stk = load_result("stacked")["benchmarks"]
    deltas = [stk[bm]["compare"]["uniform_avg"]["delta_auc"] for bm in BMS]
    los = [stk[bm]["compare"]["uniform_avg"]["ci_low"] for bm in BMS]
    his = [stk[bm]["compare"]["uniform_avg"]["ci_high"] for bm in BMS]
    ps = [stk[bm]["compare"]["uniform_avg"]["p_value"] for bm in BMS]
    ypos = np.arange(len(BMS))[::-1]

    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    ax.axvline(0, color=C_NEUT, lw=1.2, zorder=1)
    for y, d, lo, hi in zip(ypos, deltas, los, his):
        ax.plot([lo, hi], [y, y], color=C_STK, lw=2, zorder=2, solid_capstyle="round")
        ax.plot(d, y, "o", ms=8, color=C_STK, mec="white", mew=1.2, zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{bm}\n{B.BENCHMARKS[bm]['name'][:16]}" for bm in BMS], fontsize=8)
    for y, d, p in zip(ypos, deltas, ps):
        ax.text(0.062, y, f"Δ={d:+.3f} (p={p:.2f})", va="center", fontsize=7.5, color="#333")
    ax.set_xlim(-0.085, 0.13)
    ax.set_xlabel("ΔAUC (stacked − uniform_avg), 95% CI paired bootstrap", fontsize=8.5)
    ax.set_title("Stacked KHÔNG khác uniform_avg có ý nghĩa ở mọi BM (CI phủ 0)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(DOC / "fig-forest-delta.svg")
    plt.close(fig)
    print(f"   -> {DOC / 'fig-forest-delta.svg'}")


def fig_weights():
    w = load_result("stacked_weights")["benchmarks"]["BM1"]
    mods = w["modalities"]
    vals = [w["weights"][m]["mean_abs"] for m in mods]
    sds = [w["weights"][m]["sd"] for m in mods]
    colors = [C_RAD if m.startswith("rad_") else C_NON for m in mods]
    order = np.argsort(vals)
    mods = [mods[i] for i in order]
    vals = [vals[i] for i in order]
    sds = [sds[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    y = np.arange(len(mods))
    ax.barh(y, vals, color=colors, height=0.66, xerr=sds,
            error_kw=dict(ecolor="#666", elinewidth=1, capsize=2))
    ax.set_yticks(y)
    ax.set_yticklabels(mods, fontsize=8)
    for yi, v in zip(y, vals):
        ax.text(v + 0.03, yi, f"{v:.2f}", va="center", fontsize=7.5, color="#333")
    ax.set_xlabel("mean |meta-LR coef|  (độ ảnh hưởng của modality)", fontsize=8.5)
    ax.set_title("Meta ĐỀ CAO radiomics (cam), HẠ pathology (xanh, generalize) — trái kỳ vọng",
                 fontsize=8.8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_RAD, label="radiomics (ext ≤0.46)"),
                       Patch(color=C_NON, label="non-rad (pathology ext 0.77)")],
              fontsize=7.5, loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max(vals) + 0.25)
    fig.tight_layout()
    fig.savefig(DOC / "fig-meta-weights.svg")
    plt.close(fig)
    print(f"   -> {DOC / 'fig-meta-weights.svg'}")


if __name__ == "__main__":
    final_table()
    print("\nHÌNH:")
    fig_forest()
    fig_weights()
