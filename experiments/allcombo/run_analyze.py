"""Step 5 — Phân tích 26 combo: uniform_avg vs DyAM. Δ theo k + hình + quyết định."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.evaluate import load_result, save_result           # noqa: E402

DOC = Path(__file__).resolve().parents[2] / "document" / "2026-07-21_uniform-dyam-allcombo"
C_UNI, C_DYAM, C_NEUT = "#227c9d", "#d1751d", "#9aa0a6"  # xanh / cam / xám


def main():
    p = load_result("allcombo")
    combos = p["combos"]
    rows = sorted(combos.values(), key=lambda e: (e["k"], e["code"]))

    print(f"\n{'=' * 84}\n26 COMBO — uniform_avg vs DyAM (mean±sd 5 seed)\n{'=' * 84}")
    print(f"{'combo':<8}{'k':>2}{'domains':<24}{'uniform':>10}{'DyAM':>10}{'Δ(u-d)':>9}{'p':>7}{'win':>6}")
    for e in rows:
        pv, d = e["paired"]["p_value"], e["delta_mean"]
        win = "" if pv >= 0.05 else ("UNI" if d > 0 else "DYAM")
        print(f"{e['code']:<8}{e['k']:>2}{'+'.join(e['domains']):<24}"
              f"{e['uniform']['mean_auc']:>10.4f}{e['dyam']['mean_auc']:>10.4f}"
              f"{d:>+9.4f}{pv:>7.3f}{win:>6}")

    # Tổng hợp theo k
    print(f"\n{'=' * 84}\nΔ(uniform − DyAM) THEO k (câu hỏi chính)\n{'=' * 84}")
    print(f"{'k':>3}{'#combo':>8}{'mean Δ':>10}{'median Δ':>10}{'#uni_win':>10}{'#dyam_win':>11}")
    by_k = {}
    for e in rows:
        by_k.setdefault(e["k"], []).append(e)
    summary = {}
    for k in sorted(by_k):
        es = by_k[k]
        deltas = [e["delta_mean"] for e in es]
        uw = sum(1 for e in es if e["paired"]["p_value"] < 0.05 and e["delta_mean"] > 0)
        dw = sum(1 for e in es if e["paired"]["p_value"] < 0.05 and e["delta_mean"] < 0)
        summary[k] = {"n": len(es), "mean_delta": float(np.mean(deltas)),
                      "median_delta": float(np.median(deltas)), "uni_wins": uw, "dyam_wins": dw}
        print(f"{k:>3}{len(es):>8}{np.mean(deltas):>+10.4f}{np.median(deltas):>+10.4f}{uw:>10}{dw:>11}")

    tot_uni = sum(s["uni_wins"] for s in summary.values())
    tot_dyam = sum(s["dyam_wins"] for s in summary.values())
    n_uni_ahead = sum(1 for e in rows if e["delta_mean"] > 0)
    print("-" * 84)
    print(f"Tổng: {len(rows)} combo | uniform nhỉnh mean ở {n_uni_ahead}/{len(rows)} | "
          f"thắng có ý nghĩa: uniform {tot_uni}, DyAM {tot_dyam}")
    print("=" * 84)

    payload = {"summary_by_k": summary,
               "total": {"n_combo": len(rows), "uniform_ahead_mean": n_uni_ahead,
                         "uniform_sig_wins": tot_uni, "dyam_sig_wins": tot_dyam,
                         "dyam_win_combos": [e["code"] for e in rows
                                             if e["paired"]["p_value"] < 0.05 and e["delta_mean"] < 0]}}
    save_result("allcombo_analysis", payload)

    # Hình a: Δ vs k
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.axhline(0, color=C_NEUT, lw=1.2)
    for e in rows:
        jit = (hash(e["code"]) % 100 / 100 - 0.5) * 0.28
        d = e["delta_mean"]
        sig = e["paired"]["p_value"] < 0.05
        col = C_UNI if d > 0 else C_DYAM
        ax.scatter(e["k"] + jit, d, s=48 if sig else 30, color=col,
                   edgecolor="white" if sig else "none", linewidth=1.1,
                   marker="o" if d > 0 else "s", zorder=3, alpha=0.9)
    ks = sorted(summary)
    ax.plot(ks, [summary[k]["mean_delta"] for k in ks], "-", color="#333",
            lw=2, zorder=4, marker="D", ms=6, label="mean Δ theo k")
    ax.set_xticks(ks)
    ax.set_xlabel("số nguồn dữ liệu k", fontsize=9)
    ax.set_ylabel("ΔAUC (uniform_avg − DyAM)", fontsize=9)
    ax.set_title("Δ>0: uniform thắng · viền trắng = p<0.05 · mean Δ tăng theo k", fontsize=8.8)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(DOC / "fig-delta-by-k.svg")
    plt.close(fig)
    print(f"   -> {DOC / 'fig-delta-by-k.svg'}")

    # Hình b: scatter uniform vs DyAM AUC
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    lo, hi = 0.62, 0.79
    ax.plot([lo, hi], [lo, hi], "--", color=C_NEUT, lw=1.2, zorder=1)
    cmap = plt.get_cmap("viridis")
    for e in rows:
        c = cmap((e["k"] - 2) / 3)
        ax.scatter(e["dyam"]["mean_auc"], e["uniform"]["mean_auc"], s=42, color=c,
                   edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("DyAM AUC", fontsize=9)
    ax.set_ylabel("uniform_avg AUC", fontsize=9)
    ax.set_title("Trên đường chéo = uniform thắng (màu = k)", fontsize=8.8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(2, 5))
    cb = fig.colorbar(sm, ax=ax, ticks=[2, 3, 4, 5], fraction=0.046, pad=0.04)
    cb.set_label("k", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(DOC / "fig-scatter-uni-dyam.svg")
    plt.close(fig)
    print(f"   -> {DOC / 'fig-scatter-uni-dyam.svg'}")


if __name__ == "__main__":
    main()
