"""Generate Figure 3 (AUC comparison bar charts) and Figure 5 (survival
analysis summary 2x2) from the numbers reported in
paper/tables/table2_auc_ihca.tex, table3_auc_ihcg.tex, table4_survival.tex
and code/excel/survival_analysis_summary.xlsx.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # paper/
FIG_DIR = os.path.join(BASE, "figures")

# ─────────────────────────────────────────────────────────────────────────
# Figure 3 — AUC comparison bar charts
# ─────────────────────────────────────────────────────────────────────────

# IHC-A arm: 7 variants (Table 2)
ihca_labels = [
    "No clinical", "+ Labs", "+ NLP raw", "+ NLP-PCA16",
    "+ Labs + NLP", "+ Labs +\nNLP-PCA16", "+ BioClinBERT-\nPCA16",
]
ihca_auc = [0.764, 0.768, 0.781, 0.784, 0.767, 0.753, 0.767]
ihca_lo  = [0.695, 0.700, 0.717, 0.719, 0.700, 0.682, 0.701]
ihca_hi  = [0.833, 0.837, 0.845, 0.848, 0.835, 0.823, 0.833]
ihca_colors = ["grey", "orange", "#66c2a5", "#1b9e77", "#a6d96a", "#66bd63", "purple"]

# IHC-G arm: 4 variants (Table 3)
ihcg_labels = ["No clinical", "+ Labs", "+ NLP raw", "+ NLP-PCA16"]
ihcg_auc = [0.784, 0.788, 0.813, 0.812]
ihcg_lo  = [0.717, 0.723, 0.753, 0.751]
ihcg_hi  = [0.850, 0.853, 0.874, 0.873]
ihcg_colors = ["grey", "orange", "#1b9e77", "#66c2a5"]


def auc_bar(ax, labels, auc, lo, hi, colors, title, best_idx=None):
    x = np.arange(len(labels))
    auc = np.array(auc)
    lo = np.array(lo)
    hi = np.array(hi)
    err = np.vstack([auc - lo, hi - auc])
    bars = ax.bar(x, auc, yerr=err, capsize=4, color=colors,
                   edgecolor="black", linewidth=0.6)
    ax.axhline(0.80, ls="--", color="black", lw=1, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.65, 0.92)
    ax.set_title(title, fontsize=10, fontweight="bold")
    if best_idx is not None:
        ax.text(best_idx, hi[best_idx] + 0.01, r"$\dagger$",
                ha="center", fontsize=12, fontweight="bold")
    return bars


fig, ax = plt.subplots(figsize=(6, 4))
auc_bar(ax, ihca_labels, ihca_auc, ihca_lo, ihca_hi, ihca_colors,
        "IHC-A arm (7 variants)", best_idx=3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig3a_auc_ihca.pdf"))
plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 4))
auc_bar(ax, ihcg_labels, ihcg_auc, ihcg_lo, ihcg_hi, ihcg_colors,
        "IHC-G arm (4 variants)", best_idx=2)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig3b_auc_ihcg.pdf"))
plt.close(fig)

print("Figure 3 done")

# ─────────────────────────────────────────────────────────────────────────
# Figure 5 — Survival analysis summary (Table 4 / survival_analysis_summary.xlsx)
# ─────────────────────────────────────────────────────────────────────────

models = [
    "No Clinical (IHC-A)", "+ Labs (IHC-A)", "+ NLP raw (IHC-A)", "+ NLP-PCA16 (IHC-A)",
    "No Clinical (IHC-G)", "+ Labs (IHC-G)", "+ NLP raw (IHC-G)", "+ NLP-PCA16 (IHC-G)",
]
arm = ["IHC-A"] * 4 + ["IHC-G"] * 4

cindex     = [0.623, 0.626, 0.625, 0.628, 0.626, 0.632, 0.632, 0.631]
cindex_lo  = [0.580, 0.583, 0.583, 0.584, 0.583, 0.587, 0.591, 0.590]
cindex_hi  = [0.665, 0.669, 0.663, 0.668, 0.665, 0.675, 0.672, 0.672]

tdauc_6  = [0.717, 0.713, 0.715, 0.719, 0.722, 0.711, 0.720, 0.721]
tdauc_12 = [0.718, 0.707, 0.739, 0.742, 0.691, 0.686, 0.699, 0.694]
tdauc_18 = [0.700, 0.662, 0.718, 0.720, 0.667, 0.649, 0.665, 0.656]

cox_hr    = [5.30, 5.31, 5.91, 6.07, 4.74, 4.49, 4.97, 4.69]
cox_hr_lo = [3.10, 2.88, 3.40, 3.50, 2.82, 2.56, 2.98, 2.86]
cox_hr_hi = [9.07, 9.79, 10.25, 10.52, 7.97, 7.89, 8.31, 7.68]

ibs = [0.1745, 0.1739, 0.1723, 0.1716, 0.1748, 0.1746, 0.1736, 0.1736]

arm_colors = {"IHC-A": "#4393c3", "IHC-G": "#d6604d"}
bar_colors = [arm_colors[a] for a in arm]

# 5A — C-index
fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(models))
err = np.vstack([np.array(cindex) - np.array(cindex_lo),
                  np.array(cindex_hi) - np.array(cindex)])
ax.bar(x, cindex, yerr=err, capsize=4, color=bar_colors, edgecolor="black", linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=35, ha="right", fontsize=7.5)
ax.set_ylabel("Harrell's C-index")
ax.set_ylim(0.50, 0.75)
ax.axhline(0.5, ls="--", color="grey", lw=1)
handles = [plt.Rectangle((0, 0), 1, 1, color=arm_colors[a]) for a in ["IHC-A", "IHC-G"]]
ax.legend(handles, ["IHC-A", "IHC-G"], fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig5a_cindex.pdf"))
plt.close(fig)

# 5B — Time-dependent AUC
fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), sharey=True)
times = [6, 12, 18]
variant_labels = ["No clinical", "+ Labs", "+ NLP raw", "+ NLP-PCA16"]
variant_styles = ["o-", "s--", "^-.", "d:"]
for ax, a, idxs in zip(axes, ["IHC-A", "IHC-G"], [range(0, 4), range(4, 8)]):
    for vi, i in enumerate(idxs):
        vals = [tdauc_6[i], tdauc_12[i], tdauc_18[i]]
        ax.plot(times, vals, variant_styles[vi], label=variant_labels[vi])
    ax.set_title(a, fontsize=10, fontweight="bold")
    ax.set_xlabel("Time (months)")
    ax.set_xticks(times)
axes[0].set_ylabel("Time-dependent AUC")
axes[0].set_ylim(0.60, 0.78)
axes[1].legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig5b_tdauc.pdf"))
plt.close(fig)

# 5C — Forest plot of Cox HR
fig, ax = plt.subplots(figsize=(6, 4))
y = np.arange(len(models))[::-1]
for yi, i in zip(y, range(len(models))):
    ax.plot([cox_hr_lo[i], cox_hr_hi[i]], [yi, yi], color=bar_colors[i], lw=2)
    ax.plot(cox_hr[i], yi, "o", color=bar_colors[i], markersize=7,
            markeredgecolor="black")
ax.axvline(1, ls="--", color="black", lw=1)
ax.set_xscale("log")
ax.set_yticks(y)
ax.set_yticklabels(models, fontsize=8)
ax.set_xlabel("Hazard ratio (95% CI), log scale")
ax.set_xlim(1, 15)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig5c_cox_forest.pdf"))
plt.close(fig)

# 5D — Integrated Brier Score
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(x, ibs, color=bar_colors, edgecolor="black", linewidth=0.6)
ax.axhline(0.25, ls="--", color="black", lw=1, label="Random-guess (0.25)")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=35, ha="right", fontsize=7.5)
ax.set_ylabel("Integrated Brier Score")
ax.set_ylim(0, 0.30)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig5d_ibs.pdf"))
plt.close(fig)

print("Figure 5 done")
