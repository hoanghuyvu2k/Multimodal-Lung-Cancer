"""Generate Supplementary Figure S4 — OvO vs Original AUC scatter, from
code/excel/ovo_comparison_full_results_origin.xlsx (the 20-configuration
comparison reported in Table tab:ovo / sections/results.tex).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.dirname(BASE)
SUPP_FIG_DIR = os.path.join(BASE, "supplementary", "figures")

df = pd.ExcelFile(os.path.join(CODE_DIR, "excel", "ovo_comparison_full_results_origin.xlsx")).parse("All_Results")

n_modalities = df["Test Case"].apply(lambda s: len(s.split("+")))


def color_for(n):
    if n == 1:
        return "grey"
    elif n == 2:
        return "#3182bd"  # blue
    else:
        return "#de2d26"  # red


colors = n_modalities.apply(color_for)

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(df["Original_AUC"], df["OvO_AUC"], c=colors, edgecolor="black",
           s=60, zorder=3)

lims = [0.59, 0.82]
ax.plot(lims, lims, "k--", lw=1, zorder=1)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("AUC, original cooperative attention")
ax.set_ylabel("AUC, OvO competitive attention")
ax.set_aspect("equal")

handles = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
               markeredgecolor="black", markersize=8, label="1 modality"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#3182bd",
               markeredgecolor="black", markersize=8, label="2 modalities"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#de2d26",
               markeredgecolor="black", markersize=8, label="≥ 3 modalities"),
]
ax.legend(handles=handles, fontsize=8, loc="upper left")

fig.tight_layout()
fig.savefig(os.path.join(SUPP_FIG_DIR, "figS4_ovo_scatter.pdf"))
plt.close(fig)
print("Figure S4 done")
