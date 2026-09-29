"""Generate Figure 1 (study overview) and Figure 2 (NLP clinical encoding
pipeline) as simple matplotlib box-and-arrow diagrams, per
paper/drafts/figure_specs.md. These are schematic placeholders suitable for
compilation; a designer/illustrator may later replace them with polished
graphics.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

PAPER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(PAPER_DIR, "figures")


def box(ax, xy, w, h, text, fc="#e8f4f8", fontsize=8.5, ec="black"):
    b = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                        fc=fc, ec=ec, lw=1.0)
    ax.add_patch(b)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fontsize, wrap=True)
    return b


def arrow(ax, p1, p2):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                         color="black", lw=1.2)
    ax.add_patch(a)


# ─────────────────────────────────────────────────────────────────────────
# Figure 1 — Study overview (3 panels)
# ─────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
ax.axis("off")

# Panel A — Patient Cohort
box(ax, (0.2, 1.3), 2.7, 2.3,
    "A. Patient Cohort\n\n"
    "Discovery (n=247)\n"
    "ICI-treated NSCLC\n\n"
    "Validation:\n"
    "Radiomics (n=50)\n"
    "Pathology (n=71)",
    fc="#e8f4f8", fontsize=8)
box(ax, (0.3, 0.2), 2.5, 0.9,
    "Modalities: CT radiomics, pathology IHC\n"
    "(IHC-A/IHC-G), genomics, PD-L1, clinical labs",
    fc="#fefcea", fontsize=7)

arrow(ax, (2.95, 2.2), (3.5, 2.2))

# Panel B — DyAM Architecture
box(ax, (3.55, 0.8), 3.0, 2.9,
    "B. DyAM Architecture\n\n"
    "Per-modality risk score $r_i$\n"
    "+ Attention $a_i$\n\n"
    "Cooperative\n(AttentionMatrix,\nsoftplus + L1-norm)\n\n"
    "vs.\n\n"
    "Competitive (OvO)\n(AttentionMatrixOvO,\nsigmoid one-vs-others)",
    fc="#eaf7e8", fontsize=7.5)
box(ax, (4.1, 0.15), 1.9, 0.55,
    r"$\hat{y} = \sum_i r_i a_i$", fc="#ffffff", fontsize=10)

arrow(ax, (6.6, 2.2), (7.15, 2.2))

# Panel C — Evaluation
box(ax, (7.2, 0.8), 2.6, 2.9,
    "C. Evaluation\n\n"
    "• ROC curve (AUC,\n  DeLong 95% CI)\n\n"
    "• Harrell's C-index\n\n"
    "• Cox proportional\n  hazards (forest plot)\n\n"
    "• Kaplan--Meier\n  log-rank test",
    fc="#fdeef0", fontsize=7.5)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig1_overview.pdf"))
plt.close(fig)
print("Figure 1 done")

# ─────────────────────────────────────────────────────────────────────────
# Figure 2 — NLP clinical encoding pipeline
# ─────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3.2)
ax.axis("off")

steps = [
    "13 numeric\nclinical variables",
    "df_to_text_prompts()",
    "English text\nprompt",
    "all-MiniLM-L6-v2\n(sentence transformer)",
    "384-dim\nembedding\n(\"NLP raw\")",
    "PCA (n=16)",
    "16-dim\n\"NLP-PCA16\"",
]
n = len(steps)
box_w, gap = 1.18, 0.12
total_w = n * box_w + (n - 1) * gap
x0 = (10 - total_w) / 2
y0 = 1.6

for i, s in enumerate(steps):
    x = x0 + i * (box_w + gap)
    fc = "#e8f4f8" if i % 2 == 0 else "#eaf7e8"
    box(ax, (x, y0 - 0.55), box_w, 1.1, s, fc=fc, fontsize=6.8)
    if i < n - 1:
        arrow(ax, (x + box_w, y0), (x + box_w + gap, y0))

# Sidebar: example prompt
box(ax, (0.3, 0.05), total_w, 0.85,
    '"Patient is 69 years old. Smoking history in pack-years is 48. '
    'ECOG performance status is 1. Blood albumin concentration is 4.00. '
    'Derived neutrophil-to-lymphocyte ratio (dNLR) is 2.40 ... '
    'no liver metastases."',
    fc="#fefcea", fontsize=7.5)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig2_nlp_pipeline.pdf"))
plt.close(fig)
print("Figure 2 done")
