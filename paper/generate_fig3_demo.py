"""
Fig 3 (DEMO STYLE) — AUC comparison bar charts restyled to match the cohesive
demo aesthetic of Fig 1/2 (serif typography, full black frame, modality-colour
bars with black edges, dashed reference line, value labels, dagger on best).

Does NOT overwrite make_fig3_fig5.py / fig3a_auc_ihca.pdf / fig3b_auc_ihcg.pdf.
Run from: paper/  directory
Output:   paper/figures/fig3a_auc_ihca_demo.pdf, fig3b_auc_ihcg_demo.pdf
          + combined fig3_auc_demo.png preview
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
import numpy as np
from matplotlib.transforms import blended_transform_factory

BLACK = '#111111'

# ── data (identical to make_fig3_fig5.py / Tables 2 & 3) ─────────────────────
ihca_labels = ["No clinical", "+ Labs", "+ NLP raw", "+ NLP-PCA16",
               "+ Labs + NLP", "+ Labs +\nNLP-PCA16", "+ BioClinBERT-\nPCA16"]
ihca_auc = [0.764, 0.768, 0.781, 0.784, 0.767, 0.753, 0.767]
ihca_lo  = [0.695, 0.700, 0.717, 0.719, 0.700, 0.682, 0.701]
ihca_hi  = [0.833, 0.837, 0.845, 0.848, 0.835, 0.823, 0.833]
ihca_colors = ["#7F8C8D", "#B7460A", "#2E9ED6", "#1A73B5", "#27AE60",
               "#1E8449", "#6C3483"]

ihcg_labels = ["No clinical", "+ Labs", "+ NLP raw", "+ NLP-PCA16"]
ihcg_auc = [0.784, 0.788, 0.813, 0.812]
ihcg_lo  = [0.717, 0.723, 0.753, 0.751]
ihcg_hi  = [0.850, 0.853, 0.874, 0.873]
ihcg_colors = ["#7F8C8D", "#B7460A", "#1A73B5", "#2E9ED6"]


def auc_bar(ax, labels, auc, lo, hi, colors, title, best_idx):
    x = np.arange(len(labels))
    auc, lo, hi = map(np.array, (auc, lo, hi))
    err = np.vstack([auc - lo, hi - auc])
    ax.bar(x, auc, yerr=err, capsize=5, color=colors, edgecolor=BLACK,
           linewidth=1.1, width=0.72,
           error_kw=dict(ecolor=BLACK, elinewidth=1.4, capthick=1.4))
    ax.axhline(0.80, ls='--', color=BLACK, lw=1.2, alpha=0.75)
    # reference-line label placed just OUTSIDE the right spine, on the line
    # (x in axes fraction, y in data units) -> never overlaps any bar/label
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(1.015, 0.80, 'AUC\n= 0.80', fontsize=9.5, ha='left', va='center',
            color=BLACK, style='italic', transform=trans)
    # value labels on top of each bar
    for xi, a in zip(x, auc):
        ax.text(xi, a + 0.005, f'{a:.3f}', ha='center', va='bottom',
                fontsize=9, color=BLACK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=11)
    ax.set_ylabel('AUC', fontsize=13)
    ax.set_ylim(0.65, 0.92)
    ax.set_title(title, fontsize=13.5, fontweight='bold', color=BLACK)
    # full black frame (all four spines), like the demo op-boxes
    for sp in ax.spines.values():
        sp.set_edgecolor(BLACK)
        sp.set_linewidth(1.3)
    ax.tick_params(colors=BLACK, labelsize=11)
    ax.text(best_idx, hi[best_idx] + 0.014, r'$\dagger$', ha='center',
            fontsize=16, fontweight='bold', color=BLACK)


# ── individual PDFs (drop-in replacements for the subfloats) ──────────────────
# Smaller canvases -> placed at 0.48\textwidth they are scaled up less, so the
# axis/label/value fonts render noticeably larger on the printed page.
fig, ax = plt.subplots(figsize=(5.4, 4.3))
auc_bar(ax, ihca_labels, ihca_auc, ihca_lo, ihca_hi, ihca_colors,
        'IHC-A arm (7 variants)', best_idx=3)
fig.tight_layout()
fig.savefig('figures/fig3a_auc_ihca_demo.pdf', bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(4.6, 4.3))
auc_bar(ax, ihcg_labels, ihcg_auc, ihcg_lo, ihcg_hi, ihcg_colors,
        'IHC-G arm (4 variants)', best_idx=2)
fig.tight_layout()
fig.savefig('figures/fig3b_auc_ihcg_demo.pdf', bbox_inches='tight')
plt.close(fig)

# ── combined side-by-side preview (a)/(b) ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5.6),
                         gridspec_kw={'width_ratios': [7, 4]})
auc_bar(axes[0], ihca_labels, ihca_auc, ihca_lo, ihca_hi, ihca_colors,
        '(a)  IHC-A arm (7 variants)', best_idx=3)
auc_bar(axes[1], ihcg_labels, ihcg_auc, ihcg_lo, ihcg_hi, ihcg_colors,
        '(b)  IHC-G arm (4 variants)', best_idx=2)
fig.tight_layout()
fig.savefig('figures/fig3_auc_demo.png', dpi=170, bbox_inches='tight',
            facecolor='white')
plt.close(fig)

print('Saved: figures/fig3a_auc_ihca_demo.pdf, fig3b_auc_ihcg_demo.pdf, '
      'fig3_auc_demo.png')
