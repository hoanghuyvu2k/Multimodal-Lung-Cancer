"""
Generate Fig: Baseline architecture — uniform_avg (no learned attention).

Same visual language as fig_attention_arch (A/B). Here the attention weights are
NOT learned: a_i = mask_i / sum(mask)  (equal division over available modalities).
Shares per-modality risk r_i = tanh(.) and output y_hat = sum_i r_i a_i.

Run from: paper/  directory
Output:   paper/figures/fig_uniform_arch.pdf  (+ .png preview)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import numpy as np

np.random.seed(3)

C = dict(
    m1='#1A73B5', m2='#1E8449', m3='#6C3483', m4='#B7460A',
    uni='#1F6F83', uni_l='#E3EEF1',          # baseline accent (teal)
    risk='#B7460A', risk_l='#FDEBD0',
    out='#2C3E50', gray='#2C3E50', lgray='#7F8C8D', line='#9AA4AE',
    white='#FFFFFF',
)
MCOL = [C['m1'], C['m2'], C['m3'], C['m4']]
N = 4

fig = plt.figure(figsize=(13.6, 4.95))
fig.patch.set_facecolor(C['white'])
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 13.6)
ax.set_ylim(0, 4.95)
ax.axis('off')


def rbox(x, y, w, h, fc, ec, lw=1.4, alpha=1.0, pad=0.08, zorder=2, ls='-'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad={pad}',
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=zorder,
        linestyle=ls))


def txt(x, y, s, fs=9, color=C['gray'], ha='center', va='center', zorder=9, **kw):
    return ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, zorder=zorder, **kw)


def arr(x1, y1, x2, y2, color=C['line'], lw=1.4, ms=12, conn=None, zorder=5):
    p = dict(arrowstyle='-|>', color=color, lw=lw, mutation_scale=ms)
    if conn:
        p['connectionstyle'] = conn
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=p, zorder=zorder)


def heat_bar(x, y, w, h, n, col, seed):
    rng = np.random.default_rng(seed)
    cmap = plt.get_cmap('Blues')
    for k in range(n):
        ax.add_patch(Rectangle((x + k*w/n, y), w/n, h,
                     fc=cmap(0.25 + 0.6*rng.random()), ec='white', lw=0.3, zorder=5))
    ax.add_patch(Rectangle((x, y), w, h, fc='none', ec=col, lw=1.5, zorder=6))


# ── stations ────────────────────────────────────────────────────────────────
IN_X   = 1.40
MASK_X = 3.75
NORM_X = 5.55
NORM_W = 0.72
A_X    = 7.35
MUL_X  = 8.45
R_X    = 9.55
OUT_X  = 11.75

accent = C['uni']
yc = 2.42
ys = [yc + 1.20, yc + 0.40, yc - 0.40, yc - 1.20]

# ── panel frame + title ──────────────────────────────────────────────────────
rbox(0.2, 0.35, 13.2, 4.30, C['uni_l'], accent, lw=1.8, pad=0.0, alpha=0.32, zorder=1)
txt(0.55, 4.38, 'C', fs=15, fontweight='bold', color=accent)
txt(1.05, 4.38, 'Baseline  —  uniform_avg  (no learned attention)',
    fs=11.5, fontweight='bold', color=accent, ha='left')

# ── inputs ───────────────────────────────────────────────────────────────────
for i, (yy, col) in enumerate(zip(ys, MCOL)):
    heat_bar(IN_X - 0.5, yy - 0.13, 1.0, 0.26, 8, col, seed=10*i + 2)
    txt(IN_X - 0.66, yy, f'$\\mathbf{{x}}_{i+1}$', fs=9.5, color=col, ha='right')
txt(IN_X, ys[0] + 0.60, 'Modality\nfeatures', fs=8, color=C['lgray'], style='italic')

# ── mask column (data availability, not learned) ─────────────────────────────
txt(MASK_X, ys[0] + 0.62, r'mask $m_i\in\{0,1\}$', fs=8.6, color=accent, fontweight='bold')
for k, yy in enumerate(ys):
    rbox(MASK_X - 0.30, yy - 0.18, 0.60, 0.36, C['uni_l'], accent, lw=1.3, pad=0.0, zorder=6)
    txt(MASK_X, yy, '$1$', fs=9, color=accent)
    arr(IN_X + 0.54, yy, MASK_X - 0.32, yy, color=C['line'], lw=1.1, ms=10)
txt(MASK_X, ys[-1] - 0.52, 'present $=1$, absent $=0$', fs=7.2, color=C['lgray'], style='italic')

# ── L1 norm box (of the mask) → equal weights ────────────────────────────────
nb_y0 = ys[-1] - 0.34
nb_h = (ys[0] - ys[-1]) + 0.68
rbox(NORM_X, nb_y0, NORM_W, nb_h, C['white'], C['gray'], lw=1.3, pad=0.0, zorder=4)
txt(NORM_X + NORM_W/2, yc, 'L1\nnorm', fs=8, color=C['gray'], fontweight='bold')
for yy in ys:
    arr(MASK_X + 0.30, yy, NORM_X - 0.04, yy, color=accent, lw=1.1, ms=10)

# ── attention a_i = mask_i / sum(mask)  (equal, NOT learned) ──────────────────
txt(A_X, ys[0] + 0.78, r'$a_i=m_i/\sum_j m_j$', fs=8.8, color=accent, fontweight='bold')
txt(A_X, ys[0] + 0.50, 'equal — not learned', fs=7.4, color=C['lgray'], style='italic')
for yy in ys:
    rbox(A_X - 0.40, yy - 0.18, 0.80, 0.36, C['uni_l'], accent, lw=1.3, pad=0.0, zorder=6)
    txt(A_X, yy, r'$1/N$', fs=8.6, color=accent)
    arr(NORM_X + NORM_W + 0.04, yy, A_X - 0.42, yy, color=accent, lw=1.3, ms=11)

# ── risk r_i column ──────────────────────────────────────────────────────────
txt(R_X, ys[0] + 0.62, r'$r_i=\tanh(\mathbf{W}_{r,i}^{\top}\mathbf{x}_i)$', fs=8.2, color=C['risk'])
for k, yy in enumerate(ys):
    rbox(R_X - 0.36, yy - 0.18, 0.72, 0.36, C['risk_l'], C['risk'], lw=1.3, pad=0.0, zorder=6)
    txt(R_X, yy, f'$r_{k+1}$', fs=9, color=C['risk'])

# ── multiply + output ────────────────────────────────────────────────────────
for yy in ys:
    ax.add_patch(Circle((MUL_X, yy), 0.13, fc='white', ec=C['gray'], lw=1.2, zorder=7))
    txt(MUL_X, yy, r'$\times$', fs=9, color=C['gray'])
    arr(A_X + 0.40, yy, MUL_X - 0.14, yy, color=accent, lw=1.0, ms=8)
    arr(R_X - 0.36, yy, MUL_X + 0.14, yy, color=C['risk'], lw=1.0, ms=8)

ax.add_patch(Circle((OUT_X, yc), 0.36, fc=C['out'], ec='white', lw=1.6, zorder=7))
txt(OUT_X, yc, r'$\hat{y}$', fs=13, color='white', fontweight='bold')
txt(OUT_X, yc - 0.66, r'$\hat{y}=\sum_i r_i a_i$', fs=8.6, color=C['out'])
for yy in ys:
    arr(MUL_X + 0.14, yy, OUT_X - 0.34, yc, color=C['lgray'], lw=0.8, ms=8)

# ── caption comparing to attention ───────────────────────────────────────────
txt(6.8, 0.16,
    'Same head $r_i$ and output as A/B, but attention degenerates to $1/\\sum m$ '
    '(equal division over available modalities); a missing modality ($m_i=0$) '
    'drops out automatically.  Illustrated for $N=4$; supports up to $N=6$.',
    fs=7.6, color=C['lgray'], style='italic')

plt.savefig('figures/fig_uniform_arch.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('figures/fig_uniform_arch.png', dpi=170, bbox_inches='tight', facecolor='white')
print('Saved: figures/fig_uniform_arch.pdf  +  .png')
