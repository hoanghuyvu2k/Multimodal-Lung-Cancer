"""
Generate Fig: Attention-mechanism architectures (evenly spaced redesign).
  (A) Cooperative  — AttentionMatrix    : N x N linear grid, softplus, L1-norm
  (B) Competitive  — AttentionMatrixOvO : sigmoid(s_i - mean(others)), L1-norm
Both share per-modality risk r_i = tanh(.) and output y_hat = sum_i r_i a_i.

Run from: paper/  directory
Output:   paper/figures/fig_attention_arch.pdf  (+ .png preview)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import numpy as np

np.random.seed(3)

C = dict(
    m1='#1A73B5', m2='#1E8449', m3='#6C3483', m4='#B7460A',
    coop='#1E8449', coop_l='#EAF7EF',
    ovo='#6C3483',  ovo_l='#F2EAF8',
    risk='#B7460A', risk_l='#FDEBD0',
    out='#2C3E50',
    gray='#2C3E50', lgray='#7F8C8D', line='#9AA4AE',
    white='#FFFFFF',
)
MCOL = [C['m1'], C['m2'], C['m3'], C['m4']]
N = 4

fig = plt.figure(figsize=(15.5, 9.6))
fig.patch.set_facecolor(C['white'])
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 15.5)
ax.set_ylim(0, 9.6)
ax.axis('off')

def rbox(x, y, w, h, fc, ec, lw=1.4, alpha=1.0, pad=0.08, zorder=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad={pad}',
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=zorder))

def txt(x, y, s, fs=9, color=C['gray'], ha='center', va='center', zorder=9, **kw):
    return ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va,
                   zorder=zorder, **kw)

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
                     fc=cmap(0.25 + 0.6*rng.random()), ec='white', lw=0.3,
                     zorder=5))
    ax.add_patch(Rectangle((x, y), w, h, fc='none', ec=col, lw=1.5, zorder=6))

# ── evenly spaced horizontal stations (shared by both panels) ────────────────
IN_X   = 1.45     # input feature bars
NORM_X = 7.95     # L1-norm box (left edge)
NORM_W = 0.72
A_X    = 9.75     # attention a_i boxes (centre)
MUL_X  = 10.85    # multiply node
R_X    = 11.85    # risk r_i boxes (centre)
OUT_X  = 13.85    # y_hat output

def right_block(ys, yc, accent):
    """Shared right half: L1-norm -> a_i -> (x r_i) -> y_hat."""
    # L1-norm box spanning all rows
    nb_y0 = ys[-1] - 0.34
    nb_h  = (ys[0] - ys[-1]) + 0.68
    rbox(NORM_X, nb_y0, NORM_W, nb_h, C['white'], C['gray'], lw=1.3, pad=0.0,
         zorder=4)
    txt(NORM_X + NORM_W/2, yc, 'L1\nnorm', fs=8, color=C['gray'],
        fontweight='bold')

    # risk r_i column + header
    txt(R_X, ys[0] + 0.62, r'$r_i=\tanh(\mathbf{W}_{r,i}^{\top}\mathbf{x}_i)$',
        fs=8.4, color=C['risk'])
    for k, yy in enumerate(ys):
        rbox(R_X - 0.36, yy - 0.18, 0.72, 0.36, C['risk_l'], C['risk'], lw=1.3,
             pad=0.0, zorder=6)
        txt(R_X, yy, f'$r_{k+1}$', fs=9, color=C['risk'])

    # attention a_i column + header
    txt(A_X, ys[0] + 0.62, r'attention $a_i$', fs=8.6, color=accent,
        fontweight='bold')
    for k, yy in enumerate(ys):
        rbox(A_X - 0.36, yy - 0.18, 0.72, 0.36, (C['coop_l'] if accent==C['coop']
             else C['ovo_l']), accent, lw=1.3, pad=0.0, zorder=6)
        txt(A_X, yy, f'$a_{k+1}$', fs=9, color=accent)

    # norm -> a_i, a_i x r_i, -> y_hat
    for yy in ys:
        arr(NORM_X + NORM_W + 0.04, yy, A_X - 0.38, yy, color=accent, lw=1.3,
            ms=11)
        ax.add_patch(Circle((MUL_X, yy), 0.13, fc='white', ec=C['gray'],
                     lw=1.2, zorder=7))
        txt(MUL_X, yy, r'$\times$', fs=9, color=C['gray'])
        arr(A_X + 0.36, yy, MUL_X - 0.14, yy, color=accent, lw=1.0, ms=8)
        arr(R_X - 0.36, yy, MUL_X + 0.14, yy, color=C['risk'], lw=1.0, ms=8)

    # output node
    ax.add_patch(Circle((OUT_X, yc), 0.36, fc=C['out'], ec='white', lw=1.6,
                 zorder=7))
    txt(OUT_X, yc, r'$\hat{y}$', fs=13, color='white', fontweight='bold')
    txt(OUT_X, yc - 0.66, r'$\hat{y}=\sum_i r_i a_i$', fs=8.6, color=C['out'])
    for yy in ys:
        arr(MUL_X + 0.14, yy, OUT_X - 0.34, yc, color=C['lgray'], lw=0.8, ms=8)


def panel(ybox0, ybox1, yc, mode):
    accent   = C['coop'] if mode == 'coop' else C['ovo']
    accent_l = C['coop_l'] if mode == 'coop' else C['ovo_l']
    ys = [yc + 1.25, yc + 0.42, yc - 0.42, yc - 1.25]

    # ── inputs ───────────────────────────────────────────────────────────
    for i, (yy, col) in enumerate(zip(ys, MCOL)):
        heat_bar(IN_X - 0.5, yy - 0.13, 1.0, 0.26, 8, col,
                 seed=10*i + (0 if mode == 'coop' else 5))
        txt(IN_X - 0.66, yy, f'$\\mathbf{{x}}_{i+1}$', fs=9.5, color=col,
            ha='right')
    txt(IN_X, ybox1 - 0.32, 'Modality\nfeatures', fs=8, color=C['lgray'],
        style='italic')

    if mode == 'coop':
        # ── N x N attention matrix ───────────────────────────────────────
        cell = 0.52
        gw = N * cell
        GX = 3.05
        GY = yc - gw/2
        cmap = plt.get_cmap('YlGn')
        rng = np.random.default_rng(1)
        for r in range(N):
            for c in range(N):
                ax.add_patch(Rectangle((GX + c*cell, GY + (N-1-r)*cell),
                             cell, cell, fc=cmap(0.2 + 0.55*rng.random()),
                             ec='white', lw=0.9, zorder=5))
        ax.add_patch(Rectangle((GX, GY), gw, gw, fc='none', ec=accent, lw=1.7,
                     zorder=6))
        txt(GX + gw/2, GY + gw + 0.42, r'$N\times N$ linear $+$ softplus',
            fs=8.8, color=accent, fontweight='bold')
        txt(GX + gw/2, GY - 0.34, 'row $=$ source $i$,  column $=$ target $j$',
            fs=7.3, color=C['lgray'], style='italic')
        # inputs -> matrix rows
        for yy in ys:
            arr(IN_X + 0.54, yy, GX - 0.06, yy, color=C['line'], lw=1.1, ms=10)

        # matrix -> s_j scores column
        SJ_X = 6.45
        txt(SJ_X, ys[0] + 0.62,
            r'$s_j=\sum_i \mathrm{softplus}(\cdot)/\|\mathbf{x}_i\|_2$',
            fs=7.8, color=accent)
        for yy in ys:
            rbox(SJ_X - 0.34, yy - 0.18, 0.68, 0.36, C['white'], accent,
                 lw=1.2, pad=0.0, zorder=6)
            txt(SJ_X, yy, '$s_j$', fs=8.5, color=accent)
            arr(GX + gw + 0.06, yy, SJ_X - 0.36, yy, color=accent, lw=1.1,
                ms=10)
            arr(SJ_X + 0.34, yy, NORM_X - 0.04, yy, color=accent, lw=1.0, ms=9)

    else:
        # ── OvO competitive ──────────────────────────────────────────────
        SX = 3.35
        for yy, col in zip(ys, MCOL):
            ax.add_patch(Circle((SX, yy), 0.24, fc='white', ec=col, lw=2.0,
                         zorder=6))
            txt(SX, yy, '$s_i$', fs=7.8, color=col)
            arr(IN_X + 0.54, yy, SX - 0.26, yy, color=C['line'], lw=1.1, ms=10)
        txt(SX, ys[0] + 0.62, 'scores $s_i$', fs=8.2, color=C['lgray'],
            style='italic')
        txt(SX, ys[-1] - 0.55,
            r'$s_i=\mathrm{softplus}(\mathbf{W}_{a,i}^{\top}\mathbf{x}_i)/\|\mathbf{x}_i\|_2$',
            fs=7.3, color=C['lgray'])

        # mean-of-others node
        MX = 4.85
        ax.add_patch(Circle((MX, yc), 0.32, fc=accent_l, ec=accent, lw=1.6,
                     zorder=6))
        txt(MX, yc, r'$\mu_{-i}$', fs=9, color=accent)
        txt(MX, yc + 0.56, 'mean of others', fs=7.2, color=C['lgray'],
            style='italic')
        for yy in ys:
            arr(SX + 0.26, yy, MX - 0.30, yc, color=C['line'], lw=0.7, ms=7)

        # competitive comparator
        CMPX = 6.45
        rbox(CMPX - 0.72, yc - 0.62, 1.44, 1.24, accent_l, accent, lw=1.7,
             pad=0.04, zorder=6)
        txt(CMPX, yc + 0.26, r'$\sigma(s_i-\mu_{-i})$', fs=9, color=accent,
            fontweight='bold')
        txt(CMPX, yc - 0.12, 'one-vs-others', fs=7.6, color=C['gray'],
            style='italic')
        txt(CMPX, yc - 0.40, r'computed $\forall i$', fs=7.2, color=C['lgray'],
            style='italic')
        # highlighted s_1 and mu feed the comparator
        arr(SX + 0.26, ys[0], CMPX - 0.74, yc + 0.28, color=MCOL[0], lw=1.5,
            ms=11, conn='arc3,rad=-0.18')
        arr(MX + 0.32, yc, CMPX - 0.74, yc - 0.22, color=accent, lw=1.3, ms=10)
        # comparator -> norm (per row)
        for yy in ys:
            arr(CMPX + 0.72, yc, NORM_X - 0.04, yy, color=accent, lw=1.0, ms=9)

    right_block(ys, yc, accent)


# ── panel frames + titles ────────────────────────────────────────────────
rbox(0.2, 5.05, 15.1, 4.35, C['coop_l'], C['coop'], lw=1.8, pad=0.0, alpha=0.32,
     zorder=1)
txt(0.55, 9.15, 'A', fs=15, fontweight='bold', color=C['coop'])
txt(3.7, 9.15, 'Cooperative attention  —  AttentionMatrix',
    fs=11.5, fontweight='bold', color=C['coop'], ha='left')

rbox(0.2, 0.35, 15.1, 4.35, C['ovo_l'], C['ovo'], lw=1.8, pad=0.0, alpha=0.32,
     zorder=1)
txt(0.55, 4.45, 'B', fs=15, fontweight='bold', color=C['ovo'])
txt(3.7, 4.45, 'Competitive attention  —  AttentionMatrixOvO (one-vs-others)',
    fs=11.5, fontweight='bold', color=C['ovo'], ha='left')

panel(5.05, 9.40, 7.05, 'coop')
panel(0.35, 4.70, 2.45, 'ovo')

txt(7.75, 0.12, 'Illustrated for $N=4$ modalities; the model supports up to $N=6$.',
    fs=7.8, color=C['lgray'], style='italic')

plt.savefig('figures/fig_attention_arch.pdf', dpi=300, bbox_inches='tight',
            facecolor='white')
plt.savefig('figures/fig_attention_arch.png', dpi=170, bbox_inches='tight',
            facecolor='white')
print('Saved: figures/fig_attention_arch.pdf  +  .png')
