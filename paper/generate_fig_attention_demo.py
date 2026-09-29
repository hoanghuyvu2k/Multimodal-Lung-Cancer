"""
Fig 3 (DEMO STYLE) -- Architecture of the two attention mechanisms, restyled
to match the demo aesthetic used for Fig 1/2/4 (images/demo-image.png):
black DASHED rounded panels with a bold serif "(a)/(b)" title, heavy black
arrows, white op-boxes with thin black borders, data (modality feature bars,
the NxN score matrix, the per-modality score circles) keeping colour while
every structural element (boxes, arrows, panel frame) is black/white.

Does NOT overwrite generate_fig_attention.py / fig_attention_arch.pdf.
Run from: paper/  directory
Output:   paper/figures/fig_attention_arch_demo.pdf  (+ .png preview)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import numpy as np

np.random.seed(3)

BLACK = '#111111'
GRAY = '#5D6D7E'
LGRAY = '#95A5A6'
FS = 2.1    # global font-scale factor (point size of every label)

# modality colours match Fig 1's colour coding (rad/path/gen/clin)
C = dict(
    m1='#1A73B5', m2='#1E8449', m3='#6C3483', m4='#B7460A',
    coop='#1E8449', ovo='#6C3483',
    risk='#B7460A', out=BLACK,
)
MCOL = [C['m1'], C['m2'], C['m3'], C['m4']]
N = 4

# Wider canvas than the first pass -- gives every column/box more breathing
# room (the previous version had bigger text but the same tight spacing,
# which read as cramped).  Width grew ~13%; FS was trimmed slightly so the
# on-page font size is about the same as before, and the extra room goes
# entirely into gaps between elements, not bigger boxes.
fig = plt.figure(figsize=(11.3, 6.452))
fig.patch.set_facecolor('white')
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 17.5)
ax.set_ylim(0, 10.0)
ax.axis('off')


def dashed_panel(x, y, w, h, letter, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02',
        facecolor='white', edgecolor=BLACK, linewidth=1.9,
        linestyle=(0, (6, 4)), zorder=1))
    ax.text(x + 0.30, y + h + 0.14, f'({letter})  {title}', fontsize=10.2 * FS,
             fontweight='bold', color=BLACK, ha='left', va='bottom', zorder=10)


def rbox(x, y, w, h, fc, ec=BLACK, lw=1.4, pad=0.06, zorder=6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad={pad}',
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder))


def txt(x, y, s, fs=9, color=BLACK, ha='center', va='center', zorder=9, **kw):
    return ax.text(x, y, s, fontsize=fs * FS, color=color, ha=ha, va=va,
                   zorder=zorder, **kw)


def arr(x1, y1, x2, y2, color=BLACK, lw=1.6, ms=13, conn=None, zorder=5):
    p = dict(arrowstyle='-|>', color=color, lw=lw, mutation_scale=ms,
              shrinkA=0, shrinkB=0)
    if conn:
        p['connectionstyle'] = conn
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=p, zorder=zorder)


def heat_bar(x, y, w, h, n, seed):
    rng = np.random.default_rng(seed)
    cmap = plt.get_cmap('Blues')
    for k in range(n):
        ax.add_patch(Rectangle((x + k * w / n, y), w / n, h,
                     fc=cmap(0.25 + 0.6 * rng.random()), ec='white', lw=0.4,
                     zorder=5))
    ax.add_patch(Rectangle((x, y), w, h, fc='none', ec=BLACK, lw=1.3, zorder=6))


# ── evenly spaced horizontal stations (shared by both panels) ────────────────
# more room between every station than the first pass, so boxes/arrows/labels
# are not packed edge-to-edge
IN_X   = 1.55
NORM_X = 9.00
NORM_W = 0.92
A_X    = 11.05
MUL_X  = 12.35
R_X    = 13.60
OUT_X  = 16.10


def right_block(ys, yc, accent):
    """Shared right half: L1-norm -> a_i -> (x r_i) -> y_hat."""
    nb_y0 = ys[-1] - 0.34
    nb_h  = (ys[0] - ys[-1]) + 0.68
    rbox(NORM_X, nb_y0, NORM_W, nb_h, 'white', BLACK, lw=1.5, pad=0.0)
    txt(NORM_X + NORM_W / 2, yc, 'L1\nnorm', fs=6.8, color=BLACK,
        fontweight='bold')

    txt(R_X, ys[0] + 0.66, r'$r_i=\tanh(\cdot)$',
        fs=6.6, color=C['risk'])
    for k, yy in enumerate(ys):
        rbox(R_X - 0.38, yy - 0.19, 0.76, 0.38, 'white', BLACK, lw=1.3, pad=0.0)
        txt(R_X, yy, f'$r_{k+1}$', fs=7.6, color=C['risk'])

    txt(A_X, ys[0] + 0.66, r'attention $a_i$', fs=6.8, color=accent,
        fontweight='bold')
    for k, yy in enumerate(ys):
        rbox(A_X - 0.38, yy - 0.19, 0.76, 0.38, 'white', BLACK, lw=1.3, pad=0.0)
        txt(A_X, yy, f'$a_{k+1}$', fs=7.6, color=accent)

    for yy in ys:
        arr(NORM_X + NORM_W + 0.05, yy, A_X - 0.40, yy, color=BLACK, lw=1.5,
            ms=11)
        ax.add_patch(Circle((MUL_X, yy), 0.15, fc='white', ec=BLACK,
                     lw=1.3, zorder=7))
        txt(MUL_X, yy, r'$\times$', fs=7.6, color=BLACK)
        arr(A_X + 0.38, yy, MUL_X - 0.16, yy, color=accent, lw=1.1, ms=8)
        arr(R_X - 0.38, yy, MUL_X + 0.16, yy, color=C['risk'], lw=1.1, ms=8)

    ax.add_patch(Circle((OUT_X, yc), 0.40, fc=BLACK, ec='white', lw=1.7,
                 zorder=7))
    txt(OUT_X, yc, r'$\hat{y}$', fs=11, color='white', fontweight='bold')
    txt(OUT_X, yc - 0.72, r'$\hat{y}=\sum_i r_i a_i$', fs=7.2, color=BLACK)
    for yy in ys:
        arr(MUL_X + 0.16, yy, OUT_X - 0.38, yc, color=LGRAY, lw=0.8, ms=8)


def panel(ybox0, ybox1, yc, mode):
    accent = C['coop'] if mode == 'coop' else C['ovo']
    ys = [yc + 1.25, yc + 0.42, yc - 0.42, yc - 1.25]

    # ── inputs ───────────────────────────────────────────────────────────
    for i, (yy, col) in enumerate(zip(ys, MCOL)):
        heat_bar(IN_X - 0.55, yy - 0.14, 1.1, 0.28, 8,
                 seed=10 * i + (0 if mode == 'coop' else 5))
        txt(IN_X - 0.72, yy, f'$\\mathbf{{x}}_{i+1}$', fs=8.0, color=col,
            ha='right')
    txt(IN_X, ybox1 - 0.34, 'Modality\nfeatures', fs=6.6, color=GRAY,
        style='italic')

    if mode == 'coop':
        cell = 0.56
        gw = N * cell
        GX = 3.35
        GY = yc - gw / 2
        cmap = plt.get_cmap('YlGn')
        rng = np.random.default_rng(1)
        for r in range(N):
            for c in range(N):
                ax.add_patch(Rectangle((GX + c * cell, GY + (N - 1 - r) * cell),
                             cell, cell, fc=cmap(0.2 + 0.55 * rng.random()),
                             ec='white', lw=0.9, zorder=5))
        ax.add_patch(Rectangle((GX, GY), gw, gw, fc='none', ec=BLACK, lw=1.7,
                     zorder=6))
        txt(GX + gw / 2, GY + gw + 0.44, r'$N\times N$ linear $+$ softplus',
            fs=7.4, color=BLACK, fontweight='bold')
        txt(GX + gw / 2, GY - 0.60, 'row $=$ source $i$, col $=$ target $j$',
            fs=5.6, color=GRAY, style='italic')
        for yy in ys:
            arr(IN_X + 0.58, yy, GX - 0.06, yy, color=BLACK, lw=1.3, ms=10)

        SJ_X = 7.20
        txt(SJ_X, ys[0] + 0.66, r'$s_j=\sum_i \mathrm{softplus}$',
            fs=5.2, color=accent)
        for yy in ys:
            rbox(SJ_X - 0.36, yy - 0.19, 0.72, 0.38, 'white', BLACK, lw=1.3,
                 pad=0.0)
            txt(SJ_X, yy, '$s_j$', fs=7.2, color=accent)
            arr(GX + gw + 0.06, yy, SJ_X - 0.38, yy, color=BLACK, lw=1.3,
                ms=10)
            arr(SJ_X + 0.36, yy, NORM_X - 0.05, yy, color=BLACK, lw=1.3, ms=9)

    else:
        SX = 3.75
        for yy, col in zip(ys, MCOL):
            ax.add_patch(Circle((SX, yy), 0.26, fc='white', ec=BLACK, lw=1.6,
                         zorder=6))
            txt(SX, yy, '$s_i$', fs=6.8, color=col)
            arr(IN_X + 0.58, yy, SX - 0.28, yy, color=BLACK, lw=1.3, ms=10)
        txt(SX, ys[0] + 0.66, 'scores $s_i$', fs=6.8, color=GRAY,
            style='italic')
        txt(SX, ys[-1] - 0.60, r'$s_i=\mathrm{softplus}(\cdot)$',
            fs=6.2, color=GRAY)

        MX = 5.65
        ax.add_patch(Circle((MX, yc), 0.34, fc='white', ec=BLACK, lw=1.6,
                     zorder=6))
        txt(MX, yc, r'$\mu_{-i}$', fs=7.6, color=accent)
        txt(MX, yc + 0.95, 'mean of others', fs=6.0, color=GRAY,
            style='italic')
        for yy in ys:
            arr(SX + 0.28, yy, MX - 0.32, yc, color=LGRAY, lw=0.8, ms=7)

        CMPX = 7.55
        rbox(CMPX - 0.90, yc - 0.66, 1.80, 1.32, 'white', BLACK, lw=1.7,
             pad=0.05)
        txt(CMPX, yc + 0.30, r'$\sigma(s_i-\mu_{-i})$', fs=6.6, color=accent,
            fontweight='bold')
        txt(CMPX, yc - 0.24, 'one-vs-others,\ncomputed $\\forall i$', fs=5.4,
            color=BLACK, style='italic')
        arr(SX + 0.28, ys[0], CMPX - 0.92, yc + 0.30, color=MCOL[0], lw=1.5,
            ms=11, conn='arc3,rad=-0.18')
        arr(MX + 0.34, yc, CMPX - 0.92, yc - 0.24, color=accent, lw=1.3, ms=10)
        for yy in ys:
            arr(CMPX + 0.90, yc, NORM_X - 0.05, yy, color=BLACK, lw=1.3, ms=9)

    right_block(ys, yc, accent)


# ── panel frames + titles ────────────────────────────────────────────────
# Panel (a) is shifted up (extra gap above panel (b)'s dashed border) so the
# bigger (a)/(b) title text has room and never touches the panel-b frame.
dashed_panel(0.2, 5.45, 17.1, 4.35, 'a', 'Cooperative attention -- AttentionMatrix')
dashed_panel(0.2, 0.35, 17.1, 4.35, 'b', 'Competitive attention -- AttentionMatrixOvO (one-vs-others)')

panel(5.45, 9.80, 7.45, 'coop')
panel(0.35, 4.70, 2.45, 'ovo')

txt(8.75, 0.10, 'Illustrated for $N=4$ modalities; configurations in this study use up to $N=7$.',
    fs=6.4, color=GRAY, style='italic')

plt.savefig('figures/fig_attention_arch_demo.pdf', dpi=300, bbox_inches='tight',
            facecolor='white')
plt.savefig('figures/fig_attention_arch_demo.png', dpi=170, bbox_inches='tight',
            facecolor='white')
print('Saved: figures/fig_attention_arch_demo.pdf  +  .png')
