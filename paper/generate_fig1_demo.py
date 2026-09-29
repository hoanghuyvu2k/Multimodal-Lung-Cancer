"""
Fig 1 (DEMO STYLE) — Study overview redrawn in the schematic style of
images/demo-image.png (dashed panels (a)/(b)/(c), heavy black arrows, data
rendered as heatmap/matrix grids, white op-boxes with thin black borders,
coloured node graphs inside a dashed group box; structure is black/white,
colour is reserved for data tokens).  Larger fonts, no overlapping elements.

This does NOT overwrite the original generate_fig1.py / fig1_overview.pdf.
Run from: paper/  directory
Output:   paper/figures/fig1_overview_demo.pdf  (+ .png preview)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
from matplotlib.patches import (FancyBboxPatch, Circle, Ellipse, Polygon,
                                Rectangle)
import numpy as np

np.random.seed(7)

BLACK = '#111111'
GRAY = '#5D6D7E'
LGRAY = '#95A5A6'
FS = 1.9    # global font-scale factor (point size of every label)

C = dict(clin='#B7460A', gen='#6C3483', rad='#1A73B5', rad2='#2E9ED6',
         path='#1E8449', path2='#27AE60')

# Smaller physical canvas (aspect preserved so circles stay round).  Because the
# figure is placed at width=\textwidth in the paper, a smaller canvas is scaled
# UP less aggressively -> the drawn text renders larger on the printed page.
# (paper textwidth = 394.4pt; at this canvas width the on-page body-text-sized
# labels land close to the 10pt body font instead of ~4-6pt as before.)
# NOTE: figure WIDTH (and xlim) controls the on-page font size once placed at
# width=\textwidth; figure HEIGHT does not.  So the panels were made TALLER
# (more ylim, more figsize height) to give every row/box more breathing room,
# without shrinking the text.
fig = plt.figure(figsize=(10.5, 6.911))
fig.patch.set_facecolor('white')
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 19)
ax.set_ylim(0, 12.5)
ax.axis('off')


# ── demo-style primitives ────────────────────────────────────────────────────
def dashed_panel(x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02',
        facecolor='white', edgecolor=BLACK, linewidth=1.7,
        linestyle=(0, (6, 4)), zorder=1))
    ax.text(x + w / 2, y + h + 0.16, title, fontsize=9.5 * FS, fontweight='bold',
            color=BLACK, ha='center', va='bottom', zorder=10)


def opbox(x, y, w, h, label, fs=9.5, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.03',
        facecolor='white', edgecolor=BLACK, linewidth=lw, zorder=5))
    if label:
        ax.text(x + w / 2, y + h / 2, label, fontsize=fs * FS, color=BLACK,
                ha='center', va='center', zorder=8)


def barrow(x1, y1, x2, y2, conn=None, lw=2.8, ms=24, color=BLACK):
    props = dict(arrowstyle='-|>', color=color, lw=lw, mutation_scale=ms,
                 shrinkA=0, shrinkB=0)
    if conn:
        props['connectionstyle'] = conn
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=props, zorder=7)


def txt(x, y, s, fs=9, color=BLACK, ha='center', va='center', z=8, **kw):
    return ax.text(x, y, s, fontsize=fs * FS, color=color, ha=ha, va=va,
                   zorder=z, **kw)


def heat(x, y, nrow, ncol, cell, cmap, seed, ec_out=BLACK, lw_out=1.5):
    rng = np.random.default_rng(seed)
    d = rng.standard_normal((nrow, ncol))
    cm = plt.get_cmap(cmap)
    lo, hi = d.min(), d.max()
    for i in range(nrow):
        for j in range(ncol):
            t = (d[i, j] - lo) / (hi - lo + 1e-9)
            ax.add_patch(Rectangle((x + j * cell, y + (nrow - 1 - i) * cell),
                cell, cell, fc=cm(t), ec='white', lw=0.5, zorder=5))
    ax.add_patch(Rectangle((x, y), ncol * cell, nrow * cell, fc='none',
                 ec=ec_out, lw=lw_out, zorder=6))
    return ncol * cell, nrow * cell


def array_bar(x, y, n, cell_w, cell_h, cmap, seed, ec_out=BLACK, lw_out=1.4):
    """A single-row feature ARRAY (1-D vector), not a 2-D matrix -- vertically
    centred on y so it lines up with the icon on its left."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    cm = plt.get_cmap(cmap)
    lo, hi = v.min(), v.max()
    y0 = y - cell_h / 2
    for i in range(n):
        t = (v[i] - lo) / (hi - lo + 1e-9)
        ax.add_patch(Rectangle((x + i * cell_w, y0), cell_w, cell_h,
            fc=cm(t), ec='white', lw=0.6, zorder=5))
    ax.add_patch(Rectangle((x, y0), n * cell_w, cell_h, fc='none',
                 ec=ec_out, lw=lw_out, zorder=6))
    return n * cell_w, cell_h


# icons (compact, black/white with a colour accent)
def ico_people(cx, cy, col):
    for k, dx in enumerate((-0.32, 0, 0.32)):
        c = col if k != 1 else LGRAY
        ax.add_patch(Circle((cx + dx, cy + 0.18), 0.10, fc=c, ec='none', zorder=6))
        ax.add_patch(Polygon([(cx + dx - 0.15, cy - 0.22), (cx + dx + 0.15, cy - 0.22),
                     (cx + dx + 0.11, cy + 0.07), (cx + dx - 0.11, cy + 0.07)],
                     closed=True, fc=c, ec='none', zorder=6))


def ico_dna(cx, cy, col):
    t = np.linspace(0, 3.4 * np.pi, 90)
    yy = cy + (t / (3.4 * np.pi) - 0.5) * 1.0
    xa = cx + 0.18 * np.sin(t)
    xb = cx - 0.18 * np.sin(t)
    for i in range(0, len(t), 9):
        ax.plot([xa[i], xb[i]], [yy[i], yy[i]], color='#C9C9C9', lw=1.0, zorder=5)
    ax.plot(xa, yy, color=col, lw=2.2, zorder=6)
    ax.plot(xb, yy, color='#9B59B6', lw=2.2, zorder=6)


def ico_ct(cx, cy, lc):
    w, h = 0.74, 0.74
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
        boxstyle='round,pad=0.01', fc='#15202B', ec=GRAY, lw=1.0, zorder=5))
    ax.add_patch(Ellipse((cx - 0.15, cy), 0.27, 0.47, fc='#34404A', ec='none', zorder=6))
    ax.add_patch(Ellipse((cx + 0.15, cy), 0.27, 0.47, fc='#34404A', ec='none', zorder=6))
    ax.add_patch(Ellipse((cx + 0.13, cy + 0.05), 0.12, 0.15, fc=lc, ec='white',
                 lw=0.7, zorder=7))


def ico_tissue(cx, cy):
    ax.add_patch(FancyBboxPatch((cx - 0.40, cy - 0.38), 0.80, 0.76,
        boxstyle='round,pad=0.02', fc='#FBF5EF', ec='#B9A78F', lw=1.0, zorder=5))
    cols = ['#C8924B', '#B97A36', '#D8B07A', '#A8631F', '#CBA06B']
    for i in range(7):
        bx = cx + np.random.uniform(-0.27, 0.27)
        by = cy + np.random.uniform(-0.25, 0.25)
        ax.add_patch(Ellipse((bx, by), np.random.uniform(0.13, 0.24),
                     np.random.uniform(0.13, 0.24), angle=np.random.uniform(0, 180),
                     fc=cols[i % len(cols)], ec='none', alpha=0.85, zorder=6))


# ═════════════════════════════════════════════════════════════════════════════
# PANEL (a) — Multimodal data sources
# ═════════════════════════════════════════════════════════════════════════════
AX, AY, AW, AH = 0.35, 0.55, 6.35, 11.0
dashed_panel(AX, AY, AW, AH, '(a) Data sources')

mods = [
    ('clin', 'Clinical labs — 13 variables', 'people', 'YlOrBr'),
    ('gen', 'Genomics\n(NGS, MSK-IMPACT)', 'dna', 'Purples'),
    ('rad', 'CT radiomics (PyRadiomics)', 'ct', 'Blues'),
    ('path', 'Pathology — PD-L1 IHC', 'tissue', 'Greens'),
]
row_y = [9.20, 6.65, 4.10, 1.55]
ICON_BOX = 1.0  # uniform backdrop size so all four icons line up evenly
for (key, title, icon, cmap), ry in zip(mods, row_y):
    col = C[key]
    cy = ry + 0.55
    txt(AX + 0.30, ry + 1.42, title, fs=7.4, color=col, ha='left',
        fontweight='bold')
    icx = AX + 0.82
    # uniform light backdrop behind every icon -> the row reads as evenly
    # sized/aligned even though the icons themselves have different shapes
    ax.add_patch(FancyBboxPatch((icx - ICON_BOX / 2, cy - ICON_BOX / 2),
        ICON_BOX, ICON_BOX, boxstyle='round,pad=0.02', facecolor='#F4F5F6',
        edgecolor='#D5D8DC', linewidth=1.0, zorder=4))
    if icon == 'people':
        ico_people(icx, cy, col)
    elif icon == 'dna':
        ico_dna(icx, cy, col)
    elif icon == 'ct':
        ico_ct(icx, cy, '#E74C3C')
    elif icon == 'tissue':
        ico_tissue(icx, cy)
    barrow(AX + 1.42, cy, AX + 1.92, cy, lw=2.2, ms=17)
    mx = AX + 2.10
    # 1-D feature ARRAY (not a matrix): a single row of coloured cells,
    # vertically centred on the icon so the row reads as aligned
    aw_, ah_ = array_bar(mx, cy, 8, 0.28, 0.62, cmap, seed=hash(key) % 1000)
    txt(mx + aw_ + 0.20, cy, 'feature\narray', fs=7.6, color=GRAY, ha='left')

txt(AX + AW / 2, AY + 0.32, 'per-modality feature blocks', fs=7.8, color=GRAY,
    style='italic')

# ═════════════════════════════════════════════════════════════════════════════
# PANEL (b) — DyAM attention fusion
# ═════════════════════════════════════════════════════════════════════════════
BX, BY, BW, BH = 7.25, 0.55, 6.85, 11.0
dashed_panel(BX, BY, BW, BH, '(b) DyAM attention fusion')  # centre panel

# DyAM chip
opbox(BX + BW / 2 - 0.95, BY + BH - 0.80, 1.9, 0.58, 'DyAM', fs=8)

# ── modality-input group box (dashed) holding [label — node] rows ────────────
MOD6 = [('CT · PC', C['rad']), ('CT · PL/LN', C['rad2']),
        ('Path · A', C['path']), ('Path · G', C['path2']),
        ('Genomics', C['gen']), ('Clin · NLP', C['clin'])]
gx, gw = BX + 0.20, 2.35
node_x = gx + gw - 0.42
ytop, ybot = BY + BH - 1.85, BY + 2.35
ny = np.linspace(ytop, ybot, len(MOD6))
ax.add_patch(FancyBboxPatch((gx, ybot - 0.55), gw, (ytop - ybot) + 1.10,
    boxstyle='round,pad=0.03', facecolor='white', edgecolor=GRAY, linewidth=1.4,
    linestyle=(0, (4, 3)), zorder=3))
txt(gx + gw / 2, ytop + 0.85, 'modality features', fs=6.2, color=GRAY,
    style='italic')
for yy, (lab, col) in zip(ny, MOD6):
    txt(gx + 0.18, yy, lab, fs=5.6, color=BLACK, ha='left')
    ax.add_patch(Circle((node_x, yy), 0.21, fc=col, ec=BLACK, lw=1.1, zorder=6))

# attention op-box (grayscale weight matrix)
attx, atty, aw, ah = BX + 3.55, BY + 3.60, 1.75, 3.4
opbox(attx, atty, aw, ah, '', lw=1.5)
txt(attx + aw / 2, atty + ah - 0.40, 'Attention', fs=6.0, fontweight='bold')
heat(attx + 0.38, atty + 0.66, 5, 5, 0.185, 'Greys', seed=42)
txt(attx + aw / 2, atty + 0.38, r'weights $a_i$', fs=6.0, color=GRAY, style='italic')

# faint connections nodes -> attention + one heavy summary arrow
for yy in ny:
    ax.plot([node_x + 0.21, attx], [yy, atty + ah / 2], color=LGRAY, lw=0.7,
            alpha=0.5, zorder=2)
barrow(node_x + 0.42, (ytop + ybot) / 2, attx - 0.02, atty + ah / 2, lw=2.4, ms=18)

# sum / response node
sumx = BX + BW - 1.05
sumy = atty + ah / 2
ax.add_patch(Circle((sumx, sumy), 0.40, fc=BLACK, ec='white', lw=1.5, zorder=6))
txt(sumx, sumy, r'$\Sigma$', fs=11, color='white')
barrow(attx + aw + 0.02, sumy, sumx - 0.42, sumy, lw=2.6, ms=19)
# formula placed well ABOVE the sum node -- clear of the attention box top
# and both arrows
txt(sumx, sumy + 2.20, r'$\hat{y}=\sum_i r_i a_i$', fs=7.0, color=BLACK)

# response chips
opbox(sumx - 0.70, BY + 1.45, 1.40, 0.55, 'PR / CR', fs=6.5)
opbox(sumx - 0.70, BY + 0.72, 1.40, 0.55, 'SD / PD', fs=6.5)
barrow(sumx, sumy - 0.46, sumx, BY + 2.0, lw=1.9, ms=14)

# caption sits BELOW both response chips (chip bottom = BY+0.72 = 1.27),
# well clear of the group box (bottom = 2.35) above it
txt(BX + BW / 2, BY + 0.46, 'cooperative: softplus, L1-normalized',
    fs=5.6, color=BLACK)
txt(BX + BW / 2, BY + 0.14, 'competitive (OvO): sigmoid, one-vs-others',
    fs=5.6, color=BLACK)

# ═════════════════════════════════════════════════════════════════════════════
# PANEL (c) — Response prediction & evaluation
# ═════════════════════════════════════════════════════════════════════════════
CX, CY, CW, CH = 14.65, 0.55, 4.0, 11.0
dashed_panel(CX, CY, CW, CH, '(c) Evaluation')

txt(CX + CW / 2, CY + CH - 0.62, 'prediction score\nmatrix', fs=7.5,
    fontweight='bold')
heat(CX + CW / 2 - 0.48, CY + CH - 2.70, 5, 4, 0.24, 'Greys', seed=3)

def mini_roc(cx, cy, w, h):
    ax.add_patch(Rectangle((cx, cy), w, h, fc='white', ec=BLACK, lw=1.2, zorder=5))
    ax.plot([cx, cx + w], [cy, cy + h], color=LGRAY, lw=0.9, ls='--', zorder=6)
    xs = np.linspace(0, 1, 40)
    ax.plot(cx + xs * w, cy + xs ** 0.32 * h, color=C['rad'], lw=2.4, zorder=7)

def mini_km(cx, cy, w, h):
    ax.add_patch(Rectangle((cx, cy), w, h, fc='white', ec=BLACK, lw=1.2, zorder=5))
    xs = np.linspace(0, 1, 30)
    ax.plot(cx + xs * w, cy + h * (1 - 0.28 * xs), color=C['rad'], lw=2.2, zorder=7)
    ax.plot(cx + xs * w, cy + h * (0.95 - 0.82 * xs), color=C['clin'], lw=2.2, zorder=7)

def mini_forest(cx, cy, w, h):
    ax.add_patch(Rectangle((cx, cy), w, h, fc='white', ec=BLACK, lw=1.2, zorder=5))
    ax.plot([cx + 0.30 * w, cx + 0.30 * w], [cy, cy + h], color=LGRAY, lw=0.9,
            ls='--', zorder=6)
    for k, yy in enumerate(np.linspace(cy + 0.22 * h, cy + 0.80 * h, 3)):
        x0 = cx + 0.35 * w + 0.05 * k * w
        ax.plot([x0, x0 + 0.40 * w], [yy, yy], color=C['path'], lw=2.0, zorder=7)
        ax.plot(x0 + 0.18 * w, yy, 'o', color=C['path'], ms=5,
                markeredgecolor='black', zorder=8)

# Titles wrapped to 2 lines (needed at this font size in the narrow column).
# The descriptive subtitles (ROC-AUC/DeLong, log-rank, hazard ratios) are
# dropped from the drawing itself -- that detail is already spelled out in
# the figure caption -- which frees the vertical room the bigger font needs.
evals = [
    ('Binary\nclassification', mini_roc),
    ('Survival —\nKaplan–Meier', mini_km),
    ('Cox proportional\nhazards', mini_forest),
]
heat_bottom = CY + CH - 2.70  # keep in sync with the heat() call above
block_h = 2.60
ey_top = [heat_bottom - 0.20 - i * block_h for i in range(3)]
for (title, draw), yt in zip(evals, ey_top):
    barrow(CX + CW / 2, yt, CX + CW / 2, yt - 0.22, lw=1.9, ms=14)
    txt(CX + CW / 2, yt - 0.75, title, fs=7.6, fontweight='bold')
    draw(CX + CW / 2 - 0.85, yt - 2.55, 1.7, 1.30)

# ── inter-panel heavy arrows ─────────────────────────────────────────────────
barrow(AX + AW + 0.05, AY + AH / 2, BX - 0.05, AY + AH / 2, lw=3.2, ms=28)
barrow(BX + BW + 0.05, BY + BH / 2, CX - 0.05, BY + BH / 2, lw=3.2, ms=28)

plt.savefig('figures/fig1_overview_demo.pdf', dpi=300, bbox_inches='tight',
            facecolor='white')
plt.savefig('figures/fig1_overview_demo.png', dpi=170, bbox_inches='tight',
            facecolor='white')
print('Saved: figures/fig1_overview_demo.pdf  +  .png')
