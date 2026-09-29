"""
Generate Fig 1: Study Overview (graphical-abstract style)
Modelled on the reference graphical abstract (figures/model.jpg):
  - iconographic data-source tiles (people / DNA / CT / tissue)
  - an explicit DyAM neural-network diagram (input -> attention -> response)
Professional scientific figure for the MDPI Cancers paper.

Run from: paper/  directory
Output:   paper/figures/fig1_overview.pdf  (+ .png preview)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Ellipse, Polygon, Rectangle, FancyArrowPatch
import numpy as np

np.random.seed(7)

# ── Palette ──────────────────────────────────────────────────────────────────
C = dict(
    clin='#B7460A',  clin_l='#FDEBD0',
    gen='#6C3483',   gen_l='#EAD9F5',
    rad='#1A73B5',   rad_l='#DDEEFF',
    rad2='#2E9ED6',
    path='#1E8449',  path_l='#D5F5E3',
    path2='#27AE60',
    teal='#0E6655',
    gray='#2C3E50',  lgray='#7F8C8D',
    line='#AEB6BF',
    white='#FFFFFF',
    panelA='#F4F8FC', panelB='#F3FBF5', panelC='#FFF8F1',
    edgeA='#4A7BA8', edgeB='#1E8449', edgeC='#B7460A',
)

# Input modality order (top -> bottom in the network) and their colours
MOD = [
    ('CT  ·  primary (PC)',  C['rad']),
    ('CT  ·  pleural/nodal',  C['rad2']),
    ('Pathology IHC-A',       C['path']),
    ('Pathology IHC-G',       C['path2']),
    ('Genomics (NGS)',        C['gen']),
    ('Clinical  (NLP)',       C['clin']),
]

fig = plt.figure(figsize=(17, 9))
fig.patch.set_facecolor(C['white'])
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 17)
ax.set_ylim(0, 9)
ax.axis('off')

# ── Generic helpers ──────────────────────────────────────────────────────────
def rbox(x, y, w, h, fc, ec, lw=1.5, alpha=1.0, pad=0.15, zorder=2, ls='-'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad={pad}',
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=zorder,
        linestyle=ls))

def txt(x, y, s, fs=9, color=C['gray'], ha='center', va='center', zorder=8, **kw):
    return ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va,
                   zorder=zorder, **kw)

def thick_arr(x1, y1, x2, y2, color, lw=3, ms=24):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                                mutation_scale=ms), zorder=6)

# ── ICONS ────────────────────────────────────────────────────────────────────
def person(cx, cy, s, color, z=5):
    ax.add_patch(Circle((cx, cy + 0.42*s), 0.26*s, fc=color, ec='none', zorder=z))
    verts = [(cx - 0.40*s, cy - 0.55*s), (cx + 0.40*s, cy - 0.55*s),
             (cx + 0.30*s, cy + 0.18*s), (cx - 0.30*s, cy + 0.18*s)]
    ax.add_patch(Polygon(verts, closed=True, fc=color, ec='none', zorder=z,
                         joinstyle='round'))

def people_grid(cx, cy, color):
    s = 0.30
    cols, rows = 3, 2
    dx, dy = 0.42, 0.60
    x0 = cx - (cols - 1) * dx / 2
    y0 = cy + (rows - 1) * dy / 2
    shades = [color, color, '#95A5A6', color, '#95A5A6', color]
    k = 0
    for r in range(rows):
        for c in range(cols):
            person(x0 + c*dx, y0 - r*dy, s, shades[k % len(shades)])
            k += 1

def dna(cx, cy, h, color_a, color_b):
    t = np.linspace(0, 3.6*np.pi, 160)
    amp = 0.30
    yy = cy + (t/(3.6*np.pi) - 0.5) * h
    xa = cx + amp*np.sin(t)
    xb = cx - amp*np.sin(t)
    # rungs
    for i in range(0, len(t), 12):
        ax.plot([xa[i], xb[i]], [yy[i], yy[i]], color='#BFC9CA', lw=1.3,
                zorder=4, solid_capstyle='round')
    ax.plot(xa, yy, color=color_a, lw=2.6, zorder=5, solid_capstyle='round')
    ax.plot(xb, yy, color=color_b, lw=2.6, zorder=5, solid_capstyle='round')

def ct_thumb(x, y, w, h, lesion_color, z=4):
    # dark CT slice
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.01',
        fc='#15202B', ec='#5D6D7E', lw=1.0, zorder=z))
    cx, cy = x + w/2, y + h/2
    # lung fields (two dark-gray ellipses)
    ax.add_patch(Ellipse((cx - 0.20*w, cy), 0.34*w, 0.62*h, fc='#34404A',
                         ec='none', zorder=z+1))
    ax.add_patch(Ellipse((cx + 0.20*w, cy), 0.34*w, 0.62*h, fc='#34404A',
                         ec='none', zorder=z+1))
    # mediastinum
    ax.add_patch(Ellipse((cx, cy - 0.05*h), 0.16*w, 0.5*h, fc='#46535E',
                         ec='none', zorder=z+1))
    # lesion blob
    lx = cx + 0.18*w
    ax.add_patch(Ellipse((lx, cy + 0.08*h), 0.16*w, 0.20*h, fc=lesion_color,
                         ec='white', lw=0.8, zorder=z+2))

def tissue_slide(cx, cy, w, h, z=4):
    # frosted-glass slide background
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
        boxstyle='round,pad=0.02', fc='#FBF5EF', ec='#B9A78F', lw=1.0, zorder=z))
    # IHC-stained tissue blobs (DAB brown + counterstain)
    cols = ['#C8924B', '#B97A36', '#D8B07A', '#A8631F', '#CBA06B']
    for i in range(9):
        bx = cx + np.random.uniform(-0.32, 0.32) * w
        by = cy + np.random.uniform(-0.30, 0.30) * h
        bw = np.random.uniform(0.16, 0.30) * w
        bh = np.random.uniform(0.16, 0.30) * h
        ang = np.random.uniform(0, 180)
        ax.add_patch(Ellipse((bx, by), bw, bh, angle=ang,
                             fc=cols[i % len(cols)], ec='none', alpha=0.85,
                             zorder=z+1))

# ═══════════════════════════════════════════════════════════════════════════
# PANEL A — Data sources (iconographic)
# ═══════════════════════════════════════════════════════════════════════════
AX0, AX1 = 0.25, 5.45
rbox(AX0, 0.35, AX1 - AX0, 8.35, C['panelA'], C['edgeA'], lw=2, pad=0.0)
txt(AX0 + 0.32, 8.45, 'A', fs=15, fontweight='bold', color=C['edgeA'])
txt((AX0 + AX1)/2 + 0.2, 8.45, 'Multimodal data sources',
    fs=11.5, fontweight='bold', color=C['edgeA'])

TILE_X = AX0 + 0.18
TILE_W = (AX1 - AX0) - 0.36
tile_h = 1.74
tile_y = [6.30, 4.36, 2.42, 0.48]
tile_def = [
    ('clin', 'Clinical labs (13 variables)',
     ['Age · Pack-years · dNLR · Albumin · ECOG …',
      r'$\rightarrow$ NLP sentence embedding (384-d)']),
    ('gen', 'Genomics (NGS, MSK-IMPACT)',
     ['Oncogenic NSCLC driver mutations',
      'and copy-number amplifications']),
    ('rad', 'CT radiomics (PyRadiomics)',
     ['Per-lesion features by site:',
      'primary (PC) · pleural (PL) · nodal (LN)']),
    ('path', 'Pathology — PD-L1 IHC',
     ['PD-L1 texture (GLCM) features',
      r'PD-L1 % tumour positivity (TPS)']),
]

for (key, title, lines), ty in zip(tile_def, tile_y):
    col = C[key]
    rbox(TILE_X, ty, TILE_W, tile_h, C['white'], col, lw=1.6, pad=0.05, zorder=3)
    # colour spine on the left
    ax.add_patch(Rectangle((TILE_X + 0.02, ty + 0.08), 0.10, tile_h - 0.16,
                           fc=col, ec='none', zorder=4))
    icon_cx = TILE_X + 0.95
    icon_cy = ty + tile_h/2
    if key == 'clin':
        people_grid(icon_cx, icon_cy, col)
    elif key == 'gen':
        dna(icon_cx, icon_cy, 1.3, col, '#9B59B6')
    elif key == 'rad':
        # three cascading CT thumbnails
        for j, lc in enumerate(['#E74C3C', '#3498DB', '#E84393']):
            ct_thumb(TILE_X + 0.35 + j*0.40, ty + 0.45 + (2-j)*0.06,
                     0.62, 0.78, lc, z=4 + j)
        txt(icon_cx + 0.05, ty + 0.30, 'L1 · L2 · L3', fs=6.5, color=C['lgray'])
    elif key == 'path':
        tissue_slide(icon_cx, icon_cy, 1.35, 1.15)
    # text block
    tx = TILE_X + 2.05
    txt(tx, ty + tile_h - 0.42, title, fs=8.8, fontweight='bold',
        color=col, ha='left')
    for i, ln in enumerate(lines):
        txt(tx, ty + tile_h - 0.82 - i*0.34, ln, fs=7.6, color=C['gray'],
            ha='left')

# braces / arrows A -> B  (single converging arrow)
thick_arr(AX1 + 0.02, 4.5, AX1 + 0.55, 4.5, color=C['edgeA'])

# ═══════════════════════════════════════════════════════════════════════════
# PANEL B — DyAM network diagram
# ═══════════════════════════════════════════════════════════════════════════
BX0, BX1 = 6.05, 12.55
rbox(BX0, 0.35, BX1 - BX0, 8.35, C['panelB'], C['edgeB'], lw=2, pad=0.0)
txt(BX0 + 0.32, 8.45, 'B', fs=15, fontweight='bold', color=C['edgeB'])

# DyAM title chip
rbox((BX0+BX1)/2 - 0.85, 8.05, 1.7, 0.55, C['white'], C['gray'], lw=1.6, pad=0.04,
     zorder=4)
txt((BX0+BX1)/2, 8.32, 'DyAM', fs=12, fontweight='bold', color=C['gray'])

# layer x-positions (shifted right to leave room for input labels)
xin, xatt, xout = 7.55, 9.85, 11.9
node_r = 0.24
ytop, ybot = 7.0, 1.5
yin = np.linspace(ytop, ybot, len(MOD))
yatt = np.linspace(ytop - 0.15, ybot + 0.15, len(MOD))
yo = (ytop + ybot) / 2

# short node labels (full names are given in panel A / caption)
MOD_SHORT = ['CT · PC', 'CT · PL/LN', 'Path · IHC-A',
             'Path · IHC-G', 'Genomics', 'Clinical · NLP']

# column headers only (no enclosing boxes -> nothing can cross the labels/chips)
def layer_header(xc, label):
    txt(xc, 7.62, label, fs=8.5, color=C['lgray'], style='italic')
    ax.plot([xc - 0.55, xc + 0.55], [7.42, 7.42], color=C['lgray'], lw=0.7,
            alpha=0.5, zorder=2)

layer_header(xin, 'Modality inputs')
layer_header(xatt, 'Attention layer')
layer_header(xout, 'Response')

# connections: inputs -> attention (fully connected, faint)
for yi in yin:
    for ya in yatt:
        ax.plot([xin + node_r, xatt - node_r], [yi, ya],
                color=C['line'], lw=0.6, alpha=0.40, zorder=3)
# attention -> output, coloured by modality
for ya, (_, col) in zip(yatt, MOD):
    ax.plot([xatt + node_r, xout - node_r], [ya, yo],
            color=col, lw=1.3, alpha=0.7, zorder=3)

# input nodes (filled) with short labels to the left of the lane
for yi, label, (_, col) in zip(yin, MOD_SHORT, MOD):
    ax.add_patch(Circle((xin, yi), node_r, fc=col, ec='white', lw=1.4, zorder=6))
    txt(xin - node_r - 0.30, yi, label, fs=7.2, color=C['gray'], ha='right')

# attention nodes (hollow rings, modality-coloured)
for ya, (_, col) in zip(yatt, MOD):
    ax.add_patch(Circle((xatt, ya), node_r, fc='white', ec=col, lw=2.4, zorder=6))

# output node
ax.add_patch(Circle((xout, yo), node_r + 0.06, fc=C['gray'], ec='white',
                    lw=1.6, zorder=6))
txt(xout, yo - node_r - 0.32, 'ICI response', fs=7.8, fontweight='bold',
    color=C['gray'])
txt(xout, yo - node_r - 0.60, r'$\hat{y}=\sum_i r_i a_i$', fs=9, color=C['gray'])

# response split chips
rbox(xout - 0.62, 2.55, 1.25, 0.5, C['rad_l'], C['rad'], lw=1.1, pad=0.04, zorder=5)
txt(xout, 2.80, 'PR / CR', fs=7.6, fontweight='bold', color=C['rad'])
rbox(xout - 0.62, 1.85, 1.25, 0.5, C['clin_l'], C['clin'], lw=1.1, pad=0.04, zorder=5)
txt(xout, 2.10, 'SD / PD', fs=7.6, fontweight='bold', color=C['clin'])
ax.annotate('', xy=(xout, 3.05), xytext=(xout, yo - node_r - 0.7),
            arrowprops=dict(arrowstyle='-', color=C['lgray'], lw=0.8), zorder=4)

# attention-variant caption
rbox(BX0 + 0.35, 0.55, BX1 - BX0 - 0.7, 0.78, C['white'], C['edgeB'], lw=1.2,
     pad=0.05, zorder=5)
txt((BX0+BX1)/2, 1.12, 'Per-modality risk scores $r_i$ weighted by attention $a_i$',
    fs=8, color=C['gray'])
txt((BX0+BX1)/2, 0.80,
    'Cooperative (softplus, L1-norm)   vs.   Competitive OvO (sigmoid, one-vs-others)',
    fs=7.8, color=C['edgeB'], fontweight='bold')

# arrow B -> C
thick_arr(BX1 + 0.02, 4.5, BX1 + 0.55, 4.5, color=C['edgeB'])

# ═══════════════════════════════════════════════════════════════════════════
# PANEL C — Evaluation
# ═══════════════════════════════════════════════════════════════════════════
CX0, CX1 = 13.1, 16.85
rbox(CX0, 0.35, CX1 - CX0, 8.35, C['panelC'], C['edgeC'], lw=2, pad=0.0)
txt(CX0 + 0.30, 8.45, 'C', fs=15, fontweight='bold', color=C['edgeC'])
txt((CX0+CX1)/2, 8.45, 'Evaluation', fs=11.5, fontweight='bold', color=C['edgeC'])

def mini_roc(cx, cy, w, h, col):
    ax.add_patch(Rectangle((cx, cy), w, h, fc='white', ec=C['lgray'], lw=1.0,
                           zorder=4))
    ax.plot([cx, cx+w], [cy, cy+h], color=C['lgray'], lw=0.8, ls='--', zorder=5)
    xs = np.linspace(0, 1, 40)
    ys = xs ** 0.32
    ax.plot(cx + xs*w, cy + ys*h, color=col, lw=2.0, zorder=6)

def mini_km(cx, cy, w, h):
    ax.add_patch(Rectangle((cx, cy), w, h, fc='white', ec=C['lgray'], lw=1.0,
                           zorder=4))
    xs = np.linspace(0, 1, 30)
    ax.plot(cx + xs*w, cy + h*(1 - 0.25*xs), color=C['rad'], lw=1.8, zorder=6)
    ax.plot(cx + xs*w, cy + h*(0.95 - 0.8*xs), color=C['clin'], lw=1.8, zorder=6)

def section(title, y0, col, items):
    txt((CX0+CX1)/2, y0, title, fs=9, fontweight='bold', color=col)
    ax.plot([CX0+0.25, CX1-0.25], [y0-0.20, y0-0.20], color=col, lw=0.8,
            alpha=0.6)
    y = y0 - 0.5
    for name, sub in items:
        ax.text(CX0+0.35, y, '•', fontsize=11, color=col, va='center')
        txt(CX0+0.55, y+0.02, name, fs=8.4, color=C['gray'], ha='left')
        txt(CX0+0.55, y-0.26, sub, fs=7.2, color=C['lgray'], ha='left',
            style='italic')
        y -= 0.78
    return y

# Binary classification with mini ROC
section('Binary classification', 7.55, C['edgeC'], [
    ('ROC–AUC, DeLong 95% CI', 'PR/CR vs. SD/PD'),
])
mini_roc(15.55, 6.55, 1.0, 0.85, C['edgeC'])

# Survival with mini KM
section('Survival analysis', 5.95, C['teal'], [
    ("Harrell's C-index", 'PFS discrimination'),
    ('Cox proportional hazards', 'forest plot, hazard ratios'),
    ('Kaplan–Meier', 'log-rank, risk strata'),
])
mini_km(15.55, 2.95, 1.0, 0.85)

# combos box
rbox(CX0 + 0.22, 0.62, CX1 - CX0 - 0.44, 1.55, C['white'], C['edgeC'], lw=1.2,
     pad=0.05, zorder=5)
txt((CX0+CX1)/2, 1.95, 'Modality combinations', fs=8.4, fontweight='bold',
    color=C['edgeC'])
ax.plot([CX0+0.4, CX1-0.4], [1.78, 1.78], color=C['lgray'], lw=0.6, alpha=0.5)
for k, line in enumerate(['Rad-only · Path-only', 'Rad + Genomics',
                          'Rad+Gen+Clinical (NLP)', 'Full (6 modalities)']):
    txt((CX0+CX1)/2, 1.55 - k*0.30, line, fs=7.4, color=C['gray'])

# ── Save ──────────────────────────────────────────────────────────────────
plt.savefig('figures/fig1_overview.pdf', dpi=300, bbox_inches='tight',
            facecolor='white')
plt.savefig('figures/fig1_overview.png', dpi=180, bbox_inches='tight',
            facecolor='white')
print('Saved: figures/fig1_overview.pdf  +  .png')
