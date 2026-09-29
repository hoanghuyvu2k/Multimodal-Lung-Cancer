"""
Fig 2 (DEMO STYLE) — NLP clinical-encoding pipeline in the schematic style of
images/demo-image.png (dashed panel, heavy black arrows, numeric / heatmap
matrices, white op-boxes, exploded Multi-Head-Attention detail block).

Laid out in TWO rows so every box has room to breathe at the larger font
size (row 1: Tabular input -> Sentence; row 2: Encoder [+ MHA detail] ->
Embedding [+ PCA] -> Fusion), instead of cramming five stages into one row.

Does NOT overwrite generate_fig2.py / fig2_nlp_pipeline.pdf.
Run from: paper/  directory
Output:   paper/figures/fig2_nlp_pipeline_demo.pdf  (+ .png preview)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Polygon
import numpy as np

np.random.seed(11)

BLACK = '#111111'
GRAY = '#5D6D7E'
LGRAY = '#95A5A6'
FS = 1.9    # global font-scale factor (point size of every label)
C = dict(clin='#B7460A', fn='#0E6655', model='#6C3483', emb='#1A73B5',
         fuse='#2C3E50')

# Canvas kept at the same inches-per-data-unit ratio validated for Fig 1/2
# (10.5in / 19 units) but taller, since the pipeline now spans two rows.
# (figure HEIGHT does not affect the on-page font size once placed at
# width=\textwidth, so extra ylim/height is "free" breathing room.)
fig = plt.figure(figsize=(10.5, 9.78))
fig.patch.set_facecolor('white')
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 19)
ax.set_ylim(0, 17.7)
ax.axis('off')


def dashed_panel(x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02',
        facecolor='white', edgecolor=BLACK, linewidth=1.7,
        linestyle=(0, (6, 4)), zorder=1))
    ax.text(x + w / 2, y + h + 0.16, title, fontsize=9.5 * FS, fontweight='bold',
            color=BLACK, ha='center', va='bottom', zorder=10)


def opbox(x, y, w, h, label, fs=9.5, lw=1.4, tc=BLACK, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.03',
        facecolor='white', edgecolor=BLACK, linewidth=lw, zorder=5))
    if label:
        ax.text(x + w / 2, y + h / 2, label, fontsize=fs * FS, color=tc,
                ha='center', va='center', zorder=8,
                fontweight='bold' if bold else 'normal')


def barrow(x1, y1, x2, y2, conn=None, lw=2.8, ms=24, color=BLACK):
    props = dict(arrowstyle='-|>', color=color, lw=lw, mutation_scale=ms,
                 shrinkA=0, shrinkB=0)
    if conn:
        props['connectionstyle'] = conn
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=props, zorder=7)


def txt(x, y, s, fs=9, color=BLACK, ha='center', va='center', z=8, **kw):
    return ax.text(x, y, s, fontsize=fs * FS, color=color, ha=ha, va=va,
                   zorder=z, **kw)


def heat_vec(x, y, n, cw, ch, seed, ec=BLACK):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    cm = plt.get_cmap('coolwarm')
    lo, hi = v.min(), v.max()
    for i, vi in enumerate(v):
        ax.add_patch(Rectangle((x + i * cw, y), cw, ch,
                     fc=cm((vi - lo) / (hi - lo + 1e-9)), ec='white', lw=0.4, zorder=5))
    ax.add_patch(Rectangle((x, y), n * cw, ch, fc='none', ec=ec, lw=1.5, zorder=6))
    return n * cw


PANX, PANY, PANW, PANH = 0.35, 2.0, 18.3, 15.0
dashed_panel(PANX, PANY, PANW, PANH,
             '(a) NLP encoding of 13 structured clinical variables')


def stage_tag(x, y_circle, n, label):
    ax.add_patch(Circle((x, y_circle), 0.24, fc=BLACK, ec='white',
                 lw=1.5, zorder=8))
    txt(x, y_circle, str(n), fs=9.5, color='white', fontweight='bold')
    txt(x, y_circle - 0.48, label, fs=8.6, color=BLACK, fontweight='bold')


# ═══════════════════════════════════════════════════════════════════════════
# ROW 1 — Tabular input -> Sentence  (two big, spacious boxes)
# ═══════════════════════════════════════════════════════════════════════════
ROW1_TAG_Y = PANY + PANH - 0.5           # 14.5
ROW1_H = 4.0
YC1 = ROW1_TAG_Y - 0.98 - 0.40 - ROW1_H / 2   # box top = tag_label - 0.40

# STAGE 1 — clinical table
S1X, S1W, S1H = 0.9, 7.5, ROW1_H
opbox(S1X, YC1 - S1H / 2, S1W, S1H, '', lw=1.6)
txt(S1X + S1W / 2, YC1 + S1H / 2 - 0.38, '13 clinical variables', fs=8.5,
    color=C['clin'], fontweight='bold')
rows = [('age', '69'), ('pack-years', '48'), ('ECOG', '1'), ('albumin', '4.0'),
        ('dNLR', '2.4'), ('histology', 'adeno'), ('…', '…')]
ry0 = YC1 + S1H / 2 - 0.98
for i, (k, v) in enumerate(rows):
    yy = ry0 - i * 0.46
    txt(S1X + 0.34, yy, k, fs=7.4, color=BLACK, ha='left')
    txt(S1X + S1W - 0.34, yy, v, fs=7.4, color=C['clin'], ha='right',
        fontweight='bold')
stage_tag(S1X + S1W / 2, ROW1_TAG_Y, 1, 'Tabular input')
barrow(S1X + S1W + 0.10, YC1, 10.12, YC1, lw=3.0, ms=22)
txt((S1X + S1W + 0.10 + 10.12) / 2, YC1 + 0.34, 'df_to_text', fs=5.0,
    color=BLACK, fontfamily='monospace')

# STAGE 2 — English text prompt (document)
S2X, S2W, S2H = 10.2, 7.1, ROW1_H
dx, dy, fold = S2X, YC1 - S2H / 2, 0.44
ax.add_patch(Polygon([(dx, dy), (dx + S2W, dy), (dx + S2W, dy + S2H - fold),
             (dx + S2W - fold, dy + S2H), (dx, dy + S2H)], closed=True,
             fc='white', ec=BLACK, lw=1.6, zorder=5))
ax.add_patch(Polygon([(dx + S2W - fold, dy + S2H), (dx + S2W - fold, dy + S2H - fold),
             (dx + S2W, dy + S2H - fold)], closed=True, fc='#EDEDED', ec=BLACK,
             lw=1.0, zorder=6))
txt(S2X + S2W / 2, YC1 + S2H / 2 - 0.34, 'English text prompt', fs=8.5,
    color=C['fn'], fontweight='bold')
for i, ln in enumerate(['"The patient is a 69-year-old,',
                        '48 pack-years, ECOG 1,',
                        'albumin 4.0, dNLR 2.4 … "']):
    txt(S2X + 0.34, YC1 + S2H / 2 - 1.05 - i * 0.52, ln, fs=7.0, color=BLACK,
        ha='left', style='italic')
stage_tag(S2X + S2W / 2, ROW1_TAG_Y, 2, 'Sentence')

# ═══════════════════════════════════════════════════════════════════════════
# Row-wrap connector: end of row 1 (stage 2) -> start of row 2 (stage 3)
# ═══════════════════════════════════════════════════════════════════════════
ROW2_TAG_Y = YC1 - ROW1_H / 2 - 0.70
S3H = 3.8
YC2 = ROW2_TAG_Y - 0.42 - 0.30 - S3H / 2

S3X, S3W = 0.9, 4.75

# three-segment connector: straight down clear of row 1, then ACROSS at a
# height well above the "Encoder" stage-tag text/number, then straight down
# into the box's top-left corner -- never sweeps through the label like a
# single arc did.
WPY = YC1 - ROW1_H / 2 - 0.33
WPX = S3X + 0.15
ax.plot([S2X + S2W / 2, S2X + S2W / 2], [YC1 - ROW1_H / 2 - 0.03, WPY],
        color=BLACK, lw=2.6, zorder=6, solid_capstyle='round')
ax.plot([S2X + S2W / 2, WPX], [WPY, WPY],
        color=BLACK, lw=2.6, zorder=6, solid_capstyle='round')
barrow(WPX, WPY, S3X - 0.06, YC2 + S3H / 2 - 0.15,
       lw=2.6, ms=20, color=BLACK)

# ═══════════════════════════════════════════════════════════════════════════
# ROW 2 — Encoder (+ Multi-Head Attention detail) -> Embedding (+ PCA) -> Fusion
# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3 — sentence-transformer encoder
opbox(S3X, YC2 - S3H / 2, S3W, S3H, '', lw=1.6)
txt(S3X + S3W / 2, YC2 + S3H / 2 - 0.34, 'all-MiniLM-L6-v2', fs=7.0,
    color=C['model'], fontweight='bold')
txt(S3X + S3W / 2, YC2 + S3H / 2 - 0.72, '6 Transformer layers', fs=6.2,
    color=GRAY, style='italic')
for i in range(3):
    by = YC2 - 0.78 + i * 0.66      # pitch 0.66 / cao 0.48 -> khe 0.18
    opbox(S3X + 0.36, by, S3W - 0.72, 0.48, 'self-attention', fs=6.6)
txt(S3X + S3W / 2, YC2 - S3H / 2 + 0.32, 'mean-pool · L2-norm', fs=6.2,
    color=GRAY, style='italic')
stage_tag(S3X + S3W / 2, ROW2_TAG_Y, 3, 'Encoder')
barrow(S3X + S3W + 0.08, YC2, S3X + S3W + 0.78, YC2, lw=3.0, ms=22)

# ── exploded Multi-Head Attention detail (below the encoder) ──────────────────
MHW, MHH = 4.6, 3.30
MHX = S3X + S3W / 2 - MHW / 2
MHY = YC2 - S3H / 2 - 0.18 - MHH
ax.add_patch(FancyBboxPatch((MHX, MHY), MHW, MHH, boxstyle='round,pad=0.02',
    facecolor='white', edgecolor=BLACK, linewidth=1.4, zorder=5))
txt(MHX + MHW / 2, MHY + MHH - 0.36, 'Multi-Head Attention', fs=6.6,
    fontweight='bold')
stack = ['Linear', 'MatMul', 'SoftMax', 'Scale', 'MatMul']
STK_H, STK_PITCH = 0.36, 0.50   # gap 0.14 giữa các box (trước: 0.07 -> dính nhau)
sx, sy0 = MHX + 0.52, MHY + 0.30
for i, s in enumerate(stack):
    opbox(sx, sy0 + i * STK_PITCH, 1.85, STK_H, s, fs=6.6)
for i in range(len(stack) - 1):
    barrow(sx + 0.925, sy0 + i * STK_PITCH + STK_H,
           sx + 0.925, sy0 + (i + 1) * STK_PITCH, lw=1.3, ms=9)
txt(MHX + MHW - 0.72, MHY + MHH / 2 - 0.15, 'Q\nK\nV', fs=8.2,
    linespacing=1.5)
ax.annotate('', xy=(S3X + S3W / 2, MHY + MHH + 0.02),
            xytext=(S3X + S3W / 2, YC2 - S3H / 2 - 0.02),
            arrowprops=dict(arrowstyle='-', color=GRAY, lw=1.2,
                            linestyle=(0, (3, 2))), zorder=4)

# STAGE 4 — 384-dim embedding heatmap (+ PCA branch)
TOP4 = ROW2_TAG_Y - 0.48 - 0.30
S4X = 6.55
HEAT4_TOP = TOP4 - 0.72
w4 = heat_vec(S4X, HEAT4_TOP - 0.85, 24, 0.13, 0.85, seed=1)
txt(S4X + w4 / 2, TOP4 - 0.35, '384-dim embedding', fs=8.2, color=C['emb'],
    fontweight='bold')
NLPRAW_Y = HEAT4_TOP - 0.85 - 0.30
txt(S4X + w4 / 2, NLPRAW_Y, '"NLP raw"', fs=7.4, color=C['emb'],
    fontweight='bold')
EMB_MIDX = (S4X - 1.05 + S4X + w4) / 2  # centred over heatmap + PCA branch
stage_tag(EMB_MIDX, ROW2_TAG_Y, 4, 'Embedding')
bx = S4X + 0.22
barrow(bx, NLPRAW_Y - 0.18, bx, NLPRAW_Y - 0.98, lw=2.4, ms=17)

PYC = NLPRAW_Y - 1.50
# Hộp PCA dịch phải để không đè lên hộp Encoder (mép phải = S3X + S3W)
PCA_X0, PCA_W = 5.80, 2.35
opbox(PCA_X0, PYC - 0.42, PCA_W, 0.84, 'PCA (16 comp.)', fs=5.8)
txt(PCA_X0 + PCA_W / 2, PYC - 0.75, r'$\approx$94% variance', fs=6.2,
    color=GRAY, style='italic')
barrow(PCA_X0 + PCA_W + 0.06, PYC, PCA_X0 + PCA_W + 0.62, PYC, lw=2.4, ms=17)
P16X = PCA_X0 + PCA_W + 0.75
w5 = heat_vec(P16X, PYC - 0.34, 16, 0.115, 0.68, seed=4)
txt(P16X + w5 / 2, PYC + 0.72, '16-dim "NLP-PCA16"', fs=6.2, color=C['emb'],
    fontweight='bold')

# STAGE 5 — fusion (DyAM)
FX, FW = 11.9, 5.8
FYC = (TOP4 + (PANY + 0.20)) / 2 - 0.15
FH = 4.6
opbox(FX, FYC - FH / 2, FW, FH, '', lw=1.7)
txt(FX + FW / 2, FYC + FH / 2 - 0.40, 'Fusion (DyAM)', fs=8.4, color=C['fuse'],
    fontweight='bold')
for i, m in enumerate(['+ CT radiomics', '+ Pathology IHC', '+ Genomics',
                       '+ PD-L1 TPS']):
    txt(FX + 0.40, FYC + FH / 2 - 1.05 - i * 0.52, m, fs=7.6, color=BLACK,
        ha='left')
txt(FX + FW / 2, FYC - FH / 2 + 0.42, 'attention-weighted risk', fs=6.6,
    color=GRAY, style='italic')
stage_tag(FX + FW / 2, ROW2_TAG_Y, 5, 'Multimodal fusion')

barrow(S4X + w4 + 0.06, HEAT4_TOP - 0.35, FX - 0.06, FYC + FH / 2 - 0.9,
       conn='arc3,rad=0.15', lw=2.6, ms=20)
barrow(P16X + w5 + 0.06, PYC, FX - 0.06, FYC - 0.3,
       conn='arc3,rad=-0.18', lw=2.4, ms=17)

# ── caveat banner + colour-bar (below the panel) ──────────────────────────────
ax.add_patch(FancyBboxPatch((0.35, 0.35), 9.6, 1.35, boxstyle='round,pad=0.03',
    facecolor='#FFFBEC', edgecolor='#B7950B', linewidth=1.4, zorder=3))
txt(0.68, 1.42, '!', fs=13, color='#B7950B', fontweight='bold', ha='left')
txt(1.10, 1.44, r'$\mathtt{no\_scale=True}$ — RobustScaler disabled',
    fs=7.4, color=BLACK, ha='left', fontweight='bold')
txt(1.10, 1.00, 'so the cosine-normalised geometry of the',
    fs=7.0, color=BLACK, ha='left')
txt(1.10, 0.68, 'embedding space is preserved.', fs=7.0, color=BLACK,
    ha='left')

cbx, cby, cbw = 13.3, 0.75, 4.6
cm = plt.get_cmap('coolwarm')
for i in range(40):
    ax.add_patch(Rectangle((cbx + i * cbw / 40, cby), cbw / 40, 0.30,
                 fc=cm(i / 39), ec='none', zorder=4))
ax.add_patch(Rectangle((cbx, cby), cbw, 0.30, fc='none', ec=BLACK, lw=1.0, zorder=5))
txt(cbx + cbw / 2, cby + 0.72, 'embedding value', fs=7.4, color=GRAY,
    fontweight='bold')
txt(cbx, cby - 0.34, 'low', fs=6.8, color=GRAY, ha='left')
txt(cbx + cbw, cby - 0.34, 'high', fs=6.8, color=GRAY, ha='right')

plt.savefig('figures/fig2_nlp_pipeline_demo.pdf', dpi=300, bbox_inches='tight',
            facecolor='white')
plt.savefig('figures/fig2_nlp_pipeline_demo.png', dpi=170, bbox_inches='tight',
            facecolor='white')
print('Saved: figures/fig2_nlp_pipeline_demo.pdf  +  .png')
