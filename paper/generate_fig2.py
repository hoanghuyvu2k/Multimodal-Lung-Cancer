"""
Generate Fig 2: NLP encoding of structured clinical features.
Iconographic pipeline in the style of figures/model.jpg / Fig 1:
  clinical table -> text prompt -> sentence transformer -> embedding heatmap
  -> (PCA) -> fusion with other modalities.

Run from: paper/  directory
Output:   paper/figures/fig2_nlp_pipeline.pdf  (+ .png preview)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Polygon
import numpy as np

np.random.seed(11)

C = dict(
    clin='#B7460A',  clin_l='#FDEBD0',
    fn='#0E6655',    fn_l='#D1F2EB',
    model='#6C3483', model_l='#EAD9F5',
    emb='#1A73B5',   emb_l='#DDEEFF',
    fuse='#2C3E50',  fuse_l='#EAECEE',
    gray='#2C3E50',  lgray='#7F8C8D',
    line='#5D6D7E',
    white='#FFFFFF', panel='#FCFDFE',
)

fig = plt.figure(figsize=(16, 7))
fig.patch.set_facecolor(C['white'])
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 7)
ax.axis('off')

def rbox(x, y, w, h, fc, ec, lw=1.5, alpha=1.0, pad=0.12, zorder=2, ls='-'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad={pad}',
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=zorder,
        linestyle=ls))

def txt(x, y, s, fs=9, color=C['gray'], ha='center', va='center', zorder=8, **kw):
    return ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va,
                   zorder=zorder, **kw)

def arrow(x1, y1, x2, y2, color=C['line'], lw=2.4, ms=20, conn=None):
    props = dict(arrowstyle='-|>', color=color, lw=lw, mutation_scale=ms)
    if conn:
        props['connectionstyle'] = conn
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=props, zorder=6)

YC = 4.55   # main pipeline centre line

# ── Stage labels above ───────────────────────────────────────────────────────
def stage_tag(x, n, label, col):
    ax.add_patch(Circle((x, 6.62), 0.20, fc=col, ec='white', lw=1.5, zorder=7))
    txt(x, 6.62, str(n), fs=9, color='white', fontweight='bold')
    txt(x, 6.24, label, fs=8, color=col, fontweight='bold')

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1 — 13 numeric clinical variables (mini table)
# ═══════════════════════════════════════════════════════════════════════════
S1X, S1W, S1H = 0.45, 2.35, 2.7
rbox(S1X, YC - S1H/2, S1W, S1H, C['clin_l'], C['clin'], lw=1.7, pad=0.06, zorder=3)
txt(S1X + S1W/2, YC + S1H/2 - 0.28, '13 clinical variables', fs=8.5,
    fontweight='bold', color=C['clin'])
rows = [('age', '69'), ('pack-years', '48'), ('ECOG', '1'),
        ('albumin', '4.00'), ('dNLR', '2.40'), ('tumour burden', '1.25'),
        ('histology', 'adeno'), ('…', '…')]
ry0 = YC + S1H/2 - 0.62
for i, (k, v) in enumerate(rows):
    yy = ry0 - i * 0.245
    txt(S1X + 0.22, yy, k, fs=7, color=C['gray'], ha='left')
    txt(S1X + S1W - 0.22, yy, v, fs=7, color=C['clin'], ha='right',
        fontweight='bold')
    if i < len(rows) - 1:
        ax.plot([S1X + 0.18, S1X + S1W - 0.18], [yy - 0.12, yy - 0.12],
                color='#E5D5C5', lw=0.5, zorder=4)
stage_tag(S1X + S1W/2, 1, 'Tabular input', C['clin'])

# arrow 1 -> 2  with function label
arrow(S1X + S1W + 0.02, YC, S1X + S1W + 0.95, YC, color=C['fn'])
txt(S1X + S1W + 0.52, YC + 0.45, 'df_to_text', fs=7, color=C['fn'],
    fontweight='bold', fontfamily='monospace')
txt(S1X + S1W + 0.52, YC + 0.2, '_prompts()', fs=7, color=C['fn'],
    fontfamily='monospace')

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2 — English text prompt (document icon)
# ═══════════════════════════════════════════════════════════════════════════
S2X, S2W, S2H = 4.0, 2.55, 2.7
# document with folded corner
dx, dy = S2X, YC - S2H/2
fold = 0.34
verts = [(dx, dy), (dx + S2W, dy), (dx + S2W, dy + S2H - fold),
         (dx + S2W - fold, dy + S2H), (dx, dy + S2H)]
ax.add_patch(Polygon(verts, closed=True, fc='#FFFFFF', ec=C['fn'], lw=1.7,
                     zorder=3, joinstyle='round'))
ax.add_patch(Polygon([(dx + S2W - fold, dy + S2H), (dx + S2W - fold, dy + S2H - fold),
                      (dx + S2W, dy + S2H - fold)], closed=True,
                     fc=C['fn_l'], ec=C['fn'], lw=1.0, zorder=4))
txt(S2X + S2W/2, YC + S2H/2 - 0.32, 'English text prompt', fs=8.5,
    fontweight='bold', color=C['fn'])
# faux text lines
quote = ['"The patient is a 69-year-old.',
         'Smoking history in pack-years is',
         '48. ECOG performance status is 1.',
         'Blood albumin is 4.00. dNLR is',
         '2.40 … without liver metastasis."']
for i, ln in enumerate(quote):
    txt(S2X + 0.2, YC + S2H/2 - 0.72 - i*0.32, ln, fs=6.8, color=C['gray'],
        ha='left', style='italic')
stage_tag(S2X + S2W/2, 2, 'Sentence', C['fn'])

arrow(S2X + S2W + 0.02, YC, S2X + S2W + 0.78, YC, color=C['model'])

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3 — Sentence transformer (stacked encoder layers)
# ═══════════════════════════════════════════════════════════════════════════
S3X, S3W, S3H = 7.35, 2.05, 2.7
rbox(S3X, YC - S3H/2, S3W, S3H, C['model_l'], C['model'], lw=1.7, pad=0.06,
     zorder=3)
txt(S3X + S3W/2, YC + S3H/2 - 0.3, 'all-MiniLM-L6-v2', fs=8.3,
    fontweight='bold', color=C['model'])
txt(S3X + S3W/2, YC + S3H/2 - 0.55, '6 Transformer layers', fs=6.6,
    color=C['lgray'], style='italic')
# stacked transformer blocks
for i in range(3):
    by = YC - 0.72 + i * 0.48
    rbox(S3X + 0.32, by, S3W - 0.64, 0.38, C['white'], C['model'], lw=1.0,
         pad=0.03, zorder=4)
    txt(S3X + S3W/2, by + 0.19, 'self-attention', fs=6.3, color=C['model'])
txt(S3X + S3W/2, YC - S3H/2 + 0.26, 'mean-pool · L2-norm', fs=6.8,
    color=C['lgray'], style='italic')
stage_tag(S3X + S3W/2, 3, 'Encoder', C['model'])

arrow(S3X + S3W + 0.02, YC, S3X + S3W + 0.72, YC, color=C['emb'])

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 4 — 384-dim embedding (heatmap vector)  "NLP raw"
# ═══════════════════════════════════════════════════════════════════════════
def heatmap_vec(x, y, ncells, cell_w, cell_h, seed):
    rng = np.random.default_rng(seed)
    vals = rng.standard_normal(ncells)
    vmin, vmax = vals.min(), vals.max()
    cmap = plt.get_cmap('coolwarm')
    for i, v in enumerate(vals):
        t = (v - vmin) / (vmax - vmin + 1e-9)
        ax.add_patch(Rectangle((x + i*cell_w, y), cell_w, cell_h,
                               fc=cmap(t), ec='white', lw=0.4, zorder=5))
    ax.add_patch(Rectangle((x, y), ncells*cell_w, cell_h, fc='none',
                           ec=C['emb'], lw=1.4, zorder=6))

S4X = 10.05
NC1, CW1 = 22, 0.072
hm_w1 = NC1 * CW1
heatmap_vec(S4X, YC + 0.18, NC1, CW1, 0.5, seed=1)
txt(S4X + hm_w1/2, YC + 0.95, '384-dim embedding', fs=8.3, fontweight='bold',
    color=C['emb'])
txt(S4X + hm_w1/2, YC - 0.12, '"NLP raw"', fs=7.6, color=C['emb'],
    fontweight='bold')
stage_tag(S4X + hm_w1/2, 4, 'Embedding', C['emb'])

# raw path continues to fusion (upper branch)
arrow(S4X + hm_w1 + 0.05, YC + 0.43, 13.05, YC + 0.43, color=C['emb'])

# branch down to PCA (drop from the left part of the embedding)
branch_x = S4X + 0.15
ax.annotate('', xy=(branch_x, YC - 1.42), xytext=(branch_x, YC - 0.18),
            arrowprops=dict(arrowstyle='-|>', color=C['emb'], lw=2.0,
                            mutation_scale=16), zorder=6)

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 5 — PCA -> 16-dim  "NLP-PCA16"  (lower branch)
# ═══════════════════════════════════════════════════════════════════════════
PYC = YC - 1.85
# PCA funnel (centred under the down-arrow)
fw = 1.4
fx = branch_x - fw/2
rbox(fx, PYC - 0.30, fw, 0.60, C['white'], C['emb'], lw=1.4, pad=0.04, zorder=4)
txt(branch_x, PYC + 0.02, 'PCA (16 comp.)', fs=7.3, color=C['emb'],
    fontweight='bold')
txt(branch_x, PYC - 0.50, r'$\approx$80% variance', fs=6.8, color=C['lgray'],
    style='italic')
arrow(fx + fw + 0.02, PYC, fx + fw + 0.48, PYC, color=C['emb'])

# 16-dim heatmap
NC2, CW2 = 16, 0.060
hm_w2 = NC2 * CW2
P16X = fx + fw + 0.55
heatmap_vec(P16X, PYC - 0.25, NC2, CW2, 0.5, seed=4)
# label above heatmap (well clear of the fusion box on the right)
txt(P16X + hm_w2/2, PYC + 0.5, '16-dim "NLP-PCA16"', fs=7.3, fontweight='bold',
    color=C['emb'])

# 16-dim path to fusion (curves up into the fusion-box bottom edge)
arrow(P16X + hm_w2 + 0.06, PYC, 13.7, YC - 1.27,
      color=C['emb'], conn='arc3,rad=0.20')

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 6 — Fusion with other modalities (DyAM)
# ═══════════════════════════════════════════════════════════════════════════
FX, FW, FH = 13.15, 2.55, 2.5
rbox(FX, YC - FH/2, FW, FH, C['fuse_l'], C['fuse'], lw=1.8, pad=0.07, zorder=3)
txt(FX + FW/2, YC + FH/2 - 0.32, 'Fusion (DyAM)', fs=8.8, fontweight='bold',
    color=C['fuse'])
for i, m in enumerate(['+ CT radiomics', '+ Pathology IHC', '+ Genomics',
                       '+ PD-L1 TPS']):
    txt(FX + 0.25, YC + 0.55 - i*0.34, m, fs=7.3, color=C['gray'], ha='left')
txt(FX + FW/2, YC - FH/2 + 0.32, 'attention-weighted risk', fs=7,
    color=C['lgray'], style='italic')
stage_tag(FX + FW/2, 5, 'Multimodal fusion', C['fuse'])

# ═══════════════════════════════════════════════════════════════════════════
# no_scale caveat banner (bottom)
# ═══════════════════════════════════════════════════════════════════════════
rbox(0.45, 0.35, 12.3, 0.92, '#FFFBEC', '#B7950B', lw=1.3, pad=0.05, zorder=3)
txt(0.85, 0.97, '!', fs=13, color='#B7950B', fontweight='bold', ha='left')
txt(1.25, 0.97,
    r'NLP embeddings are passed with $\mathtt{no\_scale=True}$ — RobustScaler is disabled',
    fs=8.2, color=C['gray'], ha='left', fontweight='bold')
txt(1.25, 0.62,
    'so that the cosine-normalised geometry of the embedding space is preserved.',
    fs=8, color=C['gray'], ha='left')

# colour-bar legend for heatmap
cbx, cby, cbw = 13.4, 0.55, 2.0
cmap = plt.get_cmap('coolwarm')
for i in range(40):
    ax.add_patch(Rectangle((cbx + i*cbw/40, cby), cbw/40, 0.18,
                           fc=cmap(i/39), ec='none', zorder=4))
ax.add_patch(Rectangle((cbx, cby), cbw, 0.18, fc='none', ec=C['lgray'], lw=0.8,
                       zorder=5))
txt(cbx, cby + 0.42, 'low', fs=6.5, color=C['lgray'], ha='left')
txt(cbx + cbw, cby + 0.42, 'high', fs=6.5, color=C['lgray'], ha='right')
txt(cbx + cbw/2, cby + 0.42, 'embedding value', fs=6.8, color=C['lgray'])

plt.savefig('figures/fig2_nlp_pipeline.pdf', dpi=300, bbox_inches='tight',
            facecolor='white')
plt.savefig('figures/fig2_nlp_pipeline.png', dpi=180, bbox_inches='tight',
            facecolor='white')
print('Saved: figures/fig2_nlp_pipeline.pdf  +  .png')
