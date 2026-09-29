"""Generate Supplementary Figures S1 (PCA scree plot) and S2 (t-SNE
projection) for the 384-dim NLP clinical sentence embeddings (all-MiniLM-L6-v2)
of the n=247 discovery cohort, computed from real clinical data.
"""
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

PAPER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.dirname(PAPER_DIR)
SUPP_FIG_DIR = os.path.join(PAPER_DIR, "supplementary", "figures")

sys.path.insert(0, CODE_DIR)
from lung_helpers import get_clinical_table_v2
from clinical_nlp_embedding import ClinicalTextEmbedder

BASE_DB_DIR = os.path.join(CODE_DIR, "..", "datasets")

df_cohort = pd.read_csv(f"{BASE_DB_DIR}/final_cohort_listing.csv").set_index("main_index")
df_cohort_disc = df_cohort[df_cohort["cohort"] == "discovery"]

df_clinical = get_clinical_table_v2(
    path=f"{BASE_DB_DIR}/18193MSKMINDProjectM-OmnibusInventory_DATA_2021-12-20_1540-WITH-TB-and-SCANNER.csv",
    main_index_col="dmp_pt_id",
    cohort=df_cohort_disc,
)

clinical_predictors = ['age', 'pack_years', 'ecog', 'albumin', 'dnlr',
                        'brain_mets', 'liver_mets', 'tumor_burden', 'therapy_line',
                        'recieves_combo_therapy', 'site_lung', 'recieves_pdl1_therapy', 'hist_adeno']
df_labs = df_clinical[clinical_predictors]
labels = df_clinical['label'].values

embedder = ClinicalTextEmbedder(model_name='sentence-transformers/all-MiniLM-L6-v2')
df_emb = embedder.embed_dataframe(df_labs)
X = df_emb.values
print("Embeddings shape:", X.shape)

# ── S1: PCA scree plot ──────────────────────────────────────────────────
from sklearn.decomposition import PCA

n_comp = min(50, X.shape[0], X.shape[1])
pca = PCA(n_components=n_comp)
pca.fit(X)
cum_var = np.cumsum(pca.explained_variance_ratio_) * 100

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(1, n_comp + 1), cum_var, "o-", color="#1b9e77", markersize=4)
ax.axvline(16, ls="--", color="grey", lw=1)
ax.axhline(cum_var[15], ls="--", color="grey", lw=1)
ax.annotate(f"PC16: {cum_var[15]:.1f}%", xy=(16, cum_var[15]),
            xytext=(20, cum_var[15] - 15),
            arrowprops=dict(arrowstyle="->", color="black"))
ax.set_xlabel("Principal component")
ax.set_ylabel("Cumulative variance explained (%)")
ax.set_title("PCA scree plot — NLP clinical embeddings (384-dim)", fontsize=10, fontweight="bold")
ax.set_ylim(0, 100)
fig.tight_layout()
fig.savefig(os.path.join(SUPP_FIG_DIR, "figS1_pca_scree.pdf"))
plt.close(fig)
print("Figure S1 done. PC16 cumulative variance:", cum_var[15])

# ── S2: t-SNE projection ────────────────────────────────────────────────
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
X_tsne = tsne.fit_transform(X)

fig, ax = plt.subplots(figsize=(6, 5))
for lab, color, name in [(0, "#3182bd", "PR/CR (label=0)"), (1, "#de2d26", "SD/PD (label=1)")]:
    mask = labels == lab
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color, label=name,
               alpha=0.7, s=25, edgecolor="black", linewidth=0.3)
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_title("t-SNE projection of NLP clinical embeddings", fontsize=10, fontweight="bold")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(SUPP_FIG_DIR, "figS2_tsne_embeddings.pdf"))
plt.close(fig)
print("Figure S2 done")
