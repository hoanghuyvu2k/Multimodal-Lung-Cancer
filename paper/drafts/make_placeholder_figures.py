"""Generate blank placeholder PDFs for all figures referenced in the manuscript,
so that main.tex / supplementary.tex compile end-to-end before real figures exist."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

main_figs = [
    "fig1_overview", "fig2_nlp_pipeline",
    "fig3a_auc_ihca", "fig3b_auc_ihcg",
    "fig4a_km_ihca_noclin", "fig4b_km_ihca_nlppca16",
    "fig4c_km_ihcg_noclin", "fig4d_km_ihcg_nlpraw",
    "fig5a_cindex", "fig5b_tdauc", "fig5c_cox_forest", "fig5d_ibs",
]
supp_figs = [
    "figS1_pca_scree", "figS2_tsne_embeddings", "figS3_bootstrap_pvalue",
    "figS4_ovo_scatter", "figS5_modality_heatmap",
]

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for name in main_figs:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, name, ha="center", va="center", fontsize=14)
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(os.path.join(base, "figures", f"{name}.pdf"))
    plt.close(fig)

for name in supp_figs:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, name, ha="center", va="center", fontsize=14)
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(os.path.join(base, "supplementary", "figures", f"{name}.pdf"))
    plt.close(fig)

print("done")
