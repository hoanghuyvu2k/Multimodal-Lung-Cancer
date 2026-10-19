"""Kaplan-Meier at BM1 for DyAM, OvO, DyAM+NLP, OvO+NLP from the 5-seed ensemble risk score (mean over the 5 seeds of the standardised
out-of-fold score). Ra figures/fig4{a..d}_km_bm1_*.pdf.  python paper/make_km_bm1_ensemble.py"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.join(os.path.dirname(HERE), "experiments", "results", "ref", "bm1_surv_scores")
CLIN = os.path.join(os.path.dirname(os.path.dirname(HERE)), "datasets",
                    "18193MSKMINDProjectM-OmnibusInventory_DATA_2021-12-20_1540-WITH-TB-and-SCANNER.csv")
clin = pd.read_csv(CLIN).drop_duplicates("dmp_pt_id").set_index("dmp_pt_id")


def panel(fname, model, title):
    d = pd.read_csv(os.path.join(SC, f"{model}.csv"), index_col=0)
    s = ((d - d.mean()) / d.std()).mean(axis=1)
    c = clin.reindex(s.index)
    hi = (s > s.median()).values
    t, e = c["pfs"].astype(float).values, c["pfs_censor"].astype(int).values
    r = logrank_test(t[hi], t[~hi], event_observed_A=e[hi], event_observed_B=e[~hi])
    fig, ax = plt.subplots(figsize=(5, 5.2))
    for m, lab, col in ((~hi, "Low risk", "#3182bd"), (hi, "High risk", "#de2d26")):
        KaplanMeierFitter().fit(t[m], e[m], label=lab).plot_survival_function(ax=ax, color=col, ci_show=True)
    ax.set_xlim(0, 24); ax.set_ylim(0, 1.0); ax.set_xlabel("Time (months)"); ax.set_ylabel("PFS probability")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.text(0.95, 0.92, rf"log-rank $\chi^2={r.test_statistic:.1f}$", transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.8))
    ax.legend(fontsize=8, loc="lower left")
    tp = [0, 6, 12, 18, 24]
    cells = [[str(int((t[~hi] >= x).sum())) for x in tp], [str(int((t[hi] >= x).sum())) for x in tp]]
    tb = ax.table(cellText=cells, rowLabels=["Low risk", "High risk"], colLabels=[str(x) for x in tp], cellLoc="center", bbox=[0.0, -0.42, 1.0, 0.18])
    tb.auto_set_font_size(False); tb.set_fontsize(7)
    fig.subplots_adjust(bottom=0.32); fig.savefig(os.path.join(HERE, "figures", fname)); plt.close(fig)
    print(fname, f"chi2={r.test_statistic:.2f}", "n", len(s))


for f, m, ti in (("fig4a_km_bm1_dyam.pdf", "dyam", "BM1, DyAM"), ("fig4b_km_bm1_ovo.pdf", "ovo", "BM1, OvO"),
                 ("fig4c_km_bm1_dyam_nlp.pdf", "dyam_nlp", "BM1, DyAM+NLP"), ("fig4d_km_bm1_ovo_nlp.pdf", "ovo_nlp", "BM1, OvO+NLP")):
    panel(f, m, ti)
