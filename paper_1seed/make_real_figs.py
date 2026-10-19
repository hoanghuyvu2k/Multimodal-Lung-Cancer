"""Hình 4 (KM), Hình 5 (tóm tắt sống còn), Hình S3 (bootstrap), Hình S4 (scatter OvO vs DyAM) từ DỮ LIỆU THẬT của môi trường tham chiếu.

Thay cho paper/drafts/make_fig4_km.py (thời gian sống MÔ PHỎNG), make_fig3_fig5.py (số gõ cứng của phân hoạch cũ, IBS),
make_figS3_bootstrap.py (xấp xỉ chuẩn giải tích) và make_figS4_ovo_scatter.py (Excel Windows cũ).

Nguồn:  experiments/results/ref/survival_scores/*.csv   điểm nguy cơ ngoài-fold, 11 biến thể x 5 seed
        experiments/results/ref/{verify_nlp_5seed,tdauc_5seed}.json
        experiments/results/ref_all21_part*.json
        ../datasets/18193MSKMINDProjectM-...csv  (pfs tháng, pfs_censor = 1 là sự kiện)
Chạy từ paper/:   python make_real_figs.py
Hình 4 dùng seed 42 (một phân hoạch 10-fold) và chia đôi tại TRUNG VỊ điểm nguy cơ, đúng cách km_chi2() của verify_nlp_5seed.py.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "experiments", "results")
REF = os.path.join(RES, "ref")
FIG = os.path.join(HERE, "figures")
SUPP = os.path.join(HERE, "supplementary", "figures")
CLIN = os.path.join(os.path.dirname(ROOT), "datasets",
                    "18193MSKMINDProjectM-OmnibusInventory_DATA_2021-12-20_1540-WITH-TB-and-SCANNER.csv")

clin = pd.read_csv(CLIN).drop_duplicates("dmp_pt_id").set_index("dmp_pt_id")
five = json.load(open(os.path.join(REF, "verify_nlp_5seed.json"), encoding="utf-8"))["arms"]
td = json.load(open(os.path.join(REF, "tdauc_5seed.json"), encoding="utf-8"))
STEM = {"No clinical": "no_clinical", "+ Labs": "labs", "+ NLP raw": "nlp_raw", "+ NLP-PCA16": "nlppca16"}


# ---------------------------------------------------------------- Figure 4: Kaplan-Meier, seed 42, median split
def km_panel(fname, arm, variant, title):
    sc = pd.read_csv(os.path.join(REF, "survival_scores", f"{arm}__{STEM[variant]}.csv"), index_col=0)
    c = clin.reindex(sc.index)
    s = sc["seed_42"]
    hi = (s > s.median()).values
    t, e = c["pfs"].astype(float).values, c["pfs_censor"].astype(int).values
    r = logrank_test(t[hi], t[~hi], event_observed_A=e[hi], event_observed_B=e[~hi])
    fig, ax = plt.subplots(figsize=(5, 5.2))
    for m, lab, col in ((~hi, "Low risk", "#3182bd"), (hi, "High risk", "#de2d26")):
        KaplanMeierFitter().fit(t[m], e[m], label=lab).plot_survival_function(ax=ax, color=col, ci_show=True)
    ax.set_xlim(0, 24); ax.set_ylim(0, 1.0)
    ax.set_xlabel("Time (months)"); ax.set_ylabel("PFS probability")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.text(0.95, 0.92, rf"log-rank $\chi^2={r.test_statistic:.1f}$", transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.8))
    ax.legend(fontsize=8, loc="lower left")
    tp = [0, 6, 12, 18, 24]
    cells = [[str(int((t[~hi] >= x).sum())) for x in tp], [str(int((t[hi] >= x).sum())) for x in tp]]
    tb = ax.table(cellText=cells, rowLabels=["Low risk", "High risk"], colLabels=[str(x) for x in tp], cellLoc="center",
                  bbox=[0.0, -0.42, 1.0, 0.18])
    tb.auto_set_font_size(False); tb.set_fontsize(7)
    fig.subplots_adjust(bottom=0.32)
    fig.savefig(os.path.join(FIG, fname)); plt.close(fig)
    print(fname, f"chi2={r.test_statistic:.2f} n_high={int(hi.sum())} n={len(hi)} events={int(e.sum())}")


for f, a, v, ti in (("fig4a_km_ihca_noclin.pdf", "IHC-A", "No clinical", "IHC-A, no clinical"),
                    ("fig4b_km_ihca_nlppca16.pdf", "IHC-A", "+ NLP-PCA16", "IHC-A, + NLP-PCA16"),
                    ("fig4c_km_ihcg_noclin.pdf", "IHC-G", "No clinical", "IHC-G, no clinical"),
                    ("fig4d_km_ihcg_nlpraw.pdf", "IHC-G", "+ NLP raw", "IHC-G, + NLP raw")):
    km_panel(f, a, v, ti)

# ---------------------------------------------------------------- Figure 5: 5-seed mean +/- s.d.
models, arm, rows = [], [], []
for a in ("IHC-A", "IHC-G"):
    for v in STEM:
        x = five[a][v]
        models.append(f"{v.replace('No clinical', 'No Clinical')} ({a})")
        arm.append(a)
        rows.append(dict(ci=x["cindex"], hr=x["cox_hr"], km=x["km_chi2"], td=td[f"{a}__{STEM[v]}"]))
cols = {"IHC-A": "#4393c3", "IHC-G": "#d6604d"}
bc = [cols[a] for a in arm]
x = np.arange(len(models))
handles = [plt.Rectangle((0, 0), 1, 1, color=cols[a]) for a in cols]

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(x, [r["ci"]["mean"] for r in rows], yerr=[r["ci"]["sd"] for r in rows], capsize=4, color=bc, edgecolor="black", linewidth=0.6)
ax.set_xticks(x); ax.set_xticklabels(models, rotation=35, ha="right", fontsize=7.5)
ax.set_ylabel("Harrell's C-index"); ax.set_ylim(0.50, 0.75); ax.axhline(0.5, ls="--", color="grey", lw=1)
ax.legend(handles, list(cols), fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5a_cindex.pdf")); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), sharey=True)
times, sty = [6, 12, 18], ["o-", "s--", "^-.", "d:"]
for ax, a, idx in zip(axes, cols, (range(0, 4), range(4, 8))):
    for k, i in enumerate(idx):
        m = [rows[i]["td"][str(t)]["mean"] for t in times]
        sd = [rows[i]["td"][str(t)]["sd"] for t in times]
        ax.errorbar(times, m, yerr=sd, fmt=sty[k], capsize=3, label=list(STEM)[k])
    ax.set_title(a, fontsize=10, fontweight="bold"); ax.set_xlabel("Time (months)"); ax.set_xticks(times)
axes[0].set_ylabel("Time-dependent AUC"); axes[0].set_ylim(0.60, 0.74); axes[1].legend(fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5b_tdauc.pdf")); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
y = np.arange(len(models))[::-1]
for yi, i in zip(y, range(len(models))):
    h = rows[i]["hr"]
    ax.plot([h["mean"] - h["sd"], h["mean"] + h["sd"]], [yi, yi], color=bc[i], lw=2)
    ax.plot(h["mean"], yi, "o", color=bc[i], markersize=7, markeredgecolor="black")
ax.axvline(1, ls="--", color="black", lw=1); ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels(models, fontsize=8)
ax.set_xlabel(r"Hazard ratio per unit risk score (mean $\pm$ s.d., 5 partitions), log scale"); ax.set_xlim(1, 15)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5c_cox_forest.pdf")); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(x, [r["km"]["mean"] for r in rows], yerr=[r["km"]["sd"] for r in rows], capsize=4, color=bc, edgecolor="black", linewidth=0.6)
ax.axhline(3.84, ls="--", color="black", lw=1, label=r"$\chi^2 = 3.84$ ($p = 0.05$)")
ax.set_xticks(x); ax.set_xticklabels(models, rotation=35, ha="right", fontsize=7.5)
ax.set_ylabel(r"Log-rank $\chi^2$ (mean $\pm$ s.d.)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5d_km_chi2.pdf")); plt.close(fig)
print("Figure 5 done")

# ---------------------------------------------------------------- Figure S3: real paired patient-level bootstrap
a = pd.read_csv(os.path.join(REF, "survival_scores", "IHC-G__nlp_raw.csv"), index_col=0)
b = pd.read_csv(os.path.join(REF, "survival_scores", "IHC-G__labs.csv"), index_col=0)
b = b.loc[a.index]
y = a["label"].values
sc = [c for c in a.columns if c.startswith("seed_")]


def mean_auc(idx, df):
    return np.mean([roc_auc_score(y[idx], df[c].values[idx]) for c in sc])


allidx = np.arange(len(y))
obs = mean_auc(allidx, a) - mean_auc(allidx, b)
rng = np.random.default_rng(42)
bs = []
while len(bs) < 5000:
    i = rng.integers(0, len(y), len(y))
    if len(set(y[i])) == 2:
        bs.append(mean_auc(i, a) - mean_auc(i, b))
lo, hi = np.percentile(bs, [2.5, 97.5])
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(bs, bins=50, color="#66c2a5", edgecolor="black", alpha=0.85)
ax.axvline(obs, color="black", ls="--", lw=1.5, label=rf"Observed $\Delta$AUC = {obs:.3f}")
ax.axvline(0, color="red", ls=":", lw=1.5, label=r"$\Delta$AUC = 0")
ax.set_xlabel(r"$\Delta$AUC (NLP raw $-$ numeric Labs)"); ax.set_ylabel("Bootstrap resamples (of 5,000)")
ax.set_title("Bootstrap distribution of AUC difference, IHC-G arm", fontsize=10, fontweight="bold")
ax.text(0.02, 0.95, f"95% interval [{lo:+.3f}, {hi:+.3f}]", transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.85))
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(SUPP, "figS3_bootstrap_pvalue.pdf")); plt.close(fig)
print(f"Figure S3 done: observed {obs:.4f}, 95% interval [{lo:.4f}, {hi:.4f}]")

# ---------------------------------------------------------------- Figure S4: single-partition OvO vs DyAM, 21 configurations
R = {}
for f in sorted(glob.glob(os.path.join(RES, "ref_all21_part*.json"))):
    R.update(json.load(open(f, encoding="utf-8")))
K = {"TMB": 1, "PDL1": 1, "IHC-A": 1, "Gen": 1}
xs = [R[c]["dyam"]["single_run"] for c in R]
ys = [R[c]["ovo"]["single_run"] for c in R]
col = ["grey" if c in K else ("#3182bd" if len(c.split("+")) == 2 else "#de2d26") for c in R]
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(xs, ys, c=col, edgecolor="black", s=60, zorder=3)
lims = [0.59, 0.84]
ax.plot(lims, lims, "k--", lw=1, zorder=1); ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
ax.set_xlabel("AUC, original cooperative attention"); ax.set_ylabel("AUC, OvO competitive attention")
hd = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=m, markeredgecolor="black", markersize=8, label=l)
      for m, l in (("grey", "1 source"), ("#3182bd", "2 sources"), ("#de2d26", r"$\geq$ 3 sources"))]
ax.legend(handles=hd, fontsize=8, loc="upper left"); fig.tight_layout()
fig.savefig(os.path.join(SUPP, "figS4_ovo_scatter.pdf")); plt.close(fig)
print("Figure S4 done", len(xs), "configurations")
