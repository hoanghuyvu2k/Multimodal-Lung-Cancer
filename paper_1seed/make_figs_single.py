"""Hình cho bản bài MỘT PHÂN HOẠCH (seed 42, thứ tự tự nhiên; ref-linux-py38). Chạy từ paper_1seed/:  python make_figs_single.py
Hình 3 do generate_fig3_demo.py (đọc cùng JSON). Ở đây: Hình 4 (KM), Hình 5, Hình S3 (bootstrap NLP raw − Labs), Hình S4 (scatter OvO vs DyAM)."""
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
RES = os.path.join(os.path.dirname(HERE), "experiments", "results")
SG = os.path.join(RES, "ref", "single")
FIG, SUPP = os.path.join(HERE, "figures"), os.path.join(HERE, "supplementary", "figures")
CLIN = os.path.join(os.path.dirname(os.path.dirname(HERE)), "datasets",
                    "18193MSKMINDProjectM-OmnibusInventory_DATA_2021-12-20_1540-WITH-TB-and-SCANNER.csv")
clin = pd.read_csv(CLIN).drop_duplicates("dmp_pt_id").set_index("dmp_pt_id")
S = json.load(open(os.path.join(SG, "single_partition.json"), encoding="utf-8"))["clinical_variants"]
TD = json.load(open(os.path.join(SG, "tdauc_single.json"), encoding="utf-8"))
BS = json.load(open(os.path.join(SG, "bm1_survival_single.json"), encoding="utf-8"))
BT = json.load(open(os.path.join(SG, "tdauc_bm1_single.json"), encoding="utf-8"))
BM1N = {"dyam": "DyAM", "ovo": "OvO", "dyam_nlp": "DyAM+NLP", "ovo_nlp": "OvO+NLP"}
STEM = {"No clinical": "no_clinical", "+ Labs": "labs", "+ NLP raw": "nlp_raw", "+ NLP-PCA16": "nlppca16"}


def km_panel(fname, arm, variant, title, outdir=None):
    stem = f"bm1_{variant}" if arm == "BM1" else f"{arm}__{STEM[variant]}"
    sc = pd.read_csv(os.path.join(SG, "scores", f"{stem}.csv"), index_col=0)
    c = clin.reindex(sc.index)
    s = sc["score"]
    hi = (s > s.median()).values
    t, e = c["pfs"].astype(float).values, c["pfs_censor"].astype(int).values
    r = logrank_test(t[hi], t[~hi], event_observed_A=e[hi], event_observed_B=e[~hi])
    big = outdir is None          # main-text Figure 5 panels are shown small in a 2x2 composite -> larger fonts
    fs = dict(title=13, chi=12, leg=11, tab=10, lab=12, tick=11) if big else dict(title=10, chi=9, leg=8, tab=7, lab=10, tick=10)
    fig, ax = plt.subplots(figsize=(5, 5.2))
    for m, lab, col in ((~hi, "Low risk", "#3182bd"), (hi, "High risk", "#de2d26")):
        KaplanMeierFitter().fit(t[m], e[m], label=lab).plot_survival_function(ax=ax, color=col, ci_show=True)
    ax.set_xlim(0, 24); ax.set_ylim(0, 1.0); ax.set_xlabel("Time (months)", fontsize=fs["lab"]); ax.set_ylabel("PFS probability", fontsize=fs["lab"])
    ax.tick_params(labelsize=fs["tick"])
    ax.set_title(title, fontsize=fs["title"], fontweight="bold")
    ax.text(0.95, 0.92, rf"log-rank $\chi^2={r.test_statistic:.1f}$", transform=ax.transAxes, ha="right", va="top", fontsize=fs["chi"],
            bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.8))
    ax.legend(fontsize=fs["leg"], loc="upper right", bbox_to_anchor=(0.98, 0.80)) if big else ax.legend(fontsize=fs["leg"], loc="lower left")
    tp = [0, 6, 12, 18, 24]
    cells = [[str(int((t[~hi] >= x).sum())) for x in tp], [str(int((t[hi] >= x).sum())) for x in tp]]
    tb = ax.table(cellText=cells, rowLabels=["Low risk", "High risk"], colLabels=[str(x) for x in tp], cellLoc="center", bbox=[0.0, -0.42, 1.0, 0.18])
    tb.auto_set_font_size(False); tb.set_fontsize(fs["tab"])
    fig.subplots_adjust(bottom=0.32, left=0.19 if big else 0.125); fig.savefig(os.path.join(outdir or FIG, fname)); plt.close(fig)
    print(fname, f"chi2={r.test_statistic:.2f}")


for f, a, v, ti in (("figS6a_km_ihca_noclin.pdf", "IHC-A", "No clinical", "IHC-A, no clinical"),
                    ("figS6b_km_ihca_nlppca16.pdf", "IHC-A", "+ NLP-PCA16", "IHC-A, + NLP-PCA16"),
                    ("figS6c_km_ihcg_noclin.pdf", "IHC-G", "No clinical", "IHC-G, no clinical"),
                    ("figS6d_km_ihcg_nlpraw.pdf", "IHC-G", "+ NLP raw", "IHC-G, + NLP raw")):
    km_panel(f, a, v, ti, SUPP)
for f, v, ti in (("fig4a_km_bm1_dyam.pdf", "dyam", "BM1, DyAM"), ("fig4b_km_bm1_ovo.pdf", "ovo", "BM1, OvO"),
                 ("fig4c_km_bm1_dyam_nlp.pdf", "dyam_nlp", "BM1, DyAM+NLP"), ("fig4d_km_bm1_ovo_nlp.pdf", "ovo_nlp", "BM1, OvO+NLP")):
    km_panel(f, "BM1", v, ti)

models, arm, rows = [], [], []
for a in ("IHC-A", "IHC-G"):
    for v in STEM:
        x = S[f"{a}__{STEM[v]}"]
        models.append(f"{v.replace('No clinical', 'No Clinical')} ({a})"); arm.append(a)
        rows.append(dict(ci=x["cindex"], hr=x["cox_hr"], km=x["km_chi2"], td=TD[f"{a}__{STEM[v]}"]))
cols = {"IHC-A": "#4393c3", "IHC-G": "#d6604d"}
bc = [cols[a] for a in arm]
x = np.arange(len(models))
handles = [plt.Rectangle((0, 0), 1, 1, color=cols[a]) for a in cols]

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(x, [r["ci"] for r in rows], color=bc, edgecolor="black", linewidth=0.6)
ax.set_xticks(x); ax.set_xticklabels(models, rotation=35, ha="right", fontsize=7.5)
ax.set_ylabel("Harrell's C-index"); ax.set_ylim(0.50, 0.75); ax.axhline(0.5, ls="--", color="grey", lw=1)
ax.legend(handles, list(cols), fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(SUPP, "figS7a_cindex.pdf")); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), sharey=True)
times, sty = [6, 12, 18], ["o-", "s--", "^-.", "d:"]
for ax, a, idx in zip(axes, cols, (range(0, 4), range(4, 8))):
    for k, i in enumerate(idx):
        ax.plot(times, [rows[i]["td"][str(t)] for t in times], sty[k], label=list(STEM)[k])
    ax.set_title(a, fontsize=10, fontweight="bold"); ax.set_xlabel("Time (months)"); ax.set_xticks(times)
axes[0].set_ylabel("Time-dependent AUC"); axes[0].set_ylim(0.60, 0.78); axes[1].legend(fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(SUPP, "figS7b_tdauc.pdf")); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
y = np.arange(len(models))[::-1]
for yi, i in zip(y, range(len(models))):
    ax.plot(rows[i]["hr"], yi, "o", color=bc[i], markersize=8, markeredgecolor="black")
ax.axvline(1, ls="--", color="black", lw=1); ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels(models, fontsize=8)
ax.set_xlabel("Hazard ratio per unit risk score, log scale"); ax.set_xlim(1, 15)
fig.tight_layout(); fig.savefig(os.path.join(SUPP, "figS7c_cox_forest.pdf")); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(x, [r["km"] for r in rows], color=bc, edgecolor="black", linewidth=0.6)
ax.axhline(3.84, ls="--", color="black", lw=1, label=r"$\chi^2 = 3.84$ ($p = 0.05$)")
ax.set_xticks(x); ax.set_xticklabels(models, rotation=35, ha="right", fontsize=7.5)
ax.set_ylabel(r"Log-rank $\chi^2$"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(SUPP, "figS7d_km_chi2.pdf")); plt.close(fig)
print("Figure 5 done")


# ---- Figure 5 (new): survival metrics over the 21 combinations, DyAM vs OvO (+NLP in the two labs configurations)
S21 = {}
for f in glob.glob(os.path.join(SG.replace("single", "single21"), "surv_part*.json")):
    S21.update(json.load(open(f, encoding="utf-8")))
T21 = json.load(open(os.path.join(SG.replace("single", "single21"), "tdauc_21.json"), encoding="utf-8"))
order = sorted(S21.items(), key=lambda kv: kv[1]["idx"])
idxs = [v["idx"] for _, v in order]
xx = np.arange(len(idxs))
def panel(fname, getter, ylabel, ylim=None, hline=None):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    w = 0.2
    for j, (m, lab, col) in enumerate((("dyam", "DyAM", "#4393c3"), ("ovo", "OvO", "#d6604d"), ("dyam_nlp", "DyAM+NLP", "#92c5de"), ("ovo_nlp", "OvO+NLP", "#f4a582"))):
        vals = []
        for _, v in order:
            mm = m if m in v else m.replace("_nlp", "")
            vals.append(getter(v, mm))
        ax.bar(xx + (j - 1.5) * w, vals, w, label=lab, color=col, edgecolor="black", linewidth=0.3)
    ax.set_xticks(xx); ax.set_xticklabels(idxs, fontsize=10); ax.set_xlabel("Configuration (Table 6 row)", fontsize=12); ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    if ylim: ax.set_ylim(*ylim)
    if hline is not None: ax.axhline(hline, ls="--", color="grey", lw=1)
    ax.legend(fontsize=10, ncol=4, loc="upper left", columnspacing=0.8, handlelength=1.2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname)); plt.close(fig)
panel("fig5a_cindex.pdf", lambda v, m: v[m]["cindex"], "Harrell's C-index", (0.5, 0.72), 0.5)
panel("fig5b_tdauc.pdf", lambda v, m: T21[f"{v['idx']:02d}__{m}"]["12"], "Time-dependent AUC (12 mo)", (0.5, 0.85))
panel("fig5c_cox_forest.pdf", lambda v, m: v[m]["cox_hr"], "Cox HR per unit risk score", (0, 10), 1)
panel("fig5d_km_chi2.pdf", lambda v, m: v[m]["km_chi2"], r"Log-rank $\chi^2$", (0, 36), 3.84)
print("Figure 5 (21 configurations) done")

# S3: bootstrap NLP raw - Labs, IHC-G, single partition
a = pd.read_csv(os.path.join(SG, "scores", "IHC-G__nlp_raw.csv"), index_col=0)
b = pd.read_csv(os.path.join(SG, "scores", "IHC-G__labs.csv"), index_col=0).loc[a.index]
yv = a["label"].values
obs = roc_auc_score(yv, a["score"]) - roc_auc_score(yv, b["score"])
rng = np.random.default_rng(42)
bs = []
while len(bs) < 5000:
    i = rng.integers(0, len(yv), len(yv))
    if len(set(yv[i])) == 2:
        bs.append(roc_auc_score(yv[i], a["score"].values[i]) - roc_auc_score(yv[i], b["score"].values[i]))
lo, hi_ = np.percentile(bs, [2.5, 97.5])
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(bs, bins=50, color="#66c2a5", edgecolor="black", alpha=0.85)
ax.axvline(obs, color="black", ls="--", lw=1.5, label=rf"Observed $\Delta$AUC = {obs:.3f}")
ax.axvline(0, color="red", ls=":", lw=1.5, label=r"$\Delta$AUC = 0")
ax.set_xlabel(r"$\Delta$AUC (NLP raw $-$ numeric Labs)"); ax.set_ylabel("Bootstrap resamples (of 5,000)")
ax.set_title("Bootstrap distribution of AUC difference, IHC-G arm", fontsize=10, fontweight="bold")
ax.text(0.02, 0.95, f"95% interval [{lo:+.3f}, {hi_:+.3f}]", transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.85))
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(SUPP, "figS3_bootstrap_pvalue.pdf")); plt.close(fig)
print(f"S3: observed {obs:.4f}, interval [{lo:.4f}, {hi_:.4f}]")
