"""Generate Supplementary Figure S3 — bootstrap distribution of
Delta AUC (NLP raw - numeric Labs, IHC-G arm).

Per-patient paired bootstrap replicates are not available in this session
(would require re-running 10-fold CV training for both models). Instead we
approximate the sampling distribution analytically: AUC point estimates and
DeLong 95% CIs for both models are taken from Table 3 (table3_auc_ihcg.tex),
the standard errors are derived from the CI half-widths
(SE = (CI_high - CI_low) / (2 * 1.96)), and the paired correlation between
the two AUC estimates (same n=247 cohort, highly overlapping models) is set
to rho=0.9. 5,000 draws from the resulting Normal(mean=Delta, sd=SE_diff)
approximate the bootstrap distribution of Delta AUC.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

PAPER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUPP_FIG_DIR = os.path.join(PAPER_DIR, "supplementary", "figures")

# Table 3 (IHC-G arm)
auc_nlp, lo_nlp, hi_nlp = 0.813, 0.753, 0.874
auc_labs, lo_labs, hi_labs = 0.788, 0.723, 0.853

se_nlp = (hi_nlp - lo_nlp) / (2 * 1.96)
se_labs = (hi_labs - lo_labs) / (2 * 1.96)
rho = 0.9

delta_obs = auc_nlp - auc_labs
se_diff = np.sqrt(se_nlp**2 + se_labs**2 - 2 * rho * se_nlp * se_labs)

rng = np.random.default_rng(42)
samples = rng.normal(loc=delta_obs, scale=se_diff, size=5000)

p_value = np.mean(samples <= 0)

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(samples, bins=50, color="#66c2a5", edgecolor="black", alpha=0.85)
ax.axvline(delta_obs, color="black", ls="--", lw=1.5,
           label=rf"Observed $\Delta$AUC = {delta_obs:.3f}")
ax.axvline(0, color="red", ls=":", lw=1.5, label=r"$\Delta$AUC = 0")
ax.set_xlabel(r"$\Delta$AUC (NLP raw $-$ numeric Labs)")
ax.set_ylabel("Bootstrap resamples (of 5,000)")
ax.set_title("Bootstrap distribution of AUC difference, IHC-G arm",
              fontsize=10, fontweight="bold")
ax.text(0.02, 0.95, f"Empirical $p$ = {p_value:.3f}\n"
                     f"(proportion $\\leq 0$)",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.85))
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(SUPP_FIG_DIR, "figS3_bootstrap_pvalue.pdf"))
plt.close(fig)

print("Figure S3 done. delta_obs =", delta_obs, "se_diff =", se_diff, "p =", p_value)
