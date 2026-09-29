## Figure Specifications — Steps C1, C2, C3

These specs are intended for a designer/illustrator (Figs.\ 1–2) or for
direct generation from existing analysis outputs (Figs.\ 3–5, which can be
plotted directly from `summary_df` / survival outputs already produced by
`lung_helpers.py` / `survival_analysis_nlp.py`).

---

### Figure 1 — Study Overview Diagram

**Layout:** 3 horizontal panels connected by arrows (left → right flow),
recommended size 180~mm × 60~mm (full two-column width).

- **Panel A — "Patient Cohort"**: box listing the discovery cohort
  (n = 247, advanced NSCLC, ICI-treated) and two validation cohorts
  (n = 50 radiomics-evaluable, n = 71 pathology-evaluable). Below the box,
  five small icons represent the input modalities: CT scan (radiomics),
  microscope/tissue slide (pathology IHC-A/IHC-G), DNA helix (genomics),
  PD-L1 stain icon, and a document/text icon (clinical labs).
- **Panel B — "DyAM Architecture"**: $N$ modality icons feed into a central
  box labelled "Per-modality risk score $r_i$ + Attention $a_i$", with two
  branches shown side by side: "Cooperative (\texttt{AttentionMatrix})" —
  softmax/L1-normalised — and "Competitive (\texttt{AttentionMatrixOvO})" —
  one-vs-others sigmoid. Both branches converge into a single output node
  labelled "$\hat{y} = \sum_i r_i a_i$".
- **Panel C — "Evaluation"**: four small icons/labels representing the
  evaluation endpoints — ROC curve (AUC), concordance icon (C-index), forest
  plot icon (Cox HR), and a Kaplan--Meier step-curve icon (KM log-rank).

**Caption (Figure 1):**
"\textbf{Figure 1.} Study overview. (A) The discovery cohort
($n = 247$) and two validation cohorts (radiomics-evaluable, $n = 50$;
pathology-evaluable, $n = 71$) provide up to seven data modalities per
patient: CT radiomics, pathology image texture (IHC-A/IHC-G), tumour
genomics, PD-L1 TPS, and structured clinical variables. (B) The DyAM
architecture computes a per-modality risk score and combines modalities
using either cooperative (\texttt{AttentionMatrix}) or competitive
one-vs-others (\texttt{AttentionMatrixOvO}) attention to produce a fused
risk prediction $\hat{y}$. (C) Models are evaluated on both binary response
classification (AUC-ROC with DeLong 95\% CI) and survival endpoints
(Harrell's C-index, multivariate Cox hazard ratios, and Kaplan--Meier
log-rank stratification)."

---

### Figure 2 — NLP Clinical Encoding Pipeline

**Layout:** single horizontal flow diagram, recommended size 180~mm ×
50~mm, with a sidebar showing an example text prompt.

**Flow (left to right):**
"13 numeric clinical variables" → \texttt{df\_to\_text\_prompts()} →
"English text prompt" → \texttt{all-MiniLM-L6-v2} (sentence transformer) →
"384-dim embedding" → \texttt{PCA (n=16)} → "16-dim NLP-PCA16".

**Sidebar (example prompt, abbreviated to 2 lines):**
"\textit{``Patient is 68 years old, male, ECOG performance status 1,
albumin 3.8 g/dL, neutrophil-to-lymphocyte ratio 4.2, ... no liver
metastases.''}"

**Caption (Figure 2):**
"\textbf{Figure 2.} NLP-based encoding of structured clinical features.
Thirteen numeric clinical variables per patient are converted into a
natural-language sentence describing the patient's clinical profile
(example shown), which is then encoded into a 384-dimensional embedding
using a pretrained sentence transformer (all-MiniLM-L6-v2). The 384-dim
embedding (``NLP raw'') is used directly, or further reduced to 16
dimensions via PCA (``NLP-PCA16'', retaining $\approx$80\% of variance) prior
to fusion with other modalities. \texttt{no\_scale=True} is applied to NLP
embeddings to preserve their cosine-normalised geometry."

---

### Figure 3 — AUC Comparison Bar Charts

**Layout:** two side-by-side panels (3A, 3B), recommended size 180~mm ×
70~mm.

- **Panel 3A — IHC-A arm**: bar chart, x-axis = 7 model variants
  (No clinical, +Labs, +NLP raw, +NLP-PCA16, +Labs+NLP, +Labs+NLP-PCA16,
  +BioClinBERT-PCA16), y-axis = AUC (range 0.70–0.85), error bars = DeLong
  95\% CI. Bar colours grouped by category: No clinical = grey, +Labs =
  orange, NLP variants (raw/PCA16/Labs+NLP/Labs+NLP-PCA16) = shades of green,
  BioClinBERT-PCA16 = purple. A horizontal dashed reference line at
  AUC~$=0.80$ marks the conventional "good discrimination" threshold.
- **Panel 3B — IHC-G arm**: same style, x-axis = 4 model variants
  (No clinical, +Labs, +NLP raw, +NLP-PCA16), y-axis range 0.70–0.85, same
  colour scheme and AUC~$=0.80$ reference line. The +NLP raw bar (best
  overall model) is annotated with $^\dagger$.

**Caption (Figure 3):**
"\textbf{Figure 3.} AUC comparison across clinical encoding strategies for
(A) the IHC-A arm (7 variants) and (B) the IHC-G arm (4 variants). Bars show
pooled out-of-fold AUC from 10-fold cross-validation ($n = 247$); error bars
denote DeLong 95\% confidence intervals. The dashed horizontal line at
AUC~$=0.80$ indicates the conventional threshold for clinically useful
discrimination. $^\dagger$Best-performing model overall (DyAM
Rad+IHC-G+Gen+PDL1+NLP raw, AUC~$=0.813$)."

---

### Figure 4 — Kaplan--Meier Survival Curves

**Layout:** 2$\times$2 grid, recommended size 180~mm × 160~mm, each panel
with a number-at-risk table below the curves.

- **4A — IHC-A, No Clinical**: KM curves stratified at risk score~$=0$
  (high- vs.\ low-risk), x-axis = PFS (months, 0–24), y-axis = survival
  probability (0–1.0). Annotation: $\chi^2 = 28.17$, $p < 0.005$.
- **4B — IHC-A, + NLP-PCA16**: same layout. Annotation: $\chi^2 = 23.74$,
  $p < 0.005$.
- **4C — IHC-G, No Clinical**: same layout. Annotation: $\chi^2 = 20.07$,
  $p < 0.005$.
- **4D — IHC-G, + NLP raw** (highlighted panel, e.g., bold border):
  Annotation: $\chi^2 = 28.92^{\star}$, $p < 0.005$ (highest $\chi^2$ across
  all tested models).

**Caption (Figure 4):**
"\textbf{Figure 4.} Kaplan--Meier progression-free survival curves for
representative models, stratified at risk score~$=0$ into high-risk
(red) and low-risk (blue) groups, with number-at-risk tables. (A) IHC-A
arm, no clinical encoding ($\chi^2 = 28.17$). (B) IHC-A arm with
NLP-PCA16 clinical encoding ($\chi^2 = 23.74$). (C) IHC-G arm, no clinical
encoding ($\chi^2 = 20.07$). (D) IHC-G arm with NLP raw clinical encoding
($\chi^2 = 28.92$, the highest log-rank statistic among all tested models).
All comparisons reach $p < 0.005$ by the log-rank test."

---

### Figure 5 — Survival Analysis Summary

**Layout:** 2$\times$2 grid, recommended size 180~mm × 160~mm.

- **5A — C-index comparison**: bar chart of Harrell's C-index with bootstrap
  95\% CI error bars (1{,}000 resamples) for the 8 models in
  Table~\ref{tab:survival} (4 IHC-A + 4 IHC-G), grouped by arm.
- **5B — Time-dependent AUC**: line plot, x-axis = evaluation time
  (6, 12, 18 months), y-axis = tdAUC, one line per clinical-encoding variant
  (No clinical, +Labs, +NLP raw, +NLP-PCA16), separate sub-panels or line
  styles for IHC-A vs.\ IHC-G.
- **5C — Forest plot**: multivariate Cox hazard ratios (point estimate +
  95\% CI, log scale x-axis) for the risk-score covariate across the 8
  models, adjusted for age, ECOG, albumin, dNLR, and liver metastases;
  vertical reference line at HR~$=1$.
- **5D — Integrated Brier Score**: bar chart of IBS for the 8 models, with a
  horizontal dashed reference line at IBS~$=0.25$ (random-guess baseline);
  lower bars indicate better calibration.

**Caption (Figure 5):**
"\textbf{Figure 5.} Survival analysis summary across clinical encoding
strategies. (A) Harrell's C-index with bootstrap 95\% confidence intervals
for IHC-A and IHC-G arms; differences between encoding strategies were not
statistically significant (paired Wilcoxon $p > 0.05$). (B) Time-dependent
AUC at 6, 12, and 18~months; NLP-encoded models show the largest advantage
at 12--18~months. (C) Forest plot of multivariate Cox proportional hazards
ratios for the fused risk score, adjusted for age, ECOG performance status,
albumin, derived neutrophil-to-lymphocyte ratio, and liver metastases (all
$p < 0.001$). (D) Integrated Brier Score (IBS) for each model; all values
fall well below the 0.25 random-guess threshold (range 0.172--0.175),
indicating good calibration across all encoding strategies."
