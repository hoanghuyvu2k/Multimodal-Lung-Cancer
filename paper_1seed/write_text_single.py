"""Viết lại discussion/conclusion/abstract/methods/introduction/supplementary cho bản MỘT PHÂN HOẠCH."""


def sub(path, pairs):
    s = open(path, encoding='utf-8').read()
    for a, b in pairs:
        assert s.count(a) == 1, (path, a[:70], s.count(a))
        s = s.replace(a, b)
    open(path, 'w', encoding='utf-8').write(s)


# ---------------------------------------------------------------- discussion (viết lại toàn bộ)
old = open('sections/discussion.tex', encoding='utf-8').read()
a = old.index("From a clinical implementation perspective")
b = old.index("\\paragraph{Limitations.}")
impl = old[a:b]
lim_tail = old[b:]
lim_tail = lim_tail[lim_tail.index("Second, with $n = 247$"):]
disc = r"""%% sections/discussion.tex — single-partition version
\section{Discussion}
\label{sec:discussion}

In this study, we extended the DyAM multimodal attention framework with two
complementary contributions: a competitive (OvO) attention mechanism with fewer
parameters and an NLP-based encoding of tabular clinical features, and evaluated both
using binary classification and survival endpoints in a 247-patient cohort, from a single
10-fold cross-validation partition.

The strongest configuration, OvO with NLP-encoded clinical variables at the primary
benchmark (Rad+IHC-G+Gen+PDL1), reached an AUC of $0.824$ (95\% CI $0.765$--$0.883$);
under the cooperative attention of the original framework the corresponding value was $0.803$
($0.738$--$0.868$), and the cooperative model without clinical variables, the configuration of the
original publication, reached $0.783$.
These figures are close to or above the AUC of $0.80$ reported for the original
framework \cite{Vanguri2022}, and above PD-L1~TPS alone ($0.708$) and TMB alone ($0.615$)
in the same cohort, consistent with the modest discrimination reported for these
markers \cite{Lu2019,Rizvi2015}; the interval, however, is wide ($\pm 0.06$) and includes
values below $0.80$.
A harder evaluation gave lower estimates: with
training on the patients with the earliest CT accession numbers and testing on the later ones
(a proxy for time, as calendar dates were not available), OvO+NLP reached an AUC of $0.703$
(95\% bootstrap interval $0.587$--$0.819$) and DyAM+NLP $0.700$ ($0.587$--$0.812$) on the 124
later patients (21 responders); at seven sources the values were $0.707$ and $0.678$.
The intervals overlap almost completely.
The cross-validated figure should therefore not be read as the expected performance in
new patients, and, because it comes from a single partition, it may be favourable.
Further, the 247 patients were used to compare roughly 60 architecture and training
variants during development, so the largest figures partly reflect selection;
we estimate this optimism at $0.014$--$0.031$ AUC (the expected maximum of 52 candidates with no true
difference, from the standard error of score differences between closely related models in
repeated-partition runs that are not reported here).

\paragraph{OvO versus cooperative attention.}
Without NLP, OvO's AUC was higher than that of cooperative attention in 6 of 21 configurations,
lower in 11, and identical in 4 single-source configurations where the two formulations
coincide; the difference at configurations fusing five or more sources averaged $-0.004$.
With NLP-clinical encoding, OvO+NLP was at least as high as DyAM+NLP in 6 of the 8 configurations with five or more
sources (mean $+0.006$), and at the primary benchmark by $+0.021$.
All of these differences are smaller than the width of the confidence intervals
($\approx 0.14$); we therefore describe the result as comparable performance with a simpler
parameterisation, and as a possible, untested advantage when many sources are combined, not as a demonstrated superiority.
A plausible explanation is parameter economy: with the 384-dimensional NLP source
the cooperative model had 4{,}969 trainable parameters against 1{,}242 for OvO.
We did not test this explanation directly.

\paragraph{NLP encoding of clinical variables.}
The finding that held in both analyses was that the NLP encoding exceeded
numeric labs: by $+0.018$ and $+0.019$ in the two pathology arms under cooperative attention
and by $+0.019$ to $+0.033$ in the two seven-source configurations containing labs under both
attention mechanisms.
We propose that sentence embeddings capture non-linear interactions between
clinical variables --- for example, the combined implication of advanced age,
low albumin, and poor performance status --- that are not explicitly represented in a
13-dimensional numeric vector, whereas the attention mechanism would otherwise have to learn
these interactions from a small training set \cite{Reimers2019,Wang2020MiniLM}.
This is a hypothesis; our data do not test it.
The gain over omitting clinical variables altogether was smaller ($+0.014$ in the arm analysis and
$+0.020$ to $+0.024$ at the primary benchmark of the 21-configuration analysis), within the confidence intervals;
we therefore do not claim a benefit of NLP encoding over leaving out clinical variables.
The domain-specific BioClinBERT encoder was lower than the general-purpose sentence
encoder (Supplementary Note~S2).

\paragraph{Survival analysis.}
The clinical encodings did not change the survival metrics consistently: the hazard ratio of the risk score
was $5.0$--$6.3$ for every variant and was lower with labs than without in the IHC-A arm, the C-index was
$0.62$--$0.64$ with the highest value for numeric labs in both arms, and the log-rank $\chi^2$ was above
$17$ for every model.
Time-dependent AUC at 6, 12 and 18~months was within about $0.05$ across encodings and was highest at 18~months
without clinical variables.
We therefore do not retain the interpretation that NLP encoding captures prognostic rather than predictive information.

""" + impl + r"""\paragraph{Limitations.}
First, all results come from a single 10-fold cross-validation partition of a single-institution
retrospective cohort, which limits generalisability, may introduce selection bias, and means that
the figures may be more or less favourable than the average over other partitions; the confidence
intervals reflect patient sampling but not the choice of partition.
""" + lim_tail
disc = disc.replace("Second, with $n = 247$", "Second, with $n = 247$")
open('sections/discussion.tex', 'w', encoding='utf-8').write(disc)

# ---------------------------------------------------------------- conclusion
con = open('sections/conclusion.tex', encoding='utf-8').read()
i = con.index("Under repeated 10-fold partitions")
j = con.index("Validation in larger")
new = r"""In a single 10-fold partition, OvO attention was comparable to cooperative attention in mean AUC across configurations
and, with the NLP encoding added, reached $0.824$ (95\% CI 0.765--0.883) at the primary benchmark against $0.803$
for cooperative attention, with about a quarter of the trainable parameters.
These differences are smaller than the confidence intervals, so they indicate comparable performance with a
simpler model rather than an established advantage.
NLP encoding of clinical variables with a general-purpose sentence transformer exceeded
numeric labs by $+0.018$ and $+0.019$ AUC in both pathology arms and by $+0.019$ to $+0.033$ in both seven-source
configurations containing labs; its advantage over omitting clinical variables was small and is
unresolved, and the clinical encodings did not change the survival metrics consistently.
An order-based hold-out evaluation gave lower estimates ($0.703$ for OvO+NLP and $0.700$ for
DyAM+NLP), so cross-validated AUCs, which here come from a single partition, should not be read as expected
performance in new patients.
"""
open('sections/conclusion.tex', 'w', encoding='utf-8').write(con[:i] + new + con[j:])

# ---------------------------------------------------------------- main.tex abstract
m = open('main.tex', encoding='utf-8').read()
a = m.index("  In a 247-patient discovery cohort, repeated-seed evaluation")
b = m.index("}\n\n\\keyword{%")
abs_new = r"""  In a 247-patient discovery cohort, a single 10-fold cross-validation partition
  evaluated 21 modality combinations. Without NLP, OvO had the higher AUC than cooperative attention in
  6 of 21 combinations (lower in 11); with the NLP encoding added, OvO was at least as high in 6 of the
  8 combinations fusing five or more sources (mean difference $+0.006$ AUC) and reached AUC~$0.824$
  (95\% CI 0.765--0.883) against $0.803$ (0.738--0.868) for cooperative attention at the primary benchmark.
  These differences are smaller than the confidence intervals.
  Replacing numeric clinical labs by the NLP encoding raised the AUC by
  $+0.018$ and $+0.019$ in two pathology arms, while its advantage over omitting
  clinical variables was small ($+0.014$) and is regarded as unresolved.
  Clinical encodings did not change the Cox hazard ratio of the risk score (5.0--6.3)
  or the C-index (0.62--0.64) consistently.
  Performance on an order-based hold-out partition was lower (0.703 for OvO+NLP and
  0.700 for DyAM+NLP).
"""
open('main.tex', 'w', encoding='utf-8').write(m[:a] + abs_new + m[b:])

# ---------------------------------------------------------------- methods / introduction
sub('sections/methods.tex', [
("Differences between models are reported as the difference in mean AUC across\nrepeated partitions (5~seeds~$\\times$~10-fold, or 20 partitions where stated), the\nnumber of configurations or partitions favouring each model, and, where patient-level\nscores were retained, a 95\\% patient-level bootstrap interval (5000 resamples) of\nthe difference in ensemble AUC. Because the cohort is small, hypothesis tests were not\nused to claim that one model is superior; DeLong intervals are reported for single\nAUC values only. Ensemble scores average, for each patient, the per-partition scores\nafter standardisation within each partition.",
 "All results come from a single 10-fold cross-validation partition (seed~42, natural patient order, the\nprotocol of the original publication). Differences between models are reported as the difference in AUC and the\nnumber of configurations favouring each model. Because the cohort is small, hypothesis tests were not\nused to claim that one model is superior; DeLong intervals are reported for single AUC values."),
("and the survival analyses were repeated for\n5~seeds~$\\times$~10-fold partitions, reporting mean $\\pm$ s.d.\n", "from the same partition.\n"),
("was computed for each of the five partitions\n(", "was computed from the same partition\n("),
])
sub('sections/introduction.tex', [
("attention across 20 modality combinations using 10-fold cross-validation,\nwith a repeated-seed evaluation (5~seeds~$\\times$~10-fold) across\n21 modality combinations, without and with the clinical encoding below, to\ntest the reproducibility of any observed difference.",
 "attention across 21 modality combinations using 10-fold cross-validation,\nwithout and with the clinical encoding below."),
("number of configurations or partitions favouring each model and interval estimates,\nand we state where they are of the same order as the variability between partitions.",
 "number of configurations favouring each model and interval estimates,\nand we state where they are within the confidence intervals."),
])
print('main texts done')
