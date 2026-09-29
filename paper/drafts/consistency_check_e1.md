## Step E1 — Consistency Check

Review of all `sections/*.tex` against the 5 criteria below. Issues found
were fixed directly in the source files; remaining items are flagged for
the user.

### 1. Numbers consistent across text and tables

- AUC values, 95\% CIs, $\Delta$AUC, Cox HRs, tdAUC, C-index, IBS, and KM
  $\chi^2$/$p$ values quoted in `results.tex` and `discussion.tex` were
  cross-checked against `tables/table2_auc_ihca.tex`,
  `tables/table3_auc_ihcg.tex`, `tables/table4_survival.tex`, and
  `tables/table5_km.tex`. All values match (e.g., AUC~$=0.813$
  [0.753--0.874] for IHC-G+NLP raw appears identically in
  `results.tex:111`, `discussion.tex:11`, the abstract, and
  `table3_auc_ihcg.tex`).
- The OvO best-configuration value (AUC~$=0.8003$, rounded to $0.800$ in the
  abstract/conclusion and $0.8003$ in `results.tex`) is consistent —
  the abstract intentionally rounds to 3 decimals for readability while
  `results.tex` retains 4 decimals to match `tables/table_ovo` precision.

### 2. Terminology — "DyAM"

- "Dynamic Attention Multimodal (DyAM)" is spelled out on first use in
  `introduction.tex` (§1, paragraph 2) and again on first use within
  `methods.tex` (§Model Architecture, as is conventional for a
  self-contained Methods section). All subsequent mentions use "DyAM"
  alone. No mixed forms ("dynamic attention model", "DyAM framework" used
  loosely as a synonym) were found that needed correction.

### 3. Abbreviation define-on-first-use

Checked: NSCLC, ICI, AUC, CI, HR, PFS, ECOG, dNLR, IBS, GLCM, PCA, OvO, NLP,
TPS, TMB, MIRPON, IRB, DyAM.

- **Fixed**: `ECOG` was used in `methods.tex` (§Data Modalities, clinical
  labs list) without expansion. Changed to "Eastern Cooperative Oncology
  Group (ECOG) performance status" on first use; all later mentions
  (`methods.tex` §NLP Encoding, Supplementary Note S1) now correctly refer
  to the already-defined abbreviation.
- All other abbreviations are defined on first use in `introduction.tex` or
  `methods.tex` (AUC/CI/C-index are defined in §Statistical Analysis
  (`methods.tex`), which is standard practice even though AUC appears
  earlier in the Introduction in its commonly understood abbreviated form —
  acceptable per MDPI style for ubiquitous statistical terms).

### 4. Tense consistency

- **Methods**: consistently past tense ("were extracted", "was applied",
  "were evaluated") — confirmed throughout `methods.tex`.
- **Results**: consistently past tense ("achieved", "outperformed",
  "showed") — confirmed throughout `results.tex`.
- **Discussion**: mixed present (general claims, e.g., "This is best
  interpreted as...", "OvO attention is best suited to...") and past
  (study-specific findings, e.g., "OvO improved AUC in half of the tested
  configurations") — consistent with standard Discussion conventions.

### 5. Passive vs. active voice

- **Methods**: predominantly passive ("Prompts were encoded with...",
  "PCA was applied to...") — consistent with MDPI Methods style.
- **Discussion**: mixed, with active constructions used for interpretive
  claims ("We propose that...", "This suggests that...") — appropriate
  for a Discussion section.

### Outstanding items (not fixable without external input/data)

These are intentional placeholders, not consistency errors, and are
unchanged by this review:

- Author names/affiliations, funding statement, IRB protocol number,
  GitHub repository URL (`main.tex`).
- `[TO VERIFY]` validation-cohort demographic values in
  `tables/table1_patients.tex` and the L1-filter retention counts in
  `supplementary/supplementary.tex` Table~S2.
- `DyAM2024` reference entry (currently `@unpublished{...}`,
  "manuscript in preparation") — see `drafts/references_todo.md` item 3.
