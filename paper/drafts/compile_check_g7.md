## Step G7 — LaTeX Compile Check

`pdflatex`/`latexmk`/`texcount`/`detex` are not available in this
environment, so a full `pdflatex main → bibtex main → pdflatex main →
pdflatex main` run could not be executed here. Instead, the following
static checks were performed (scripts: `drafts/syntax_check.py`,
`drafts/wordcount.py`):

### Checks passed

- **Brace balance**: no unmatched `{`/`}` in any `.tex` file.
- **Environment balance**: every `\begin{...}` has a matching `\end{...}`
  (table, figure, equation, tabular, subfigure, minipage, document).
- **`\input{}` targets**: all 5 section files + `tables/table2`–`table5`
  resolve to existing files.
- **`\cite{}` keys**: all 15 distinct keys used in `sections/*.tex` exist in
  `references.bib`, and all 18 `references.bib` entries are cited at least
  once (no orphans, no undefined `TODO` DOIs remaining).
- **`\label{}`/`\ref{}`**: no duplicate labels; all `\ref{}`/`\eqref{}`
  targets in the text resolve to an existing `\label{}`. All five main
  figures (`fig:overview`, `fig:nlp_pipeline`, `fig:auc_comparison`,
  `fig:km_curves`, `fig:survival_summary`) and Table~`tab:km` are now
  explicitly referenced in the prose (previously `tab:km` and all figure
  labels were defined but unreferenced — fixed).

### Issue found and fixed

- **`tables/table1_patients.tex` was never `\input{}`** — the table file
  existed and was `\ref{tab:patients}`'d in `methods.tex`, but the content
  was not included anywhere in the document. Fixed by adding
  `\input{tables/table1_patients}` immediately after the reference in
  `sections/methods.tex` §Study Design and Patient Cohort.

### Items that cannot be verified without a LaTeX toolchain / external files

1. **`mdpi.cls` / `mdpi.bst`** are not present in `paper/`. These must be
   downloaded from the official MDPI LaTeX template
   (https://www.mdpi.com/authors/latex) and placed in `paper/` (or on the
   TeX search path) before `main.tex` will compile with
   `\documentclass[cancers,article,...]{mdpi}`. A fallback
   `\documentclass[12pt,a4paper]{article}` is commented out in `main.tex`
   for a quick non-MDPI-styled compile check.
2. **Figure files do not exist yet**: `figures/fig1_overview.pdf` through
   `fig5d_ibs.pdf` (10 files) and `supplementary/figures/figS1`–`figS5.pdf`
   (5 files) are referenced via `\includegraphics` but the `figures/` and
   `supplementary/figures/` directories are currently empty. These must be
   generated (per `drafts/figure_specs.md`) and placed at the referenced
   paths before a full PDF can be produced. Without these files, `pdflatex`
   will report "File `figures/fig1_overview.pdf` not found" errors but will
   still produce a PDF with placeholder boxes (using the `graphicx`
   `draft` option) or fail depending on configuration.
3. **Word/page count** (Step H2): main-text sections total approximately
   **3,446 words** (excluding tables, figure captions, abstract, and
   references), and the abstract is **197 words** (within the MDPI
   *Cancers* 200-word limit). This is on the lower end of the recommended
   4,000–8,000-word range for an MDPI Original Research article, but all
   planned content (Introduction, Methods, Results, Discussion,
   Conclusions) is substantively complete; the gap is mostly accounted for
   by table and figure content (5 main tables + 5 main figures + 2
   supplementary tables + 5 supplementary figures), which add significant
   page length not captured by this word count.

### Recommended next actions (for the user, outside this session)

- Download `mdpi.cls`/`mdpi.bst` and run `latexmk -pdf main` locally to
  obtain the actual compile log.
- Generate the 10 main figures + 5 supplementary figures from existing
  analysis outputs (`lung_helpers.py` summary dataframes,
  `survival_analysis_nlp.py` outputs) following `drafts/figure_specs.md`,
  and save as PDF/PNG at the paths referenced in `sections/*.tex` and
  `supplementary/supplementary.tex`.
- Resolve the remaining `[TO VERIFY]` placeholders in
  `tables/table1_patients.tex` and `supplementary/supplementary.tex`
  (Table S2) against the actual validation-cohort data.
- Fill in author names/affiliations, funding, IRB protocol number, and
  GitHub repo URL placeholders in `main.tex` (currently bracketed
  `[PLACEHOLDER]` text by design).
