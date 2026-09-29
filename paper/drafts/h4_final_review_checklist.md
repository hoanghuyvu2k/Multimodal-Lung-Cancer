## Step H4 — Final PDF Review Checklist

This checklist is for the user to work through visually once `main.pdf` and
`supplementary.pdf` have been compiled locally (requires `mdpi.cls`,
`mdpi.bst`, and the figure files listed in `drafts/compile_check_g7.md`).
Items already verified statically in this session (Steps G7/H1/E1) are
marked `[x]`; items requiring a compiled PDF or external data are `[ ]`.

### Typography & Layout

- [ ] No lines overflow the margin (no "Overfull \hbox" warnings in the log)
- [ ] Spacing is even between paragraphs; no orphaned single lines at page
      breaks
- [ ] Figures and tables do not split across pages between caption and
      content
- [x] All subfigures (A)/(B)/(C)/(D) are labelled and referenced
      consistently (Figures 1–5 use `(A)`–`(D)` in captions; Figure 1 has
      no subfigure letters as it is a single diagram)
- [ ] `pdflscape` landscape pages (if any wide table is added later) render
      correctly

### Content

- [x] Abstract is 200 words (≤ 200-word MDPI *Cancers* limit — at the cap
      after the 2026-08-19 OvO honesty rewrite, trim further before
      submission if any other edit adds words), contains no citations
- [x] Keywords: 8 terms (within the typical 5–10 range)
- [x] All 5 main figures (`fig:overview`, `fig:nlp_pipeline`,
      `fig:auc_comparison`, `fig:km_curves`, `fig:survival_summary`) are
      mentioned via `\ref{}` in the body text before/at their first
      appearance
- [ ] STALE (pre-2026-08-19 OvO rewrite): originally checked 5 main tables
      (`tab:patients`, `tab:modalities`, `tab:auc_ihca`, `tab:auc_ihcg`,
      `tab:ovo`, `tab:survival`, `tab:km` — 7 labelled tables total). A 6th
      main table, `tab:ovo_confirm` (repeated-seed BM1–4 comparison incl.
      OvO+NLP, in `results.tex`), was added — re-verify all `\ref{}`
      mentions and the "5 main tables" count in `cover_letter.tex` (now 6)
      before submission.
- [ ] No orphaned section headings at the bottom of a page (visual check
      after compile)
- [x] No leftover `TODO`/`PLACEHOLDER` text in `references.bib` or section
      bodies (author/affiliation/funding bracketed placeholders in
      `main.tex` are intentional and listed separately below)

### References

- [x] All `\cite{}` keys (15 distinct, 18 total bib entries) resolve — no
      `[?]` undefined citations expected
- [ ] Reference list renders with continuous numbering in MDPI (Vancouver
      numeric) style after `bibtex`/`biblatex` run
- [ ] DOI links in the compiled reference list are clickable
      (`hyperref` + `doi` field — verify in PDF)
- [ ] `Trebeschi2019`, `Ma2021` (SMIL/AAAI 2021), and `Siegel2024` DOIs
      double-checked against publisher records before submission

### Numbers

- [x] Every AUC/CI/HR/$\chi^2$/$p$-value mentioned in Abstract, Discussion,
      and Conclusion matches Tables 2–5 (verified in
      `drafts/consistency_check_e1.md`)
- [x] $p$-value format consistent: `$p < 0.005$` / `$p < 0.001$` /
      `$p = 0.xxx$` used consistently (no mixing of `p<.05` vs
      `p < 0.05`)
- [x] CI format consistent: `[lower--upper]` (en-dash) used throughout
      (no `(lower, upper)` mixing)

### MDPI-Specific

- [x] `\authcontributions` (CRediT) section present (placeholders to be
      filled with real initials)
- [x] `\funding` section present (placeholder)
- [x] `\conflictsofinterest` section present ("no conflicts of interest")
- [x] `\dataavailability` section present (GitHub placeholder)
- [x] `\institutionalreview` and `\informedconsent` sections present
      (IRB protocol number placeholder)
- [ ] `\TitleCitation` (running head) fits within MDPI's character limit for
      the journal page header (visual check after compile)

### Remaining placeholders requiring user input before submission

1. `main.tex`: real author names, affiliations, corresponding-author email,
   funding statement, IRB protocol number, GitHub repository URL, CRediT
   initials.
2. `tables/table1_patients.tex`: `[TO VERIFY]` validation-cohort columns
   (age, sex, histology, ECOG, therapy type, modality availability, label
   distribution, PFS).
3. `supplementary/supplementary.tex` Table~S2: `[TO VERIFY]` post-filter
   feature counts for CT radiomics and IHC-G GLCM features.
4. `references.bib`: `DyAM2024` entry — confirm publication status (see
   `drafts/references_todo.md`).
5. 10 main figures (`figures/fig1_*.pdf`–`fig5d_*.pdf`) and 5 supplementary
   figures (`supplementary/figures/figS1_*.pdf`–`figS5_*.pdf`) need to be
   generated per `drafts/figure_specs.md`.
