# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Documentation convention

When creating any file inside `document/`, follow this folder naming rule:

1. **Check today's date** using `currentDate` from context (format `YYYY-MM-DD`).
2. **Check if a dated folder already exists** for today under `document/` (pattern: `YYYY-MM-DD_*`).
3. **If no folder for today exists**, create one named `YYYY-MM-DD_<topic>` where `<topic>` is a short English slug (2–4 words, lowercase, hyphen-separated) derived from the research topic of the session (e.g., `2026-06-08_pathology-pdl1-glcm`).
4. **If a folder for today already exists**, place the file in that existing folder (do not create a second dated folder unless the topic is clearly different).
5. Write the `.md` file inside that dated folder — never write documentation files directly under `document/` root.
6. **NEVER delete or move existing folders** inside `document/` (e.g., `data/`, `base-model/`, etc.). This rule only applies to *new* files being created.

Example structure:
```
document/
  data/                              ← existing folder, do NOT touch
  base-model/                        ← existing folder, do NOT touch
  2026-06-08_pathology-pdl1-glcm/   ← new dated folder for new docs
    lung_pathology_pdl1_glcm_v3.md
  2026-06-09_nlp-clinical-embedding/
    clinical_nlp_embedding.md
```

## Session handoff rule

Whenever the user asks to **summarize the session** (e.g. "summarize session", "summize session",
"tổng hợp/summary session", "tạo handoff"), write a handoff file:

1. Path: `document/handoff/handoff_<YYYY-MM-DD>.md` (today's date from `currentDate`). The `handoff/`
   folder is the ONE exception to the "dated subfolder" rule above — handoff files live directly in it.
2. If a handoff file for today already exists, append a new timestamped section rather than overwriting.
3. Content = context for a NEW session with no prior memory: what the session worked on, key findings
   and decisions, current state of any in-progress plan/loop (which step, what's next), relevant file
   paths, and known gotchas. Write it so a fresh session can resume with zero extra questions.

## Full evaluation rule

Whenever training/evaluating a method or config change with a real conclusion in mind (new attention
variant, new modality encoding, new baseline, etc.) — not a quick debug/smoke run — always evaluate on
**all 21 modality combinations from the original paper** (`Figures-Finalized.ipynb` cell 18), not just
the 4 primary benchmarks (BM1–BM4). The 21 combos are enumerated in `PAPER_COMBOS` in
`experiments/allcombo/run_paper_combos.py` and documented in `result/training-results.md` §2.5/§2.8.

- BM1–BM4 are a convenient 4-combo subset for quick comparison, not a substitute for the full sweep —
  prior runs (`result/training-results.md` §2.5) showed rankings between methods can flip between small
  and large combos (e.g. DyAM only beats `uniform_avg` at k=2 sources, never at k≥3), so a BM-only
  comparison can hide a false win/loss.
- Use the standard 5-seed × 10-fold harness (`result/training-results.md` §0) for the full sweep, the
  same way `run_paper_combos.py` does.
- Log the full-sweep results per the Result logging rule below (one entry per experiment, not one row
  per combo — a summary table across the 21 combos is enough, matching the style of §2.5/§2.8).

## Result logging rule

After **any** training/experiment run finishes (whether the result is positive, negative, or a wash),
append an entry to `result/training-results.md` — never overwrite prior entries. Each entry must include:

1. Experiment name/question + date.
2. Script/notebook run + path to the raw output (`experiments/results/*.json`, `excel/*.xlsx`, etc.).
3. **Full training config**: epochs, lr, alpha, beta, folds, seeds, `cross_modality_enabled`,
   `attention_gate_enabled`, model type (`train`/`train_ovo`/`train_gmu`/`train_uniform_avg`/`train_LR`/...),
   modality combo, cohort/dataset used, l1_filter if any — call out any deviation from the standard config
   documented in `result/training-results.md` §0.
4. Result numbers: AUC (+CI if available), or C-index/HR/tdAUC for survival runs, mean±sd if multi-seed.
5. A one-line verdict (KEEP/DELETE, positive/negative/on par with baseline).

This applies even to negative results — the project's established discipline is to record failed
experiments too, so they aren't re-run blindly in a later session.

## Figure layout check rule

The paper's schematic figures are hand-positioned matplotlib scripts
(`paper/generate_fig*_demo.py`) with no auto-layout, so labels silently overflow their boxes or
collide when text/font/geometry changes. **Eyeballing the rendered PNG is not sufficient** — it has
missed real overflows before.

After editing ANY `paper/generate_fig*.py`, run:

```bash
python paper/check_figures.py     # exit 0 = clean, 1 = layout errors
```

It re-executes each figure script and measures real text bounding boxes (`get_window_extent`),
reporting three error classes: `TEXT-OVERLAP` (labels colliding), `BOX-OVERFLOW` (label wider than
its containing box), `FIG-OVERFLOW` (only checked when the script does NOT save with
`bbox_inches='tight'`, since tight-bbox expands the canvas instead of clipping).

Then re-render the figure(s), view the PNG to confirm it still reads well, and rebuild the paper.
When adding a new figure script, add it to `FIG_SCRIPTS` in `check_figures.py`.

## Table 1 (patient characteristics) rule

`paper/tables/table1_patients.tex` is **generated**, not hand-written:

```bash
python paper/make_table1.py            # writes the .tex
python paper/make_table1.py --dry-run  # print only
```

Never edit that .tex by hand — re-run the script instead (and port the numbers into
`paper/word_export/build_manuscript_docx.py`). Two traps the script exists to prevent:

- **Cohort join**: only `discovery` joins the omnibus inventory by `dmp_pt_id`. `rad_valid` joins by
  `radiology_accession_number`/`did_acc`, `path_valid` by `pdl1_image_id`/`slide_id`
  (see `document/data/omnibus-inventory-analysis.md` §1.1). The omnibus file has **366 rows** but the
  discovery cohort is **247** — a hand-written Table 1 previously reported the 366-row demographics
  under an "n = 247" heading and went unnoticed for months.
- **`pfs_censor` = 1 means EVENT observed**, matching how the code passes it to lifelines
  `event_observed=`. Do not flip it.

Rounding uses half-up (`Decimal`), not `f'{x:.1f}'` — the latter turns a true median of 2.55 into 2.5.

## Word manuscript export rule

The Word export pipeline is a SINGLE self-contained script:
`paper/word_export/build_manuscript_docx.py` (OMML equation builder embedded inside, no other
local imports). Do NOT rebuild it from scratch.

**Trigger — runs in two situations:**
- When the user asks to convert/export the paper to Word (docx).
- **Automatically every time the user asks to compile the paper PDF** (`latexmk`/`pdflatex` on
  `paper/main.tex`): after the PDF build succeeds, also run the Word export in the same task so
  `paper/manuscript_word.docx` never drifts out of date. The user should not have to ask.

**How:**
1. **Run:** `python paper/word_export/build_manuscript_docx.py` → outputs `paper/manuscript_word.docx`
   (auto-falls back to `_v2`, `_v3`... if the target is locked open in Word).
2. **Content is hand-ported** from `paper/sections/*.tex` + `paper/tables/*.tex` into the script —
   if the LaTeX sections changed since the last export, FIRST update the corresponding
   text/tables inside `build_manuscript_docx.py` to match (numbers must stay identical to the
   LaTeX version), THEN run it.
3. **Figures:** the script embeds PNG only. Figures that exist only as PDF must first be converted:
   `pdftocairo -png -r 200 -singlefile paper/figures/<fig>.pdf paper/figures/converted/<fig>`
   (pdftocairo ships with MiKTeX). Already-converted PNGs live in `paper/figures/converted/`.
4. **Math:** all equations use native Word OMML (namespace `m` inside the script) —
   display equations via `add_equation(...)`, inline math inside sentences via
   `P_mix("text ", IM(...), " text")`; figure captions accept a list of parts too.
   Never render math as plain Unicode text (sub/superscript chars) — that was explicitly rejected.
5. Format: plain scientific-paper layout (Title/Abstract/Keywords/IMRaD/Declarations/numbered
   references), NOT the MDPI template.

All scripts must be run from the `code/` directory. Data files are expected at `../datasets/` (one level up).

```bash
# Run a specific comparison experiment
python compare_ovo_attention_full.py   # OvO vs original attention (all modality combos, 10-fold CV)
python compare_nlp_clinical_full.py    # NLP embedding vs raw clinical (full run)
python compare_4way_full.py            # 4-way: Original | OvO | NLP | NLP+OvO
python compare_ovo_subsample.py        # OvO vs original with ShuffleSplit
python debug_dyam_step_by_step.py      # Step-by-step debugging of a single forward pass

# NLP requires extra dependency (not in requirements.txt)
pip install sentence-transformers
```

There are no test suites, linting configs, or build steps. This is a research codebase.

All scripts use `importlib.reload(lung_helpers)` at the top — this is intentional for hot-reload during development, not a bug.

## Architecture

### Core module: `lung_helpers.py`

The single large library (~5300 lines) containing everything. Importing it also sets global matplotlib/seaborn style and suppresses FutureWarnings. Key sections:

**Model classes (PyTorch `nn.Module`):**
- `AttentionMatrix` – cooperative attention: N×N linear layers, softplus activation, L1-normalized weights
- `AttentionMatrixOvO` – competitive attention: N linear layers, `sigmoid(score_i - mean_others)`, L1-normalized
- `AttentionMatrixOvO_Softmax` – same but uses softmax instead of sigmoid
- `MultiModalDynamicModel` – wraps `AttentionMatrix`; handles RobustScaler per modality, BCEWithLogitsLoss with auto class-weighting, Adam optimizer, z-score output normalization
- `MultiModalDynamicModelOvO` – wraps `AttentionMatrixOvO`
- `MultiModalDynamicModelGMU` – wraps `GatedMultimodalUnitFusion` (alternative architecture)
- `MaskedMultiModalLoader` – PyTorch `Dataset` for multimodal batching with mask
- `MILR`, `MultiLesionModel` – older/alternative model variants

**Training functions** (all return `(summary_df, coef_df)` except `train_subsample`):
- `train(...)` – KFold CV, returns per-patient `summary_df` with risk/attention/share scores
- `train_ovo(...)` – same but uses `MultiModalDynamicModelOvO`
- `train_gmu(...)` – same but uses `MultiModalDynamicModelGMU`
- `train_subsample(...)` – ShuffleSplit (always 20 splits, 10% test), returns list of `(auc, ci, scores, labels)`
- `train_eval_all(...)` – single train/val split, no CV
- `train_LR(...)`, `train_XGBoost(...)`, `train_RandomForest(...)` – baseline comparators

**Data preparation functions** (build `modality_dict` + `modality_MASK` in-place):
- `get_clinical_table_v2(path, main_index_col, cohort)` – loads and filters clinical CSV
- `decorate_with_site_index(df_radiology)` – splits radiomics by lesion type (PC/PL/LN)
- `prepare_rad_modality_by_size(df_dict, df, modality_mask, sites, name, ...)` – selects top-N lesions by size
- `prepare_rad_modality(df_dict, df, modality_mask, sites, l_idx, name)` – selects specific lesion index
- `prepare_other_modalities(df_dict, df, modality_mask, name)` – for genomics/pathology/clinical
- `get_training_data(modal_list, modal_dict, df_mask, df_outcomes)` – aligns all modalities to common patient index, returns `(list_of_dfs, mask_df, outcomes_df)`

**Evaluation:**
- `auc_roc_ci(y_true, y_pred, alpha)` – AUC + DeLong confidence intervals

### NLP Clinical module: `clinical_nlp_embedding.py`

Standalone module. Converts 13 numeric clinical columns into 384-dim sentence embeddings.
- `df_to_text_prompts(df)` – builds English sentences from clinical row values
- `ClinicalTextEmbedder` – wraps `sentence-transformers/all-MiniLM-L6-v2`
- `prepare_nlp_clinical_modality(df_dict, df_labs, modality_mask, name)` – end-to-end integration

**Critical:** Always pass `no_scale=[idx]` for NLP modalities in `model_params`. The embeddings are cosine-normalized; RobustScaler will corrupt them.

## Typical experiment pattern

```python
import lung_helpers, importlib
importlib.reload(lung_helpers)
from lung_helpers import *

BASE_DB_DIR = '../datasets'

# 1. Load cohort and outcomes
df_cohort = pd.read_csv(f"{BASE_DB_DIR}/final_cohort_listing.csv").set_index('main_index')
df_cohort_disc = df_cohort[df_cohort['cohort'] == 'discovery']
df_clinical = get_clinical_table_v2(path=f"{BASE_DB_DIR}/clinical.csv",
                                    main_index_col='dmp_pt_id', cohort=df_cohort_disc)
df_outcomes = df_clinical[['label']].copy()

# 2. Load raw data
df_radiology = pd.read_parquet(f"{BASE_DB_DIR}/lung_radiomics_...parquet")
df_genomic   = pd.read_parquet(f"{BASE_DB_DIR}/genomic_data_v3.parquet")

# 3. Build modality_dict and modality_MASK
modality_dict = {}
modality_MASK = pd.DataFrame(index=df_outcomes.index)

df_rad_by_site = decorate_with_site_index(df_radiology)
prepare_rad_modality_by_size(modality_dict, df_rad_by_site, modality_MASK,
                              sites=['PC'], name='rad_lesion_pc')
prepare_other_modalities(modality_dict, df_genomic, modality_MASK, name='gen_driver_mut_amp')

# 4. Align and train
modal_list = ['rad_lesion_pc', 'gen_driver_mut_amp']
data, mask, labels = get_training_data(modal_list, modality_dict, modality_MASK, df_outcomes)

model_params = {'epochs': 100, 'lr': 0.01, 'alpha': 1.0, 'beta': 1.0,
                'cross_modality_enabled': False, 'attention_gate_enabled': True}
l1_filter = {0: {'l1_selection_df': df_radiology, 'kwargs': {'robustness_cutoff': 0.15, 'outlier_cutoff': 6}}}

summary_df, coef_df = train(data, mask, labels, l1_filter, model_params, folds=10)
auc, ci = auc_roc_ci(summary_df['label'].values, summary_df['score'].values, 0.95)
```

## Key data conventions

- Patient index column: `main_index` in all datasets
- Radiomics `job_tag`: `'filtered-radiomics'` (original) vs `'pertubation-radiomics'` (perturbations)
- Labels: `label=0` → PR/CR (response), `label=1` → SD/POD (no response)
- Parquet files may have case-variant filenames; scripts use try/except to handle both
- Output Excel files are written to `excel/` subdirectory (create if missing)

## Documentation

Full project documentation is in `README.md` (combined) with detailed references in `document/`:
- `document/kien-truc-mo-hinh.md` – model architecture detail
- `document/huong-dan-train.md` – `train()` walkthrough with concrete numbers
- `document/ovo-attention.md` – OvO variant, experimental results across 20 test cases
- `document/nlp-clinical.md` – NLP clinical embedding implementation guide
- `document/du-lieu-radiomics.md` – radiomics feature naming conventions
