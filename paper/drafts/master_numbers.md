# Master Numbers — Nguồn duy nhất cho mọi số trong paper

> Mọi số trong paper PHẢI lấy từ file này. Cập nhật file này khi có số mới.  
> Nguồn: nlp-ket-qua.md, ovo-complete.md, survival-analysis-plan.md (kết quả 2026-06-09)

---

## Cohort

| Cohort | n | Event rate | Median PFS | PFS range |
|---|---|---|---|---|
| Discovery | 247 | 84.6% (209/247) | 2.7 months | 0.1–49.1 m |
| Radiology validation | 50 | — | — | — |
| Pathology validation | 71 | — | — | — |
| Label ratio (PR/CR : SD/PD) | 1 : 2.98 | — | — | — |

---

## Table A1 — AUC Comparison: IHC-A Arm
*(10-fold KFold, n=247, Discovery cohort)*

| Model | AUC | CI lower | CI upper | ΔAUC vs NoClin |
|---|---|---|---|---|
| DyAM Rad+IHC-A+Gen+PDL1 (No Clin) | 0.764 | 0.695 | 0.833 | ref |
| + Labs (13d) | 0.768 | 0.700 | 0.837 | +0.004 |
| + NLP raw (384d) | 0.781 | 0.717 | 0.845 | +0.017 |
| **+ NLP-PCA16 (16d)** ★ | **0.784** | **0.719** | **0.848** | **+0.020** |
| + Labs + NLP (combined) | 0.767 | 0.700 | 0.835 | +0.003 |
| + Labs + NLP-PCA16 (x2) | 0.753 | 0.682 | 0.823 | −0.011 |
| + BioClinBERT-PCA16 | 0.767 | 0.701 | 0.833 | +0.003 |

---

## Table A2 — AUC Comparison: IHC-G Arm

| Model | AUC | CI lower | CI upper | ΔAUC vs NoClin |
|---|---|---|---|---|
| DyAM Rad+IHC-G+Gen+PDL1 (No Clin) | 0.784 | 0.717 | 0.850 | ref |
| + Labs (13d) | 0.788 | 0.723 | 0.853 | +0.004 |
| **+ NLP raw (384d)** ★ | **0.813** | **0.753** | **0.874** | **+0.030** |
| + NLP-PCA16 (16d) | 0.812 | 0.751 | 0.873 | +0.028 |

---

## Table A3 — OvO vs Original (top cases)
*(20 test cases, SINGLE 10-fold CV partition — exploratory screen only)*

> ⚠️ **Superseded by Table A3b below.** This table uses one fixed CV
> partition (`KFold(random_state=0)`) — a repeated-seed confirmatory check
> (5 seeds × 10-fold, `result/training-results.md` §2.2/§2.5/§2.9) found the
> "best overall" row here does NOT reproduce: OvO and Original are
> statistically indistinguishable at every primary benchmark
> (`|ΔAUC| ≤ 0.011, p ≥ 0.17`). Keep this table for the paper's own
> "exploratory screen" narrative (Section~ovo in `results.tex`), but do
> **not** cite "AUC 0.8003" or "OvO improves low-modality performance" as a
> standalone claim anywhere else — always pair it with Table A3b's numbers.

**Summary:**
- OvO better: 7/20 (35%) | Original better: 9/20 (45%) | Equal: 4/20 (20%)
- Mean ΔAUC: −0.32% | Best OvO: +3.27% | Worst OvO: −4.94%

| Test Case | Original AUC | OvO AUC | ΔAUC |
|---|---|---|---|
| PDL1+Gen | 0.6931 | **0.7157** | +3.27% |
| Rad+Gen | 0.7384 | **0.7548** | +2.23% |
| Rad+IHC-G+Gen+PDL1 | 0.7839 | **0.8003** | +2.10% |
| Rad+IHC-A+Gen+PDL1 | 0.7640 | **0.7765** | +1.63% |
| TMB+PDL1 | 0.7051 | **0.7151** | +1.42% |
| Rad+IHC-A+Gen | **0.7567** | 0.7193 | −4.94% |
| IHC-G+Gen | **0.7558** | 0.7231 | −4.33% |
| Rad+IHC-A+Gen+PDL1+Labs | **0.7683** | 0.7466 | −2.83% |

**Best overall (single partition — not robust, see Table A3b):** OvO Rad+IHC-G+Gen+PDL1 = AUC **0.8003** [0.739–0.862]

---

## Table A3b — OvO vs Original vs Uniform-avg vs OvO+NLP (confirmatory, 5-seed × 10-fold)
*(21 paper-matched combos, `experiments/allcombo/run_paper_ovo.py` + `experiments/nlp_verify/run_ovo_nlp.py`,
patient-label re-shuffle per seed, paired bootstrap significance testing.
Full 21-row table: `result/training-results.md` §2.9. This is the number set
actually used in `results.tex`/`discussion.tex`/`conclusion.tex`.)*

**Primary benchmarks (5-seed mean AUC ± sd):**

| BM | Uniform-avg | Original (DyAM) | OvO | OvO+NLP | OvO vs Original |
|---|---|---|---|---|---|
| BM1 (Rad+IHC-G+Gen+PDL1) | 0.7746±0.019 | 0.7595±0.022 | 0.7728±0.020 | **0.7822±0.017** | Δ=+0.0133, p=ns |
| BM2 (BM1+Labs) | 0.7665±0.011 | 0.7638±0.018 | 0.7632±0.010 | 0.7822±0.017 (=BM1 combo) | Δ=−0.0006, p=ns |
| BM3 (Rad+Gen) | 0.7110±0.015 | 0.7095±0.018 | 0.7132±0.015 | 0.7077±0.014 | Δ=+0.0037, p=ns |
| BM4 (PDL1+Gen) | 0.7191±0.008 | 0.7072±0.009 | 0.7183±0.012 | **0.7472±0.007** | Δ=+0.0111, p=ns |

`|ΔAUC(OvO−Original)| ≤ 0.011, p ≥ 0.17` at all four → **statistically indistinguishable**.

**Best overall (robust, 5-seed):** OvO+NLP BM1/BM4 tie with uniform_avg+NLP as the highest robust
numbers in the whole project (~0.78–0.78, see `result/training-results.md` §3) — no model/attention
combination clears **AUC 0.80** on the 5-seed metric. The single-partition 0.8003/0.8133 numbers are
NOT robust and should not be reported as "best overall" without this caveat.

**Where Original DID significantly beat OvO/uniform (2 configs, few non-redundant modalities):**

| Config | Original | OvO/uniform | p |
|---|---|---|---|
| Rad (3-lesion radiomics only) | 0.6725 | 0.6591 (OvO) / 0.6561 (uniform) | 0.017 |
| IHC-G+Gen | 0.7470 | 0.7193 (OvO) / 0.7184 (uniform) | 0.010 |

**OvO+NLP robustness check:** improves over OvO-no-clinical in 15/21 configs (identical set to
uniform_avg+NLP), declines in the same 6 single/near-single-modality configs — confirms the
NLP-clinical gain is attention-mechanism-agnostic, not OvO-specific or DyAM-specific.

---

## Table A4 — Survival Analysis Summary
*(n=247, 84.6% events, median PFS=2.7m)*

| Model | AUC (binary) | C-index | CI lower | CI upper | tdAUC 6m | tdAUC 12m | tdAUC 18m | Cox HR | Cox CI lower | Cox CI upper | Cox p | IBS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| No Clinical (IHC-A) | 0.776 | 0.623 | 0.580 | 0.665 | 0.717 | 0.718 | 0.700 | 5.30 | 3.10 | 9.07 | <0.001 | 0.1745 |
| + Labs (IHC-A) | 0.754 | 0.626 | 0.583 | 0.669 | 0.713 | 0.707 | 0.662 | 5.31 | 2.88 | 9.79 | <0.001 | 0.1739 |
| + NLP raw (IHC-A) | 0.774 | 0.625 | 0.583 | 0.663 | 0.715 | 0.739 | 0.718 | 5.91 | 3.40 | 10.25 | <0.001 | 0.1723 |
| **+ NLP-PCA16 (IHC-A)** ★ | **0.773** | **0.628** | **0.584** | **0.668** | **0.719** | **0.742** | **0.720** | **6.07** | **3.50** | **10.52** | **<0.001** | **0.1716** |
| No Clinical (IHC-G) | 0.793 | 0.626 | 0.583 | 0.665 | 0.722 | 0.691 | 0.667 | 4.74 | 2.82 | 7.97 | <0.001 | 0.1748 |
| + Labs (IHC-G) | 0.776 | 0.632 | 0.587 | 0.675 | 0.711 | 0.686 | 0.649 | 4.49 | 2.56 | 7.89 | <0.001 | 0.1746 |
| **+ NLP raw (IHC-G)** ★ | **0.797** | **0.632** | **0.591** | **0.672** | **0.720** | **0.699** | **0.665** | **4.97** | **2.98** | **8.31** | **<0.001** | **0.1736** |
| + NLP-PCA16 (IHC-G) | 0.796 | 0.631 | 0.590 | 0.672 | 0.721 | 0.694 | 0.656 | 4.69 | 2.86 | 7.68 | <0.001 | 0.1736 |

---

## Table A5 — Kaplan-Meier Log-Rank

| Model | IHC Arm | χ² | p-value |
|---|---|---|---|
| DyAM No Clinical | IHC-A | 28.17 | <0.005 |
| + NLP-PCA16 | IHC-A | 23.74 | <0.005 |
| DyAM No Clinical | IHC-G | 20.07 | <0.005 |
| **+ NLP-PCA16** | **IHC-G** | **28.92** | **<0.005** ★ |

---

## Table A6 — Wilcoxon Paired C-index Test (per-fold)

| Arm | ΔC (NLP-PCA16 vs NoClin) | p-value | Kết luận |
|---|---|---|---|
| IHC-A | +0.005 | 0.492 | Not significant |
| IHC-G | +0.005 | 0.375 | Not significant |

---

## Single-Modality Baselines

> ⚠️ **KHÔNG dùng bảng LR bên dưới trong paper** (gỡ khỏi §3.1 + Discussion
> ngày 2026-08-19 theo yêu cầu: các số logistic-regression này không do
> nhóm tự chạy nên không trích dẫn). Giữ lại chỉ để tham khảo nội bộ.
> Baseline đơn-modality dùng trong paper là **DyAM 5-seed** ở bảng kế tiếp.

### DyAM đơn-modality — 5-seed × 10-fold (ĐANG DÙNG trong paper §3.1 + Discussion)
Nguồn: `experiments/results/paper_combos.json` (khoá `dyam`), combo #1–#5.

| Nguồn dữ liệu | AUC (mean ± sd) |
|---|---|
| TMB | 0.616 ± 0.001 |
| Pathology IHC-A | 0.635 ± 0.021 |
| CT radiomics (PC/PL/LN) | 0.673 ± 0.020 |
| Genomics (mut+amp) | 0.676 ± 0.013 |
| **PD-L1 TPS** | **0.720 ± 0.012** (đơn-modality mạnh nhất) |
| Fusion 4 nguồn (BM1) | 0.760 ± 0.022 |
| Fusion 4 nguồn + NLP | 0.783 ± 0.016 |

### (cũ, không trích dẫn) LR baselines

| Model | AUC |
|---|---|
| LR Clinical (13 labs) | 0.570 |
| LR Rad-PC | 0.640 |
| LR PDL1-TPS | 0.729 |
| LR Gen-Combined | 0.650 |
| NLP-only | 0.539 |
| Labs-only | 0.594 |

---

*Tạo: 2026-06-09 | Nguồn: nlp-ket-qua.md, ovo-complete.md, survival-analysis-plan.md*
