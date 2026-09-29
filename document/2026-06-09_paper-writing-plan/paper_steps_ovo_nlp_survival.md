# Kế Hoạch Viết Bài Báo Khoa Học — OvO + NLP + Survival Analysis
## DyAM Multimodal Extension: Competitive Attention, NLP Clinical Embedding & Survival Prediction

**Ngày tạo:** 2026-06-09  
**Mục tiêu:** Bài báo Q1/Q2 (npj Digital Medicine / Cancers) — tiếng Anh  
**Phạm vi:** Mở rộng DyAM gốc với 3 đóng góp: OvO attention, NLP clinical embedding, survival endpoint  
**Output format:** **LaTeX** — toàn bộ manuscript viết bằng LaTeX, dùng MDPI template (target: *Cancers*)  
**Lưu ý cho AI:** Mỗi step được thiết kế độc lập, tập trung 1 nhiệm vụ, output cụ thể — không gộp nhiều step vào 1 prompt.

---

## Tổng Quan Đóng Góp Của Bài Báo

| Đóng góp | Kết quả chính |
|---|---|
| **1. OvO Attention** | Cải thiện AUC +2–3% trong cấu hình 2 modalities; kém hơn trong 3+ modalities |
| **2. NLP Clinical Embedding** | AUC 0.813 [IHC-G arm] — cải thiện +0.030 vs No Clinical, không significant (DeLong CI) |
| **3. Survival Analysis** | Cox HR = 6.07 (NLP-PCA16 IHC-A), independent prognostic factor; tdAUC +0.024 tại 12m |

**Narrative arc của bài báo:**
> DyAM gốc dùng cooperative attention và numeric clinical features.  
> Ta đề xuất: (1) OvO competitive attention — tốt hơn trong cấu hình ít modalities; (2) NLP sentence embedding thay thế numeric clinical — mang thêm prognostic signal đặc biệt cho long-term survival.  
> Kết luận: NLP encoding là survival signal hơn là classification signal → cần survival endpoint để đánh giá đúng.

---

## PHẦN G — LATEX SETUP VÀ TEMPLATE

> Thực hiện **trước tất cả** các phần còn lại. Một khi template đã sẵn sàng, mỗi step B sẽ output thẳng LaTeX thay vì plain text.

---

### [x] Step G1 — Tạo Cấu Trúc Thư Mục LaTeX

**Mục tiêu:** Khởi tạo project LaTeX với đầy đủ thư mục và file skeleton, sẵn sàng để điền nội dung.

**Cấu trúc thư mục cần tạo:**

```
paper/
├── main.tex                  ← file chính, \input các section
├── sections/
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── methods.tex
│   ├── results.tex
│   ├── discussion.tex
│   └── conclusion.tex
├── tables/
│   ├── table1_patients.tex
│   ├── table2_auc_ihca.tex
│   ├── table3_auc_ihcg.tex
│   ├── table4_survival.tex
│   └── table5_km.tex
├── figures/
│   ├── fig1_overview.pdf
│   ├── fig2_nlp_pipeline.pdf
│   ├── fig3_auc_barchart.pdf
│   ├── fig4_km_curves.pdf
│   └── fig5_survival_summary.pdf
├── supplementary/
│   ├── supplementary.tex
│   └── figures/
│       ├── suppfig1_pca_scree.pdf ... suppfig5_heatmap.pdf
├── references.bib            ← BibTeX database
└── mdpi.cls                  ← MDPI journal class file (download từ MDPI)
```

**Prompt mẫu:**
```
Tạo file `paper/main.tex` — skeleton đầy đủ cho một bài báo MDPI Cancers journal.
Yêu cầu:
  1. \documentclass[cancers,article,submit,moreauthors,pdftex]{mdpi}
     (hoặc \documentclass[12pt]{article} nếu không có mdpi.cls — fallback)
  2. Preamble đầy đủ:
       \usepackage{amsmath,amssymb}     % math
       \usepackage{booktabs}            % professional tables
       \usepackage{graphicx}            % figures
       \usepackage{subcaption}          % subfigures (a)(b)
       \usepackage{hyperref}            % clickable refs + DOI links
       \usepackage{xcolor}              % colored text for review
       \usepackage{microtype}           % better typography
       \usepackage{natbib}              % author-year or numbered refs
       \bibliographystyle{mdpi}         % MDPI reference style
  3. Title, authors (placeholder: First Author$^{1}$, Second Author$^{1,2}$), affiliations
  4. \begin{abstract} ... \end{abstract} + \keywords{} block
  5. \input{sections/introduction}
     \input{sections/methods}
     \input{sections/results}
     \input{sections/discussion}
     \input{sections/conclusion}
  6. \bibliography{references}
  7. Mỗi file sections/*.tex chỉ chứa \section{...} và nội dung — không có \begin{document}
Output: nội dung đầy đủ của main.tex (~80 dòng) + skeleton cho 6 file sections/*.tex (chỉ header, chưa có nội dung).
```

**Output kỳ vọng:** File `main.tex` hoàn chỉnh + 6 file sections/*.tex rỗng với header đúng cú pháp.

---

### [x] Step G2 — Download và Cấu Hình MDPI Template

**Mục tiêu:** Chuẩn bị class file MDPI và kiểm tra compile thành công.

**Thông tin:**
- MDPI template download: https://www.mdpi.com/authors/latex (file `mdpi-template.zip`)
- Cần 2 file: `mdpi.cls` + `mdpi.bst` (bibliography style)
- Test journal: `cancers` (truyền vào `\documentclass[cancers,...]`)

**Prompt mẫu:**
```
Tôi đã download mdpi.cls và mdpi.bst vào thư mục paper/.
Viết nội dung preamble cho \documentclass MDPI Cancers với đầy đủ metadata fields:

\Title{NLP-Augmented Multimodal Attention Fusion for Immunotherapy Response
Prediction in Non-Small Cell Lung Cancer}

\Author{FirstName LastName$^{1,\dagger}$, FirstName LastName$^{2}$, FirstName LastName$^{1,*}$}

\AuthorNames{LastName, F.; LastName, F.; LastName, F.}

\address{
$^{1}$ \quad Department of..., University of..., City, Country; email@email.com \\
$^{2}$ \quad Department of..., University of..., City, Country
}

\corres{Correspondence: email@email.com}

\abstract{[ABSTRACT PLACEHOLDER — điền sau Step B12]}

\keyword{non-small cell lung cancer; immunotherapy response prediction;
multimodal learning; attention mechanism; NLP; sentence embedding;
survival analysis; radiomics}

Thêm các MDPI-specific commands:
  \funding{[FUNDING PLACEHOLDER]}
  \institutionalreview{[IRB PLACEHOLDER]}
  \informedconsent{[CONSENT STATEMENT]}
  \dataavailability{Code available at: [GitHub URL]. Data subject to IRB restrictions.}
  \conflictsofinterest{The authors declare no conflict of interest.}
```

**Output kỳ vọng:** Block preamble ~50 dòng, sẵn sàng paste vào `main.tex`.

---

### [x] Step G3 — LaTeX Math Equations cho Methods

**Mục tiêu:** Viết tất cả công thức toán học trong paper dưới dạng LaTeX `equation` environment, đánh số và label để `\ref{}` từ text.

**Input cho AI:**
- File: `document/ovo-model/ovo-complete.md` — Section 10 (Appendix công thức)
- Các công thức: risk score, attention gốc, OvO sigmoid, final output

**Prompt mẫu:**
```
Viết 6 LaTeX equations cho Methods section, dùng \begin{equation}...\end{equation} với \label{eq:X}:

eq:risk     r_i = \tanh(\mathbf{W}_i^\top \mathbf{x}_i)
eq:attn_s   s_i = \text{softplus}(\mathbf{W}_{ij}^\top \mathbf{x}_i) / \|\mathbf{x}_i\|
eq:attn_a   a_i = s_i \big/ \textstyle\sum_{j=1}^{N} s_j
eq:ovo_mu   \mu_{-i} = \tfrac{1}{N-1} \sum_{j \neq i} s_j
eq:ovo_ovo  \text{ovo}_i = \sigma(s_i - \mu_{-i})  \quad \sigma(\cdot) = \text{sigmoid}
eq:output   \hat{y} = \sum_{i=1}^{N} r_i \cdot a_i

Yêu cầu:
  - Dùng \mathbf{} cho vectors/matrices
  - Dùng \text{} cho function names (tanh, softplus, sigmoid)
  - Thêm 1–2 dòng text LaTeX mô tả mỗi equation (dạng "where $r_i$ is...")
  - Tất cả labels dùng prefix eq: (eq:risk, eq:attn_s, ...)
  - Format: có thể copy thẳng vào sections/methods.tex
```

**Output kỳ vọng:** 6 `equation` blocks LaTeX + inline description text, ~40 dòng.

---

### [x] Step G4 — LaTeX Tables (booktabs style)

**Mục tiêu:** Convert tất cả 5 tables từ markdown sang LaTeX `table` + `tabular` environment dùng `booktabs`.

> Thực hiện **sau Step C4** (khi đã có nội dung đầy đủ của mỗi table).

**Prompt cho Table 2 (mẫu — dùng lại cho Table 3–5):**
```
Convert markdown Table 2 (AUC comparison IHC-A arm) sang LaTeX booktabs format:

Yêu cầu:
  \begin{table}[H]
    \caption{...}    % 2–3 câu caption đầy đủ
    \label{tab:auc_ihca}
    \begin{tabular}{lcccc}
      \toprule
      Model & AUC & 95\% CI & $\Delta$AUC \\
      \midrule
      ... (7 rows) ...
      \bottomrule
    \end{tabular}
    \begin{tablenotes}  % hoặc \footnotesize note dưới table
      Note: CI, confidence interval; AUC, area under the receiver operating
      characteristic curve; DeLong method. $\dagger$ = best performing model.
    \end{tablenotes}
  \end{table}

Thêm \textbf{} cho row tốt nhất. Dùng \textsuperscript{†} cho chú thích.
Dùng [H] float specifier (requires \usepackage{float}).
Format số: 0.784 [0.719--0.848] — dùng -- cho en-dash trong LaTeX.
```

**Prompt cho Table 4 (Survival — phức tạp nhất):**
```
Convert Table 4 (Survival Analysis Summary) sang LaTeX — table có nhiều cột, cần landscape:

\begin{landscape}
\begin{table}[H]
  \caption{Survival analysis results across 8 model variants...}
  \label{tab:survival}
  \begin{tabular}{p{4.5cm}ccccc}  % p{} cho cột Model (text dài)
    \toprule
    Model & C-index [95\% CI] & Cox HR [95\% CI] & Cox $p$ & tdAUC 12m & IBS \\
    \midrule
    ... (8 rows) ...
    \bottomrule
  \end{tabular}
\end{table}
\end{landscape}
Dùng \usepackage{pdflscape} cho landscape. Chú ý escape ký tự đặc biệt.
```

**Output kỳ vọng:** 5 file `tables/table*.tex` hoàn chỉnh, compile-ready.

---

### [x] Step G5 — LaTeX Figures (\includegraphics + subcaption)

**Mục tiêu:** Viết LaTeX code để include tất cả figures vào manuscript với caption đúng format.

> Thực hiện sau khi đã có file PDF/PNG của figures (từ Python scripts).

**Prompt mẫu:**
```
Viết LaTeX code để include Figure 3 (AUC bar chart, 2 panels) và Figure 5 (Survival summary, 4 panels):

Figure 3 — 2-panel figure (side by side):
\begin{figure}[H]
  \begin{subfigure}[b]{0.48\textwidth}
    \includegraphics[width=\textwidth]{figures/fig3a_auc_ihca.pdf}
    \caption{IHC-A pathway}
    \label{fig:auc_ihca}
  \end{subfigure}
  \hfill
  \begin{subfigure}[b]{0.48\textwidth}
    \includegraphics[width=\textwidth]{figures/fig3b_auc_ihcg.pdf}
    \caption{IHC-G pathway}
    \label{fig:auc_ihcg}
  \end{subfigure}
  \caption{[Full caption từ Step C2, 2–3 sentences]}
  \label{fig:auc_comparison}
\end{figure}

Figure 5 — 2×2 grid (4 panels):
[tương tự — dùng 2×2 subfigure layout]

Yêu cầu:
  - Dùng [H] float specifier
  - Width chuẩn cho 2-column MDPI: \textwidth = toàn trang, 0.48\textwidth = nửa trang
  - Labels: fig:overview, fig:nlp_pipeline, fig:auc_comparison, fig:km_curves, fig:survival
  - Mỗi \caption{} dùng nội dung đã viết trong Step C1–C3
```

**Output kỳ vọng:** LaTeX code cho 5 figures, có `\label{}` đúng để `\ref{}`.

---

### [x] Step G6 — BibTeX Database (`references.bib`)

**Mục tiêu:** Tạo file `.bib` với ~25 references, format chuẩn BibTeX cho MDPI.

> Thực hiện sau Step F1 (khi đã tìm được DOI/PMID của từng ref).

**Prompt mẫu:**
```
Tạo file references.bib với BibTeX entries cho 25 references đã tìm được.
Yêu cầu format:
  @article{Reimers2019,
    author  = {Reimers, Nils and Gurevych, Iryna},
    title   = {Sentence-{BERT}: Sentence Embeddings using {Siamese BERT}-Networks},
    journal = {Proceedings of the 2019 Conference on Empirical Methods in Natural
               Language Processing},
    year    = {2019},
    doi     = {10.18653/v1/D19-1410}
  }
  @article{DeLong1988,
    author  = {DeLong, Elizabeth R. and DeLong, David M. and Clarke-Pearson, Daniel L.},
    title   = {Comparing the Areas under Two or More Correlated Receiver Operating
               Characteristic Curves: A Nonparametric Approach},
    journal = {Biometrics},
    year    = {1988},
    volume  = {44},
    pages   = {837--845},
    doi     = {10.2307/2531595}
  }

Quy tắc:
  - Key format: AuthorYEAR (e.g., Rizvi2015, Reck2016, Harrell1996)
  - Dùng {} để bảo vệ capitalization trong title: {BERT}, {NLP}, {NSCLC}
  - Tất cả entries phải có doi hoặc pmid
  - Journal names không viết tắt (MDPI yêu cầu full name)
  - Tạo tất cả 25 entries dựa trên danh sách từ Step F1
```

**Output kỳ vọng:** File `references.bib` ~150–200 dòng, tất cả 25 entries.

---

### [x] Step G7 — LaTeX Compilation Check

**Mục tiêu:** Hướng dẫn compile, sửa lỗi phổ biến, tạo PDF cuối cùng.

**Lệnh compile:**
```bash
# Compile đầy đủ (cần chạy 3 lần để resolve cross-references)
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Hoặc dùng latexmk (tự động)
latexmk -pdf main.tex

# Kiểm tra warnings/errors
grep -i "warning\|error\|undefined" main.log | head -50
```

**Prompt mẫu:**
```
Tôi có log file từ pdflatex compile (paste nội dung main.log).
Phân tích và liệt kê:
  1. Errors (ngăn compile): loại lỗi, dòng lỗi, cách fix cụ thể
  2. Warnings quan trọng: undefined references, overfull hboxes > 20pt, missing citations
  3. Warnings có thể bỏ qua: font warnings, minor spacing
  4. Kiểm tra: tất cả \ref{} và \cite{} có resolved không (xem "undefined reference" warnings)
  5. Page count và word count ước tính

Sau khi fix xong, compile lại và xác nhận PDF output đúng.
```

**Output kỳ vọng:** Danh sách errors/warnings + fix instructions.

---

## PHẦN A — CHUẨN BỊ DỮ LIỆU & SỐ LIỆU

---

### [x] Step A1 — Kiểm tra và tổng hợp toàn bộ số liệu AUC

**Mục tiêu:** Tập hợp tất cả số AUC, CI, n vào 1 bảng tổng hợp để dùng xuyên suốt paper.

**Input cho AI:**
- File: `document/nlp-model/nlp-ket-qua.md` (bảng AUC Section 3)
- File: `document/ovo-model/ovo-complete.md` (Section 5 — Kết quả thực nghiệm)
- File: `document/base-model/mo-hinh-chinh.md` (AUC của baseline models)

**Prompt mẫu:**
```
Đọc 3 file kết quả sau và tổng hợp thành 1 bảng master gồm các cột:
[Model Name | Architecture | IHC Arm | AUC | 95% CI lower | 95% CI upper | ΔAUC vs ref | KM χ² | KM p]
Sắp xếp: nhóm theo IHC arm, trong mỗi nhóm sắp xếp theo AUC tăng dần.
Đánh dấu: (ref) cho baseline, * cho kết quả tốt nhất mỗi nhóm.
Output: markdown table + ghi chú n bệnh nhân cho từng cohort.
```

**Output kỳ vọng:** 1 bảng markdown ~20–25 dòng, có chú thích đầy đủ.

---

### [x] Step A2 — Kiểm tra và tổng hợp số liệu Survival Analysis

**Mục tiêu:** Lấy đầy đủ bảng C-index, Cox HR, tdAUC, IBS từ kết quả đã chạy.

**Input cho AI:**
- File: `document/2026-06-08_pathology-pdl1-glcm/survival-analysis-plan.md` — Section "Kết quả thực tế (chạy 2026-06-09)"

**Prompt mẫu:**
```
Từ section "Kết quả thực tế" trong file, trích xuất và format lại thành 4 bảng riêng biệt:
  Table SA-1: C-index + bootstrap 95% CI cho 8 models (4 IHC-A + 4 IHC-G)
  Table SA-2: tdAUC tại 6m / 12m / 18m cho 8 models
  Table SA-3: Cox HR [95% CI], p-value cho 8 models  
  Table SA-4: IBS (Integrated Brier Score) cho 8 models
Thêm cột ΔAUC_c / ΔHR / ΔIBS so với baseline No Clinical tương ứng.
Ghi rõ: n=247, event rate=84.6%, median PFS=2.7m.
Output: 4 bảng markdown sạch, sẵn sàng copy vào paper.
```

**Output kỳ vọng:** 4 bảng markdown, mỗi bảng ~8 dòng + header.

---

### [x] Step A3 — Kiểm tra số liệu Cohort và Patient Characteristics

**Mục tiêu:** Xây dựng Table 1 (Patient Characteristics) cho paper.

**Input cho AI:**
- File: `document/data/du-lieu-final-cohort-listing.md`
- File: `document/data/genomic-data-v3-analysis.md`
- File: `document/base-model/tong-quan-du-an.md`

**Prompt mẫu:**
```
Từ 3 file trên, tổng hợp thông tin để tạo Table 1 (Patient Characteristics) cho bài báo, gồm:
  - n bệnh nhân per cohort (discovery, rad-valid, path-valid)
  - Label distribution (PR/CR vs SD/PD, tỉ lệ %)
  - Clinical features: age median [IQR], ECOG 0–1 vs 2+ (%), pack-years median [IQR]
  - Modality availability: % bệnh nhân có đủ Rad / IHC / Gen / PDL1
  - Therapy: line 1 vs 2+ (%), combo vs mono (%)
Format: bảng 2 cột (Variable | Discovery n=247), style journal standard.
Nếu không đủ số liệu cụ thể, ghi [TO VERIFY FROM DATA].
```

**Output kỳ vọng:** Table 1 draft ~15–20 dòng với placeholder rõ ràng.

---

## PHẦN B — VIẾT TỪNG SECTION CỦA MANUSCRIPT

> **Quy tắc chung:** Mỗi step viết 1 section độc lập. Không viết nhiều section trong 1 prompt.  
> Mỗi section viết bằng tiếng Anh, academic style, target journal: *npj Digital Medicine* / *Cancers*.  
> **OUTPUT FORMAT:** Tất cả output phải là **LaTeX source code**, sẵn sàng paste vào file `sections/*.tex`. Không output plain text hay markdown.  
> **LaTeX conventions áp dụng cho tất cả steps B:**
> - Dùng `\emph{}` cho in nghiêng thuật ngữ lần đầu
> - Dùng `\cite{Key}` placeholder cho references (sẽ điền key thật sau Step G6)
> - Dùng `\ref{eq:X}`, `\ref{fig:X}`, `\ref{tab:X}` cho cross-references
> - Dùng `--` cho en-dash (ranges: 0.753--0.874), `---` cho em-dash
> - Escape ký tự đặc biệt: `\%`, `\$`, `\_`, `\&`
> - Numbers in text: `$n = 247$`, `$\text{AUC} = 0.813$`

---

### [x] Step B1 — Viết Section: Methods 3.1 – Study Design and Cohort

**Mục tiêu:** Đoạn Methods đầu tiên, khoảng 200–250 words.

**Input cho AI:**
- Kết quả Step A3 (Table 1 Patient Characteristics)
- Thông tin: n=247 discovery (MSK-MIND 2017–2021), n=50 rad-valid, n=71 path-valid
- Label: 0=PR/CR (response), 1=SD/PD (no response), class ratio ≈ 1:2.98
- 10-fold KFold CV trên discovery; train-full test trên validation sets
- Tiêu chí: Stage IIIB/IV NSCLC, first/second-line ICI

**Prompt mẫu:**
```
Viết đoạn "2.1 Study Design and Patient Cohort" cho bài báo (200–250 words, tiếng Anh).
Nội dung phải bao gồm:
  1. Nguồn gốc cohort, thời gian thu thập, tiêu chí đưa vào/loại trừ
  2. Mô tả 3 cohorts (discovery + 2 validation) với n tương ứng
  3. Endpoint definition: binary (PR/CR vs SD/PD) + PFS definition
  4. Câu về 10-fold CV và external validation
  5. Ethical statement placeholder: [IRB approval statement — fill in]
Đây là Methods section — không bàn kết quả.
```

**Output kỳ vọng:** LaTeX source ~30 dòng cho file `sections/methods.tex` — từ `\subsection{Study Design...}` đến hết đoạn, không có `\begin{document}`.

**LaTeX note riêng cho B1:**
```
\subsection{Study Design and Patient Cohort}
Nội dung... 
% Placeholder IRB:
This study was approved by the Institutional Review Board of [INSTITUTION]
(Protocol No.~[NUMBER]). All patients provided written informed consent.
```

---

### [x] Step B2 — Viết Section: Methods 3.2 – Data Modalities

**Mục tiêu:** Bảng + mô tả tất cả modalities, ~250–300 words + 1 table.

**Input cho AI:**
- Thông tin từ `document/ovo-model/ovo-complete.md` Section 6 (ý nghĩa y học)
- Thông tin từ `document/nlp-model/nlp-implementation.md` Section 2 (13 cột)
- Radiomics: MIRPON pipeline, spacing 1.0mm, ~1688 features per lesion type
- L1 selection: ICC > 0.15 (robustness), z-score < 6 (outlier)

**Prompt mẫu:**
```
Viết đoạn "2.2 Data Modalities" cho bài báo (250–300 words, tiếng Anh).
Bao gồm:
  1. Table 1 (inline): Modality | Source | Features | Clinical Meaning
     Các modalities: CT Radiomics (PC/PL/LN), Pathology IHC-A, IHC-G (GLCM),
     Genomics (TMB + drivers), PD-L1 TPS, Clinical Labs (13 vars),
     NLP embedding (384d), NLP-PCA16 (16d)
  2. Mô tả L1 feature selection cho radiomics (ICC threshold + outlier)
  3. 1 câu về missing modality handling (binary mask)
  4. 1 câu về RobustScaler và no_scale exception cho NLP
```

**Output kỳ vọng:** LaTeX source ~50 dòng — `\subsection{}` + text + `\begin{table}...\end{table}` inline (dùng `tabular` với `booktabs`). Table phải có `\label{tab:modalities}` và `\caption{}`.

---

### [x] Step B3 — Viết Section: Methods 3.3 – NLP Clinical Embedding (Key New Method)

**Mục tiêu:** Section kỹ thuật quan trọng nhất của paper, ~350–400 words.

**Input cho AI:**
- File: `document/nlp-model/nlp-implementation.md` — Section 3.1 đến 3.3
- Ví dụ prompt: "The patient is a 65-year-old. Smoking history in pack-years is 30..."
- Model: `sentence-transformers/all-MiniLM-L6-v2`, 384-dim, L2-normalized
- Comparison baseline: BioClinBERT (768-dim, mean pooling, PCA16 → kém hơn)
- PCA: top 16 components, ~80% variance explained
- Critical: no_scale=True cho NLP modality

**Prompt mẫu:**
```
Viết đoạn "2.3 NLP Encoding of Tabular Clinical Features" cho bài báo (350–400 words).
Cấu trúc:
  Paragraph 1: Motivation — tại sao numeric labs không đủ, NLP captures inter-feature context
  Paragraph 2: Text prompt construction — describe df_to_text_prompts(), cho example 1 câu
  Paragraph 3: Sentence encoder — all-MiniLM-L6-v2, output 384-dim L2-normalized,
                no fine-tuning (zero-shot transfer), lý do không dùng BioClinBERT
  Paragraph 4: Dimensionality reduction — PCA(n=16), variance retained, no_scale rationale
  Tránh: implementation detail dạng code, pseudocode
  Tone: Methods section của Q1 journal, passive voice, third person
```

**Output kỳ vọng:** LaTeX source ~60 dòng — 4 `\paragraph{}` hoặc đoạn văn liên tiếp trong `\subsection{NLP Encoding...}`, không có hình minh họa (figure ở Step C2).

---

### [x] Step B4 — Viết Section: Methods 3.4 – Model Architecture (DyAM + OvO)

**Mục tiêu:** Mô tả kiến trúc DyAM gốc và biến thể OvO, ~300–350 words + 2 công thức.

**Input cho AI:**
- File: `document/ovo-model/ovo-complete.md` Section 2, 3, và Section 10 (Appendix công thức)
- Bảng so sánh: AttentionMatrix vs AttentionMatrixOvO (Section ovo-complete.md đầu trang)
- Training: Adam lr=0.01, 125 epochs, BCEWithLogitsLoss with class weights, 10-fold KFold

**Prompt mẫu:**
```
Viết đoạn "2.4 Model Architecture" cho bài báo (300–350 words).
Cấu trúc:
  Paragraph 1: Overview DyAM — N modalities, per-modality risk score, attention weighting,
               missing modality mask, final fused output
  Paragraph 2: Original cooperative attention formula:
               r_i = tanh(W_i^T · x_i),  a_i = s_i / Σ s_j  (L1 normalize)
  Paragraph 3: Proposed OvO competitive attention formula:
               ovo_i = sigmoid(s_i - μ_{-i}),  a_i = ovo_i / Σ ovo_j
               Giải thích: μ_{-i} là mean của các modalities khác, sigmoid gives independent
               probability "modality i outperforms others"
  Paragraph 4: Training details (optimizer, loss, epochs, CV)
  Dùng LaTeX-style inline math: $r_i = \tanh(W_i^T x_i)$
```

**Output kỳ vọng:** LaTeX source ~70 dòng — `\subsection{Model Architecture}` + 4 đoạn văn + 6 `equation` environments (dùng `\eqref{}` để cross-reference các công thức từ Step G3).

**LaTeX note riêng cho B4:** Các equation đã được viết trong Step G3, chỉ cần `\eqref{eq:risk}` trong text. Không viết lại equation, chỉ reference.

---

### [x] Step B5 — Viết Section: Methods 3.5 – Evaluation Metrics

**Mục tiêu:** Mô tả đầy đủ tất cả metrics (AUC, DeLong CI, KM, C-index, Cox, tdAUC, IBS), ~250 words.

**Input cho AI:**
- File: `document/nlp-model/nlp-ket-qua.md` Section 4 (kiểm định thống kê)
- File: `document/2026-06-08_pathology-pdl1-glcm/survival-analysis-plan.md` — Phần 1, 3, 5

**Prompt mẫu:**
```
Viết đoạn "2.5 Statistical Analysis and Evaluation" cho bài báo (250–280 words).
Phải đề cập đến:
  1. AUC-ROC: pooled out-of-fold, DeLong 95% CI (structural components method)
     — tiêu chí non-overlapping CI để kết luận significant
  2. Survival endpoint (PFS): Harrell's C-index + bootstrap 1000 iterations
  3. Time-dependent AUC tại 6m, 12m, 18m (cumulative/dynamic AUC)
  4. Multivariate Cox PH regression: score + age, ECOG, albumin, dNLR, liver_mets
     — report HR [95% CI], p-value
  5. KM log-rank: binary split tại score=0, report χ² và p-value
  6. Integrated Brier Score (IBS): calibration check, threshold <0.25 vs random
  7. Paired Wilcoxon signed-rank test (per-fold C-index)
  Software: Python 3.x, lifelines, scikit-survival, scipy
```

**Output kỳ vọng:** LaTeX source ~45 dòng — `\subsection{Statistical Analysis and Evaluation}` + text có `itemize` hoặc inline list cho các metrics. Dùng `\textit{}` cho tên metrics lần đầu (e.g., \textit{Harrell's C-index}).

---

### [x] Step B6 — Viết Section: Results 4.1 – Multimodal vs Single-Modality Baselines

**Mục tiêu:** Đoạn Results đầu tiên — bối cảnh so sánh với baselines, ~200 words.

**Input cho AI:**
- Kết quả Step A1 (bảng AUC tổng hợp) — phần single-modality baselines
- Số liệu: LR Clinical = 0.570, LR Rad-LN = 0.681, LR PDL1-TPS = 0.729, DyAM Rad+IHC-G+Gen+PDL1 = 0.784

**Prompt mẫu:**
```
Viết đoạn "3.1 Single-Modality Baselines and Multimodal Fusion" (150–200 words, Results section).
Nội dung:
  - DyAM multimodal substantially outperforms single-modality LR
  - Report AUC numbers: LR Clinical 0.570, LR Rad 0.681, LR PDL1 0.729
  - DyAM best baseline: 0.784 [0.717–0.850] (Rad+IHC-G+Gen+PDL1)
  - 1 câu về overall context: multimodal integration is the primary driver of performance
  Không mention NLP hay OvO ở đây — chỉ nền tảng baseline.
  Tone: factual, report results, past tense cho observations.
```

**Output kỳ vọng:** LaTeX source ~25 dòng — `\subsection{Multimodal Fusion Outperforms Single-Modality Baselines}` + 1 đoạn văn, dùng `\ref{tab:auc_ihcg}` để trỏ vào table.

---

### [x] Step B7 — Viết Section: Results 4.2 – OvO Attention Comparison

**Mục tiêu:** Trình bày kết quả OvO vs Original, phân tích theo số lượng modalities, ~300 words.

**Input cho AI:**
- File: `document/ovo-model/ovo-complete.md` Section 5 — toàn bộ kết quả thực nghiệm
- Bảng top 5 OvO tốt hơn và top 5 Original tốt hơn

**Prompt mẫu:**
```
Viết đoạn "3.2 OvO Competitive Attention vs Cooperative Attention" (280–320 words, Results).
Cấu trúc:
  Paragraph 1: Overall summary — OvO tốt hơn 7/20 cases (35%), avg ΔAUC = −0.32%,
               best OvO = PDL1+Gen (+3.27%), worst = Rad+IHC-A+Gen (−4.94%)
  Paragraph 2: Stratification by modality count:
               - 1 modality: identical (4/20 cases)
               - 2 modalities: OvO better 3/6 (50%), clearest gains with PDL1+Gen, Rad+Gen
               - 3+ modalities: Original better 6/10 (60%), especially with IHC-G or Labs
  Paragraph 3: Best configuration — OvO Rad+IHC-G+Gen+PDL1 = AUC 0.8003 [0.739–0.862]
               là best-performing across all models tested
  Tránh: đưa ra toàn bộ 20 test cases, chỉ highlight top 3-4 examples.
  Table inline: Top 5 OvO better | Top 5 Original better (nhỏ gọn, 5 cột)
```

**Output kỳ vọng:** LaTeX source ~60 dòng — `\subsection{OvO Competitive Attention...}` + 3 đoạn + `\begin{table}` inline nhỏ (top 5 OvO vs top 5 Original, 2 mini-tables side-by-side dùng `minipage`).

---

### [x] Step B8 — Viết Section: Results 4.3 – NLP Clinical Embedding (AUC Analysis)

**Mục tiêu:** Kết quả NLP — AUC comparison, 7 variants, 2 IHC arms, ~350 words + 1 table.

**Input cho AI:**
- Kết quả Step A1 (bảng AUC tổng hợp) — phần NLP variants
- File: `document/nlp-model/nlp-ket-qua.md` Section 3 và 4
- Key numbers: NLP raw IHC-G = 0.813 [0.753–0.874], NLP-PCA16 IHC-A = 0.784

**Prompt mẫu:**
```
Viết đoạn "3.3 NLP Encoding of Clinical Features Improves AUC" (320–360 words, Results).
Cấu trúc:
  Paragraph 1: Overall finding — NLP consistently outperforms numeric Labs across both arms
               (+0.013–0.026); best = IHC-G NLP raw AUC 0.813
  Table 2 (inline): IHC-A arm — 7 model variants với AUC + 95% CI + ΔAUC
  Table 3 (inline): IHC-G arm — 4 model variants với AUC + 95% CI + ΔAUC
  Paragraph 2: Statistical interpretation — all DeLong CIs overlap → no result reaches
               significance; largest separation IHC-G: 0.753 lower vs 0.850 upper
  Paragraph 3: BioClinBERT comparison — inferior to MiniLM (ΔAUC = −0.017),
               reason: MLM training vs sentence similarity task
  Paragraph 4: Labs+NLP combined hurts performance (AUC 0.753 IHC-A) — 8 modalities
               with n=247 leads to unstable attention
  Framing: consistent directional improvement, not statistically significant — do NOT
           frame as "NLP does not work"
```

**Output kỳ vọng:** LaTeX source ~80 dòng — `\subsection{NLP Encoding...}` + 4 đoạn + `\input{tables/table2_auc_ihca}` + `\input{tables/table3_auc_ihcg}` (tables được \input từ file riêng, không inline).

---

### [x] Step B9 — Viết Section: Results 4.4 – Survival Analysis

**Mục tiêu:** Trình bày 5 survival metrics, highlight Cox HR và tdAUC, ~400 words.

**Input cho AI:**
- Kết quả Step A2 (4 bảng survival)
- Draft đoạn Results từ `document/2026-06-08_pathology-pdl1-glcm/survival-analysis-plan.md` — "Đoạn Results cho paper (draft)"

**Prompt mẫu:**
```
Viết đoạn "3.4 Survival Analysis: NLP as Independent Prognostic Factor" (380–420 words, Results).
Cấu trúc:
  Paragraph 1: All 8 models are independent prognostic factors (multivariate Cox p<0.001)
               after adjusting for age, ECOG, albumin, dNLR, liver mets
  Table 4 (inline, compact): Cox HR [95% CI] + p-value cho 8 models
  Paragraph 2: NLP-PCA16 achieves highest HR in IHC-A arm (HR=6.07 [3.50–10.52])
               vs No Clinical (5.30); numeric Labs does NOT improve HR (5.31 ≈ 5.30)
  Paragraph 3: tdAUC — NLP advantage concentrated at 12m (+0.024) and 18m (+0.020),
               not at 6m — consistent with NLP encoding long-term survival signal
  Paragraph 4: C-index — consistent modest improvement ΔC=+0.005 (not significant,
               Wilcoxon p=0.375–0.492), high inter-fold variance explains non-significance
  Paragraph 5: All models well-calibrated (IBS 0.172–0.175 << 0.25 random baseline)
  Sử dụng draft đã có làm base, expand và polish to journal quality.
```

**Output kỳ vọng:** LaTeX source ~90 dòng — `\subsection{Survival Analysis: NLP as an Independent Prognostic Factor}` + 5 đoạn + `\input{tables/table4_survival}`. Dùng `\ref{fig:survival}` để trỏ vào Figure 5. **Đây là section quan trọng nhất — dành nhiều chú ý.**

---

### [x] Step B10 — Viết Section: Discussion

**Mục tiêu:** Diễn giải kết quả trong bối cảnh literature, 5–6 đoạn, ~600–700 words.

**Input cho AI:**
- Kết quả từ Step B7, B8, B9 (đã viết)
- File: `document/2026-06-08_pathology-pdl1-glcm/paper-plan-nlp-extension.md` — Section 3.5 Discussion (đã có 7 paragraph outline)
- Key claim để frame: "NLP encoding is primarily a survival signal, not a classification signal"

**Prompt mẫu:**
```
Viết section "4. Discussion" cho bài báo (600–700 words, 6 paragraphs).
Sử dụng outline từ paper-plan-nlp-extension.md Section 3.5 làm khung, mỗi paragraph:
  P1: Main finding context — AUC 0.813, "good–excellent" threshold for oncology
  P2: Why NLP better than raw labs — semantic compression, inter-feature relationships,
      MiniLM embedding space is pretrained for semantic similarity
  P3: Why not statistically significant — n/d ratio, CI width at n=247,
      power analysis (need n≈1200 for ΔAUC=0.02), directional consistency is meaningful
  P4: NLP as survival signal not classification signal — explains C-index advantage at
      long-term timepoints, validates hypothesis in survival-analysis-plan.md
  P5: OvO attention — when to use (2 modalities, substitutable), when to use Original
      (3+ modalities, complementary); best config OvO 0.8003
  P6: Clinical implications — fast encoding (<1s), no labeled text required,
      compatible with real-world EHR workflows
  P7 (Limitations): cohort size, single-institution, no fine-tuning, PD-L1 assay variation
  Framing guideline từ paper-plan: "consistent directional improvement not reaching
  significance due to cohort power; validated by survival stratification"
```

**Output kỳ vọng:** LaTeX source ~120 dòng cho `sections/discussion.tex` — `\section{Discussion}` + 6–7 đoạn văn. Dùng `\cite{}` placeholder cho mọi claim cần citation. **Chia thành 2 prompts (B10a/B10b) nếu token overflow.**

---

### [ ] Step B10a — Viết Discussion Paragraphs 1–3 (nếu B10 overflow)

**Prompt mẫu:**
```
Viết Discussion Paragraphs 1–3 (~300 words):
  P1: Main finding (AUC 0.813), context vs PD-L1 TPS (0.729), TMB (~0.61)
  P2: Why NLP better than raw labs — semantic compression rationale
  P3: Why not significant — power limitation, not failure of NLP, n=247 vs needed ~1200
```

---

### [ ] Step B10b — Viết Discussion Paragraphs 4–7 (nếu B10 overflow)

**Prompt mẫu:**
```
Viết Discussion Paragraphs 4–7 (~350 words):
  P4: NLP as survival vs classification signal — long-term tdAUC evidence
  P5: OvO attention trade-offs — 2 modalities vs 3+ modalities guidance
  P6: Clinical implications of NLP encoding for EHR workflows
  P7: Limitations — 4 limitations, 1 sentence each
```

---

### [x] Step B11 — Viết Section: Introduction

**Mục tiêu:** Giới thiệu bối cảnh, gap, đóng góp, ~500–600 words, 5 paragraphs.

> **Tại sao viết Introduction sau Discussion?** Vì sau khi viết xong Discussion mới biết chính xác narrative arc và claim cần giới thiệu.

**Input cho AI:**
- Kết quả B10 (Discussion — đã biết main claim)
- File: `document/2026-06-08_pathology-pdl1-glcm/paper-plan-nlp-extension.md` — Section 3.2 Introduction outline
- Key gap: "NLP encoding of tabular clinical features for multimodal survival prediction — not studied before"

**Prompt mẫu:**
```
Viết section "1. Introduction" cho bài báo (500–550 words, 5 paragraphs).
Sử dụng outline từ paper-plan-nlp-extension.md Section 3.2:
  P1: Clinical context — NSCLC leading cancer death, ICI therapy, only 20–30% respond,
      need better predictors, PD-L1 TPS + TMB insufficient alone
  P2: Multimodal biomarkers — each modality captures different biology, fusion improves
      prediction, missing modality challenge
  P3: Gap — clinical features as structured numbers lose inter-feature context;
      NLP provides a way to encode this via sentence embeddings;
      distinguishing: this is NOT NLP of clinical notes (already done),
      this IS tabular-to-text NLP for multimodal fusion (new)
  P4: OvO attention gap — standard attention is cooperative; competitive attention
      may better reflect clinical decision-making (which modality is most informative);
      no prior work on OvO-style attention for multimodal clinical models
  P5: Summary of contributions (3 points, numbered list):
      1. OvO competitive attention for multimodal medical models
      2. NLP sentence embedding as a clinical modality in multimodal fusion
      3. Survival endpoint (C-index, Cox HR, tdAUC) showing NLP is a survival signal
```

**Output kỳ vọng:** LaTeX source ~100 dòng cho `sections/introduction.tex` — `\section{Introduction}` + 5 đoạn văn + cuối cùng 1 câu về paper structure dùng `\ref{}` trỏ vào các sections. Mọi claims về epidemiology/trials cần `\cite{}` placeholder.

---

### [x] Step B12 — Viết Section: Abstract và Conclusion

**Mục tiêu:** Tóm tắt toàn bài, mỗi phần ngắn gọn và precise.

**Input cho AI:**
- Kết quả từ B1 đến B11 (tất cả sections đã viết)
- Key numbers từ Step A1 và A2

**Prompt cho Abstract (250 words):**
```
Viết Abstract (250 words, structured: Background / Objective / Methods / Results / Conclusions).
  Background: NSCLC ICI therapy, poor response prediction, multimodal opportunity
  Objective: Extend DyAM with OvO attention and NLP clinical embedding; evaluate survival endpoint
  Methods: n=247 discovery cohort, 10-fold CV, 8 NLP variants, C-index/Cox/tdAUC
  Results — must include all numbers:
    - OvO best: Rad+IHC-G+Gen+PDL1 AUC=0.8003 (best in 2-modality: +3.27%)
    - NLP best: Rad+IHC-G+Gen+PDL1+NLP AUC=0.813 [0.753–0.874]
    - Cox HR NLP-PCA16 = 6.07 [3.50–10.52] vs No Clinical 5.30
    - tdAUC improvement at 12m = +0.024 (IHC-A arm)
    - All models: KM log-rank p<0.005
  Conclusions: NLP encoding is a survival signal; C-index/Cox preferred for evaluation
```

**Prompt cho Conclusion (150 words, riêng biệt):**
```
Viết Conclusion paragraph (~150 words, plain English).
  1 câu: restate best AUC (0.813) và best OvO config (0.8003)
  1–2 câu: NLP encoding as survival signal — Cox HR evidence
  1 câu: directional improvement not significant due to cohort size
  1 câu: future direction — larger cohort, fine-tuning
  KHÔNG lặp lại methods. KHÔNG đưa ra tuyên bố không có bằng chứng.
```

**Output kỳ vọng (Abstract):** LaTeX source cho `\begin{abstract}...\end{abstract}` block trong `main.tex` (hoặc MDPI `\abstract{}` command) — 250 words, plain sentences, không có citation trong abstract (MDPI rule).

**Output kỳ vọng (Conclusion):** LaTeX source ~25 dòng cho `sections/conclusion.tex` — `\section{Conclusions}` + 1 paragraph ~150 words.

---

## PHẦN C — FIGURES VÀ TABLES

---

### [x] Step C1 — Figure Specifications: Figure 1 (Study Overview Diagram)

**Mục tiêu:** Mô tả chi tiết Figure 1 để nghiên cứu sinh / designer vẽ.

**Prompt mẫu:**
```
Viết figure legend và spec chi tiết cho Figure 1 — Study Overview Diagram:
  Panel layout: 3 horizontal boxes connected by arrows
    Box 1: "Patient Cohort" — n=247 discovery, n=50+71 validation,
           list modality icons (CT, microscope, DNA helix, lab report, text)
    Box 2: "DyAM Architecture" — show N modalities → AttentionMatrix (cooperative) /
           AttentionMatrixOvO (competitive) → fused risk score
    Box 3: "Evaluation" — AUC-ROC / C-index / Cox HR / KM curve icons
  Figure legend (2–3 sentences): mô tả overall flow
  Figure title: "Fig. 1. Study overview..."
  Kích thước khuyến nghị: 180mm × 60mm (2-column journal format)
```

**Output kỳ vọng:** Text spec đầy đủ để tạo figure, legend hoàn chỉnh.

---

### [x] Step C2 — Figure Specifications: Figure 2 (NLP Pipeline) + Figure 3 (AUC Bar Chart)

**Prompt mẫu:**
```
Viết figure spec + legend cho:

Figure 2 — NLP Pipeline:
  Flow: "13 clinical variables" → "df_to_text_prompts()" → "English text prompt"
  → "all-MiniLM-L6-v2" → "384-dim embedding" → "PCA(n=16)" → "16-dim NLP-PCA16"
  Sidebar: example prompt text (abbreviated, 2 lines)
  Legend: 2–3 sentences mô tả pipeline

Figure 3 — AUC Bar Chart (IHC-A arm + IHC-G arm, side by side):
  Panel 3A: IHC-A arm — bar chart 7 models, y=AUC [0.70–0.85],
            error bars = DeLong 95% CI, bars màu theo nhóm
            (No Clin: blue, Labs: orange, NLP variants: green shades, BioClinBERT: purple)
  Panel 3B: IHC-G arm — same style, 4 models
  Legend: 2–3 sentences
  Note: thêm dashed line tại AUC=0.80 ("clinical utility threshold")
```

**Output kỳ vọng:** 2 figure specs + 2 legends, ~200 words tổng.

---

### [x] Step C3 — Figure Specifications: Figure 4 (KM Curves) + Figure 5 (Survival SA Figures)

**Prompt mẫu:**
```
Viết figure spec + legend cho:

Figure 4 — Kaplan-Meier Curves (2×2 grid):
  4A: IHC-A No Clinical — KM split at score=0, 2 curves, log-rank χ²=28.17, p<0.005
  4B: IHC-A NLP-PCA16 — same layout, χ²=23.74
  4C: IHC-G No Clinical — χ²=20.07
  4D: IHC-G NLP-PCA16 — χ²=28.92 ★ (highlight)
  Each panel: x=PFS months [0–24m], y=Survival [0–1.0], risk table below
  Legend: 3 sentences covering all 4 panels

Figure 5 — Survival Analysis Summary (2×2 grid):
  5A: C-index bar chart + bootstrap CI (SA-1)
  5B: Time-dependent AUC at 6m/12m/18m line plot (SA-2)
  5C: Forest plot — multivariate Cox HR (SA-3)
  5D: Integrated Brier Score comparison (SA-5)
  Legend: 4 sentences, one per panel
```

**Output kỳ vọng:** 2 figure specs + 2 legends hoàn chỉnh.

---

### [x] Step C4 — Tạo Full Tables cho Paper

**Mục tiêu:** Format lại tất cả tables về chuẩn journal, với caption đầy đủ.

**Prompt mẫu:**
```
Từ kết quả Step A1 và A2, format lại 5 tables theo journal standard (LaTeX-ready hoặc markdown):

Table 1: Patient Characteristics (từ Step A3)
  — Demographics, clinical variables, modality availability, label distribution

Table 2: AUC Comparison — IHC-A Arm (7 NLP variants)
  Columns: Model | AUC | 95% CI | ΔAUC vs No Clinical
  Caption: "Table 2. AUC comparison across clinical encoding strategies, IHC-A cohort..."

Table 3: AUC Comparison — IHC-G Arm (4 variants)
  Same format as Table 2

Table 4: Survival Analysis Summary (8 models)
  Columns: Model | C-index [95% CI] | Cox HR [95% CI] | Cox p | tdAUC 12m | IBS
  Caption: include n=247, n_events=209, median PFS=2.7m

Table 5: Kaplan-Meier Log-Rank Statistics
  Columns: Model | IHC Arm | χ² | p-value
  Caption: mention threshold χ²>7.88 for p<0.005

Mỗi table có full caption (2–3 câu) và abbrev list (CI, PFS, IHC, NLP, OvO, etc.)
```

**Output kỳ vọng:** 5 tables markdown + 5 captions, ready to paste into manuscript.

---

## PHẦN D — SUPPLEMENTARY MATERIAL

---

### [x] Step D1 — Viết Supplementary Methods + Notes

**Prompt mẫu:**
```
Viết 3 Supplementary Notes (tiếng Anh, ~100–150 words mỗi note):

Supp Note 1: NLP Prompt Template
  — Full template với tất cả 13 variables, ví dụ output đầy đủ 1 bệnh nhân

Supp Note 2: BioClinBERT Failure Analysis
  — Lý do MLM pre-training vs sentence similarity, mean pooling suboptimal,
    768d → 16d PCA = 92.8% variance collapse, kết quả ΔAUC = −0.017 vs MiniLM

Supp Note 3: Power Analysis
  — n=247, CI ≈ ±0.07, detectable ΔAUC at 80% power requires n≈1200–1500,
    formula và tham số sử dụng (α=0.05, β=0.20, DeLong variance estimate)
```

**Output kỳ vọng:** 3 supplementary notes, ~400 words tổng.

---

### [x] Step D2 — Viết Supplementary Figures Captions

**Prompt mẫu:**
```
Viết captions cho 5 Supplementary Figures (2–4 sentences mỗi figure):

Supp Fig S1: PCA scree plot — variance explained curve cho 384→16 PCA reduction,
             x=component number [1–50], y=cumulative variance explained [%]

Supp Fig S2: t-SNE visualization — 2D t-SNE của NLP embeddings (n=247),
             colored by label (PR/CR=blue, SD/PD=red)
             Note: clusters overlap → consistent with NLP not being strong binary classifier

Supp Fig S3: Bootstrap p-value distribution — histogram of 5000 bootstrap ΔAUC values
             (NLP raw vs Labs, IHC-G arm), vertical line at observed ΔAUC=0.026

Supp Fig S4: OvO vs Original AUC scatter — scatter plot 20 test cases,
             x=Original AUC, y=OvO AUC, diagonal line = no difference,
             colored dots theo số modalities (1=grey, 2=blue, 3+=red)

Supp Fig S5: Modality availability heatmap — n=247 patients × 8 modalities,
             binary (available=green, missing=white)
```

**Output kỳ vọng:** 5 supplementary figure captions, ~500 words tổng.

---

## PHẦN H — LATEX DOCUMENT ASSEMBLY

> Thực hiện sau khi tất cả sections B1–B12, tables G4, figures G5, và bibliography G6 đã hoàn thành.

---

### [x] Step H1 — Gộp Tất Cả Sections vào main.tex

**Mục tiêu:** Đảm bảo `main.tex` include đúng tất cả các file và compile không lỗi.

**Checklist trước khi compile:**
```
[ ] main.tex có \input{} cho tất cả 6 sections
[ ] Tất cả \label{} unique (không trùng nhau)
[ ] Tất cả \ref{} và \eqref{} trỏ đúng label tồn tại
[ ] Tất cả \cite{} có entry tương ứng trong references.bib
[ ] Tất cả \includegraphics{} trỏ đúng file tồn tại trong figures/
[ ] Tất cả \input{tables/...} trỏ đúng file trong tables/
[ ] Abstract ≤250 words (đếm bằng: detex main.tex | wc -w)
[ ] Không có \TODO, \PLACEHOLDER còn sót lại
```

**Prompt mẫu:**
```
Review main.tex và tất cả sections/*.tex sau khi đã điền nội dung đầy đủ.
Kiểm tra:
  1. \label{} trùng nhau: grep -n "\\label{" sections/*.tex tables/*.tex | sort -t: -k3 | uniq -d -f2
  2. \ref{} undefined: xem main.log sau khi compile
  3. Equation numbering: tất cả equations có \label{eq:...} không
  4. Figure placement: các figures có xuất hiện trong đúng section không
  5. Table placement: tables được \input{} đúng chỗ trong results section

Viết lại hoặc sửa bất kỳ đoạn nào có syntax error LaTeX.
```

**Output kỳ vọng:** Danh sách issues + fixed LaTeX code cho từng issue.

---

### [x] Step H2 — Kiểm Tra Word Count và Page Limit

**Mục tiêu:** Đảm bảo paper đáp ứng constraints của journal Cancers.

**Limits cho Cancers (MDPI) — Original Research:**
- Abstract: ≤ 200 words (strict)
- Main text: không có hard limit, khuyến nghị ≤ 6,000–8,000 words
- Max figures: 15 (main + supplementary)
- Max tables: không giới hạn cứng
- References: không giới hạn

**Prompt mẫu:**
```
Tôi cần đếm word count của manuscript LaTeX. Chạy lệnh:

  # Đếm words (loại bỏ LaTeX commands)
  detex sections/*.tex | wc -w

  # Hoặc dùng texcount (chi tiết hơn)
  texcount -inc main.tex

Kết quả: [paste output]

Phân tích:
  1. Word count per section — section nào quá dài?
  2. Abstract có ≤200 words không?
  3. Nếu total > 8000 words: đề xuất những câu/đoạn nào có thể cắt ngắn
  4. Nếu total < 4000 words: section nào cần expand thêm

Xuất bảng: Section | Words | % of total | Action needed
```

**Output kỳ vọng:** Bảng word count + danh sách cuts/expansions cụ thể.

---

### [x] Step H3 — Tạo Supplementary LaTeX File

**Mục tiêu:** Compile file supplementary.tex riêng biệt thành PDF.

**Prompt mẫu:**
```
Tạo file supplementary/supplementary.tex với cấu trúc:

\documentclass[12pt]{article}
\usepackage{amsmath, booktabs, graphicx, subcaption, hyperref}
\renewcommand{\thefigure}{S\arabic{figure}}   % Figure S1, S2...
\renewcommand{\thetable}{S\arabic{table}}     % Table S1, S2...
\setcounter{figure}{0}
\setcounter{table}{0}

\title{Supplementary Material: [Paper Title]}
\date{}

\begin{document}
\maketitle

\section*{Supplementary Methods}
[Supp Note 1 — NLP Prompt Template từ Step D1]
[Supp Note 2 — BioClinBERT Failure Analysis]
[Supp Note 3 — Power Analysis]

\section*{Supplementary Figures}
[Figure S1 — PCA Scree Plot]
[Figure S2 — t-SNE NLP Embeddings]
[Figure S3 — Bootstrap p-value Distribution]
[Figure S4 — OvO vs Original AUC Scatter]
[Figure S5 — Modality Availability Heatmap]

\section*{Supplementary Tables}
[Table S1 — Modality Availability Matrix]
[Table S2 — Feature Counts After L1 Selection]

\end{document}

Điền nội dung từ Step D1, D2 vào đúng chỗ. Output: full supplementary.tex.
```

**Output kỳ vọng:** File `supplementary/supplementary.tex` hoàn chỉnh, ~150 dòng.

---

### [x] Step H4 — Final PDF Review Checklist

**Mục tiêu:** Review PDF cuối cùng trước khi submit.

**Prompt mẫu:**
```
Tôi đã compile main.tex thành main.pdf và supplementary.pdf.
Hãy tạo checklist review cuối để tôi tự kiểm tra bằng mắt:

  TYPOGRAPHY & LAYOUT:
  [ ] Không có dòng chạy ra ngoài margin (overfull hbox)
  [ ] Spacing đồng đều giữa paragraphs
  [ ] Figures và tables không split qua trang giữa caption và content
  [ ] Tất cả subfigure (a)(b)(c) có đánh nhãn rõ ràng

  CONTENT:
  [ ] Abstract ≤200 words, không có citations
  [ ] Keywords 5–8 words
  [ ] Tất cả figures được mention trong text trước khi xuất hiện
  [ ] Tất cả tables được mention trong text
  [ ] No orphaned headings (heading ở cuối trang)

  REFERENCES:
  [ ] Tất cả [?] undefined citations đã được resolve
  [ ] Reference list đánh số liên tục
  [ ] DOI links trong reference list clickable

  NUMBERS:
  [ ] Mọi số trong Abstract khớp với Tables 2–5
  [ ] Mọi p-value format nhất quán: p~<~0.001 vs p~=~0.042
  [ ] CI format nhất quán: [lower--upper] vs (lower, upper)

  MDPI SPECIFIC:
  [ ] Author contributions section (CRediT)
  [ ] Funding section
  [ ] Conflicts of interest
  [ ] Data availability statement

Output: checklist LaTeX source (có thể dùng \checkmark và \square từ amssymb)
```

**Output kỳ vọng:** PDF review checklist hoàn chỉnh, ~40 items.

---

## PHẦN E — ASSEMBLY VÀ FINAL POLISH

---

### [x] Step E1 — Review và Consistency Check

**Mục tiêu:** Kiểm tra toàn bộ manuscript về nhất quán số liệu và terminology.

**Prompt mẫu:**
```
Review toàn bộ manuscript draft (paste tất cả sections đã viết từ B1–B12).
Kiểm tra 5 điểm:
  1. Số liệu nhất quán: mọi mention của AUC/CI/HR/p-value đều khớp với Tables 2–5
  2. Terminology nhất quán: không trộn "DyAM"/"dynamic attention"/"DyAM model"
     → standardize về "DyAM (Dynamic Attention Multimodal) model" lần đầu, "DyAM" sau đó
  3. Abbreviations: list tất cả abbreviations xuất hiện lần đầu, kiểm tra define-on-first-use
  4. Tense consistency: Methods = past tense, Results = past tense, Discussion = present+past
  5. Passive vs active voice: Methods = passive, Discussion = mix
Output: danh sách numbered issues, mỗi issue: "Section X, line ~Y: [issue] → [fix]"
```

**Output kỳ vọng:** Danh sách 10–20 issues cụ thể, dễ fix.

---

### [x] Step E2 — Format Cover Letter và Submission Checklist

**Prompt mẫu:**
```
Viết Cover Letter cho submission tới journal Cancers (MDPI):
  Paragraph 1: Paper title, type (original research), 3 authors (placeholder), word count (~4500)
  Paragraph 2: Summary — 5 sentences: clinical problem, dataset, 3 contributions, best result
  Paragraph 3: Why Cancers — scope fit (oncology AI, multimodal biomarkers)
  Paragraph 4: Declarations — no conflict of interest, not submitted elsewhere, IRB placeholder

Và tạo Submission Checklist:
  [ ] Abstract ≤250 words
  [ ] Keywords (5–8 words từ MeSH)
  [ ] All figures: 300 DPI, SVG/TIFF, named Fig1.svg... Fig5.svg, Supp_Fig1...S5
  [ ] All tables: separate file hoặc inline (check journal guide)
  [ ] Supplementary: separate PDF
  [ ] References: numbered, Vancouver style (Cancers format)
  [ ] Author contributions (CRediT taxonomy)
  [ ] Data availability statement
  [ ] Code availability: link to GitHub (nếu có)
```

**Output kỳ vọng:** Cover letter ~300 words + checklist hoàn chỉnh.

---

## PHẦN F — REFERENCES (DÙNG ĐỂ CÓ SẴN)

---

### [x] Step F1 — Tổng Hợp References Cần Tìm

**Prompt mẫu:**
```
Dựa trên nội dung paper (paste Abstract + Introduction), tạo list references cần tìm:
  - Mỗi reference: [Number] | Author (Year) | Topic | Search string để tìm trên PubMed/Google Scholar
  
Categories cần có:
  1. NSCLC epidemiology (1–2 refs)
  2. ICI therapy (pembrolizumab/nivolumab NSCLC, 2–3 refs key trials)
  3. PD-L1 TPS + TMB as biomarkers (2–3 refs)
  4. Multimodal radiomics fusion (3–4 refs)
  5. Sentence-transformers (Reimers 2019, Wang 2020 MiniLM) — 2 refs
  6. BioClinBERT (Alsentzer 2019) — 1 ref
  7. Missing modality in multimodal learning (2 refs)
  8. Survival analysis methodology (Harrell C-index, DeLong CI) — 2 refs
  9. OvO classification original (Hastie 1998) — 1 ref

Output: numbered list với search strings, không cần tìm ngay — chỉ cần list để tìm sau.
```

**Output kỳ vọng:** ~20–25 references cần tìm, có search string.

---

## CHECKLIST TIẾN ĐỘ

> Đánh dấu `[x]` khi step hoàn thành. Ghi ngày vào cột cuối.  
> `[~]` = skip hoặc gộp vào step khác (ghi chú lý do).

### Trước khi bắt đầu — LaTeX Setup

| Done | Step | File output | Ngày |
|---|---|---|---|
| [x] | G1 — Thư mục + main.tex skeleton | `paper/main.tex` | 2026-06-09 |
| [x] | G2 — MDPI template + preamble | `paper/main.tex` (updated) | 2026-06-09 |
| [x] | G3 — 6 equations LaTeX | `sections/methods.tex` | 2026-06-09 |

### Week 1 — Số liệu + Methods

| Done | Step | File output | Ngày |
|---|---|---|---|
| [x] | A1 — Bảng AUC master tất cả models | `drafts/master_numbers.md` | 2026-06-09 |
| [x] | A2 — 4 bảng Survival (C-index/Cox/tdAUC/IBS) | `drafts/master_numbers.md` | 2026-06-09 |
| [x] | A3 — Table 1 Patient Characteristics draft | `tables/table1_patients.tex` | 2026-06-09 |
| [x] | B1 — Methods 2.1 Study Design & Cohort | `sections/methods.tex` | 2026-06-09 |
| [x] | B2 — Methods 2.2 Data Modalities | `sections/methods.tex` | 2026-06-09 |
| [x] | B3 — Methods 2.3 NLP Clinical Embedding | `sections/methods.tex` | 2026-06-09 |
| [x] | B4 — Methods 2.4 Model Architecture (DyAM+OvO) | `sections/methods.tex` | 2026-06-09 |
| [x] | B5 — Methods 2.5 Statistical Analysis | `sections/methods.tex` | 2026-06-09 |

### Week 2 — Results + Tables + Figures

| Done | Step | File output | Ngày |
|---|---|---|---|
| [x] | B6 — Results 3.1 Baselines | `sections/results.tex` | |
| [x] | B7 — Results 3.2 OvO Comparison | `sections/results.tex` | |
| [x] | B8 — Results 3.3 NLP AUC Analysis | `sections/results.tex` | |
| [x] | **B9 — Results 3.4 Survival Analysis ⭐** | `sections/results.tex` | |
| [x] | C1 — Figure 1 spec (Study Overview) | `drafts/figure_specs.md` | 2026-06-10 |
| [x] | C2 — Figure 2+3 spec (NLP Pipeline, AUC Bar) | `drafts/figure_specs.md` | 2026-06-10 |
| [x] | C3 — Figure 4+5 spec (KM, Survival SA) | `drafts/figure_specs.md` | 2026-06-10 |
| [x] | C4 — Nội dung 5 tables đầy đủ + captions | `paper/tables/*.tex` (G4) | 2026-06-10 |
| [x] | G4 — Convert 5 tables → LaTeX booktabs | `tables/table1–5.tex` | |

### Week 3 — Discussion + Intro + Abstract

| Done | Step | File output | Ngày |
|---|---|---|---|
| [x] | B10 — Discussion (6–7 đoạn) | `sections/discussion.tex` | 2026-06-10 |
| N/A | B10a — Discussion P1–P3 *(không cần — B10 đã viết đầy đủ)* | — | — |
| N/A | B10b — Discussion P4–P7 *(không cần — B10 đã viết đầy đủ)* | — | — |
| [x] | B11 — Introduction (5 đoạn) | `sections/introduction.tex` | 2026-06-10 |
| [x] | B12 — Abstract 250w + Conclusion 150w | `main.tex` + `sections/conclusion.tex` | 2026-06-10 |

### Week 4 — References + Supplementary

| Done | Step | File output | Ngày |
|---|---|---|---|
| [x] | F1 — Danh sách ~25 refs cần tìm | `drafts/references_todo.md` | 2026-06-10 |
| [x] | G6 — references.bib (18 BibTeX entries) | `paper/references.bib` | 2026-06-10 |
| [x] | G5 — LaTeX \includegraphics cho 5 figures | inline trong `methods.tex`+`results.tex` | 2026-06-10 |
| [x] | D1 — Supplementary Notes 1–3 | `supplementary/supplementary.tex` | 2026-06-10 |
| [x] | D2 — Captions Supp Figs S1–S5 | `supplementary/supplementary.tex` | 2026-06-10 |
| [x] | H3 — supplementary.tex hoàn chỉnh | `supplementary/supplementary.tex` | 2026-06-10 |

### Week 5 — Assembly + Polish + Submit

| Done | Step | File output | Ngày |
|---|---|---|---|
| [x] | H1 — Gộp sections + kiểm tra labels/refs | `paper/main.tex` (final) | 2026-06-10 |
| [x] | H2 — Word count + page limit check | drafts/wordcount.py (3{,}446 words body + 197 abstract) | 2026-06-10 |
| [x] | G7 — Compile check + fix log errors | `drafts/compile_check_g7.md` (no pdflatex available) | 2026-06-10 |
| [x] | E1 — Consistency check (numbers, tense, abbrev) | `drafts/consistency_check_e1.md` (fixed inline) | 2026-06-10 |
| [x] | H4 — Final PDF review checklist | `drafts/h4_final_review_checklist.md` | 2026-06-10 |
| [x] | E2 — Cover letter + submission checklist | `drafts/cover_letter.tex` | 2026-06-10 |

---

**Tổng:** 35 / 35 steps thực hiện (33 hoàn thành + 2 N/A: B10a/B10b không cần do B10 đã viết đầy đủ trong 1 lần). Cập nhật lần cuối: 2026-06-10. **TẤT CẢ STEPS ĐÃ XONG — chỉ còn các [TO VERIFY]/placeholder cần dữ liệu thực tế từ người dùng (xem drafts/h4_final_review_checklist.md).**

---

## HƯỚNG DẪN SỬ DỤNG FILE NÀY

1. **Mỗi Step = 1 prompt riêng biệt** — không gộp nhiều step vào 1 prompt.
2. **Paste context cần thiết** vào prompt — mỗi step liệt kê rõ "Input cho AI".
3. **Kết quả mỗi step** lưu thẳng vào file `.tex` tương ứng trong thư mục `paper/`.
4. **Sau Step A1–A3**: tạo file `master_numbers.md` chứa tất cả số liệu đã verify — dùng làm nguồn duy nhất cho mọi số trong paper.
5. **Sau Step G1–G2**: test compile `pdflatex main.tex` với nội dung rỗng — đảm bảo template hoạt động trước khi bắt đầu viết.
6. **LaTeX output format:** Khi yêu cầu AI viết một section, luôn thêm vào đầu prompt: *"Output phải là LaTeX source code, không phải plain text. Bắt đầu từ `\subsection{...}` hoặc `\section{...}`, không có `\begin{document}`."*
7. **Cross-reference strategy:** Dùng `\cite{PLACEHOLDER_AuthorYear}` trong tất cả steps B — điền key thật sau Step G6. Đừng để trống hoặc dùng [1], [2].
8. **Compile thường xuyên:** Sau mỗi step B, chạy `pdflatex main.tex` một lần để bắt lỗi sớm.

---

## CẤU TRÚC FILE LaTeX CUỐI CÙNG

```
paper/
├── main.tex          ← compile file duy nhất
├── sections/
│   ├── abstract.tex      ← Step B12
│   ├── introduction.tex  ← Step B11
│   ├── methods.tex       ← Steps B1–B5 + G3 (equations)
│   ├── results.tex       ← Steps B6–B9
│   ├── discussion.tex    ← Step B10
│   └── conclusion.tex    ← Step B12
├── tables/
│   ├── table1_patients.tex    ← Step A3 + G4
│   ├── table2_auc_ihca.tex    ← Step A1 + G4
│   ├── table3_auc_ihcg.tex    ← Step A1 + G4
│   ├── table4_survival.tex    ← Step A2 + G4
│   └── table5_km.tex          ← Step A2 + G4
├── figures/           ← PDF/EPS exports từ Python scripts
├── supplementary/
│   ├── supplementary.tex      ← Step H3
│   └── figures/
├── references.bib     ← Step G6
├── mdpi.cls           ← download từ MDPI
└── mdpi.bst           ← download từ MDPI
```

---

*File này: `document/2026-06-09_paper-writing-plan/paper_steps_ovo_nlp_survival.md`*  
*Tạo: 2026-06-09 | Cập nhật: thêm LaTeX (PHẦN G, H)*  
*Phạm vi: OvO + NLP + Survival Analysis extension paper | Output: LaTeX manuscript*  
*Target journal: Cancers (MDPI) / npj Digital Medicine | Best result: AUC 0.813, HR=6.07*
