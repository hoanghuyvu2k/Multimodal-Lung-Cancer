# -*- coding: utf-8 -*-
"""Build a plain Word (.docx) version of the manuscript, independent of the
MDPI LaTeX template. Basic scientific-paper structure: Title / Authors /
Abstract / Keywords / Introduction / Methods / Results / Discussion /
Conclusions / Declarations / References.

Quy trình (xem thêm CLAUDE.md mục "Word manuscript export rule"):
1. Nội dung ở đây được chuyển TAY từ paper/sections/*.tex + tables/*.tex —
   khi sections thay đổi, phải cập nhật text tương ứng trong file này.
2. Hình PDF-only phải convert sang PNG trước (pdftocairo -png -r 200
   -singlefile <fig>.pdf figures/converted/<fig>) — script này chỉ nhúng PNG.
3. Công thức toán dùng OMML (builder nhúng ngay trong file, namespace `m`) —
   display qua add_equation(), inline giữa text qua P_mix(..., IM(...), ...).
4. Chạy:  python paper/word_export/build_manuscript_docx.py
   Output: paper/manuscript_word.docx (nếu file đang mở trong Word sẽ bị
   khoá — script tự fallback sang manuscript_word_v2.docx, v3, ...).
"""
import re
import types
from pathlib import Path

from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ════════════════════════════════════════════════════════════ OMML builder ═
# Minimal Office Math Markup Language builder — sinh equation gốc của Word
# (biến số nghiêng, phân số xếp chồng, chỉ số trên/dưới, Σ có cận, ngoặc
# tự giãn) thay vì text Unicode gần đúng.


def _M(tag):
    return OxmlElement(f"m:{tag}")


def _set_val(el, tag, val):
    child = _M(tag)
    child.set(qn("m:val"), val)
    el.append(child)
    return child


def _omml_run(text, upright=False):
    """Một math run. upright=True cho tên hàm (tanh, softplus...); mặc định
    nghiêng theo quy ước biến số toán học."""
    r = _M("r")
    if upright:
        rpr = _M("rPr")
        _set_val(rpr, "sty", "p")
        r.append(rpr)
    t = _M("t")
    t.text = text
    t.set(qn("xml:space"), "preserve")
    r.append(t)
    return r


def _omml_group(*elements):
    """Gom nhiều phần tử thành 1 'base' giả (được _flatten_omml trải phẳng)."""
    wrapper = _M("_group_")
    for e in elements:
        wrapper.append(e)
    return wrapper


def _flatten_omml(parent, el):
    if el.tag == qn("m:_group_"):
        for c in list(el):
            parent.append(c)
    else:
        parent.append(el)


def _e_wrap(*elements):
    e = _M("e")
    for el in elements:
        _flatten_omml(e, el)
    return e


def _omml_as_list(x):
    if isinstance(x, str):
        return [_omml_run(x)]
    if isinstance(x, (list, tuple)):
        return [_omml_run(i) if isinstance(i, str) else i for i in x]
    return [x]


def _omml_sub(base, sub_val):
    sSub = _M("sSub")
    sSub.append(_e_wrap(*_omml_as_list(base)))
    subEl = _M("sub")
    for el in _omml_as_list(sub_val):
        _flatten_omml(subEl, el)
    sSub.append(subEl)
    return sSub


def _omml_sup(base, sup_val):
    sSup = _M("sSup")
    sSup.append(_e_wrap(*_omml_as_list(base)))
    supEl = _M("sup")
    for el in _omml_as_list(sup_val):
        _flatten_omml(supEl, el)
    sSup.append(supEl)
    return sSup


def _omml_subsup(base, sub_val, sup_val):
    s = _M("sSubSup")
    s.append(_e_wrap(*_omml_as_list(base)))
    subEl = _M("sub")
    for el in _omml_as_list(sub_val):
        _flatten_omml(subEl, el)
    s.append(subEl)
    supEl = _M("sup")
    for el in _omml_as_list(sup_val):
        _flatten_omml(supEl, el)
    s.append(supEl)
    return s


def _omml_frac(num, den):
    f = _M("f")
    f.append(_M("fPr"))
    numEl = _M("num")
    for el in _omml_as_list(num):
        _flatten_omml(numEl, el)
    denEl = _M("den")
    for el in _omml_as_list(den):
        _flatten_omml(denEl, el)
    f.append(numEl)
    f.append(denEl)
    return f


def _omml_nary_sum(sub_val, sup_val, body, chr_="∑"):
    n = _M("nary")
    naryPr = _M("naryPr")
    chrEl = _M("chr")
    chrEl.set(qn("m:val"), chr_)
    naryPr.append(chrEl)
    _set_val(naryPr, "limLoc", "subSup")
    grow = _M("grow")
    _set_val(grow, "val", "1")
    naryPr.append(grow)
    n.append(naryPr)
    if sub_val is not None:
        subEl = _M("sub")
        for el in _omml_as_list(sub_val):
            _flatten_omml(subEl, el)
        n.append(subEl)
    if sup_val is not None:
        supEl = _M("sup")
        for el in _omml_as_list(sup_val):
            _flatten_omml(supEl, el)
        n.append(supEl)
    e = _M("e")
    for el in _omml_as_list(body):
        _flatten_omml(e, el)
    n.append(e)
    return n


def _omml_delim(body, beg="(", end=")"):
    d = _M("d")
    dpr = _M("dPr")
    beg_el = _M("begChr")
    beg_el.set(qn("m:val"), beg)
    end_el = _M("endChr")
    end_el.set(qn("m:val"), end)
    dpr.append(beg_el)
    dpr.append(end_el)
    d.append(dpr)
    e = _M("e")
    for el in _omml_as_list(body):
        _flatten_omml(e, el)
    d.append(e)
    return d


def _omml_func(name, arg):
    f = _M("func")
    fName = _M("fName")
    fName.append(_omml_run(name, upright=True))
    f.append(fName)
    e = _M("e")
    for el in _omml_as_list(arg):
        _flatten_omml(e, el)
    f.append(e)
    return f


def _omml_acc(base, chr_="̂"):
    a = _M("acc")
    accPr = _M("accPr")
    chrEl = _M("chr")
    chrEl.set(qn("m:val"), chr_)
    accPr.append(chrEl)
    a.append(accPr)
    e = _M("e")
    for el in _omml_as_list(base):
        _flatten_omml(e, el)
    a.append(e)
    return a


def _omml_build(paragraph, *elements):
    oMath = _M("oMath")
    for el in elements:
        _flatten_omml(oMath, el)
    paragraph._p.append(oMath)
    return oMath


# Namespace `m` — giữ nguyên cú pháp m.sub(...), m.frac(...) ở phần thân bài.
m = types.SimpleNamespace(
    run=_omml_run, group=_omml_group, sub=_omml_sub, sup=_omml_sup,
    subsup=_omml_subsup, frac=_omml_frac, nary_sum=_omml_nary_sum,
    delim=_omml_delim, func=_omml_func, acc=_omml_acc,
    build_oMath=_omml_build,
)
# ══════════════════════════════════════════════════════ end OMML builder ═

_HERE = Path(__file__).resolve().parent
PAPER_DIR = _HERE.parent
OUT_PATH = str(PAPER_DIR / "manuscript_word.docx")
FIGDIR = str(PAPER_DIR / "figures")
FIGDIR_CONV = str(PAPER_DIR / "figures" / "converted")

doc = Document()

# ---------------------------------------------------------------- styles ---
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)
doc.styles["Normal"].paragraph_format.space_after = Pt(8)

for i, size in zip((1, 2, 3), (16, 13, 12)):
    h = doc.styles[f"Heading {i}"]
    h.font.size = Pt(size)
    h.font.bold = True
    h.font.color.rgb = RGBColor(0, 0, 0)
    h.font.name = "Calibri"

sec = doc.sections[0]
sec.left_margin = sec.right_margin = Cm(2.5)
sec.top_margin = sec.bottom_margin = Cm(2.5)

# ------------------------------------------------------------- run markup ---
TOKEN_RE = re.compile(r"(\*\*.+?\*\*|\*.+?\*|\^\{.+?\}|_\{.+?\})")


def add_markup(paragraph, text):
    """Very small markup: **bold**, *italic*, ^{sup}, _{sub}."""
    for chunk in TOKEN_RE.split(text):
        if not chunk:
            continue
        if chunk.startswith("**") and chunk.endswith("**"):
            r = paragraph.add_run(chunk[2:-2])
            r.bold = True
        elif chunk.startswith("*") and chunk.endswith("*"):
            r = paragraph.add_run(chunk[1:-1])
            r.italic = True
        elif chunk.startswith("^{") and chunk.endswith("}"):
            r = paragraph.add_run(chunk[2:-1])
            r.font.superscript = True
        elif chunk.startswith("_{") and chunk.endswith("}"):
            r = paragraph.add_run(chunk[2:-1])
            r.font.subscript = True
        else:
            paragraph.add_run(chunk)


def P(text="", style=None, align=None, space_before=None, space_after=None):
    p = doc.add_paragraph(style=style)
    if align is not None:
        p.alignment = align
    if space_before is not None:
        p.paragraph_format.space_before = Pt(space_before)
    if space_after is not None:
        p.paragraph_format.space_after = Pt(space_after)
    if text:
        add_markup(p, text)
    return p


def H(text, level=1):
    return doc.add_heading(text, level=level)


def set_cell_text(cell, text, bold=False, align=None, size=9):
    cell.text = ""
    p = cell.paragraphs[0]
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(2)
    for chunk in TOKEN_RE.split(text):
        if not chunk:
            continue
        if chunk.startswith("**") and chunk.endswith("**"):
            r = p.add_run(chunk[2:-2]); r.bold = True
        else:
            r = p.add_run(chunk)
        r.font.size = Pt(size)
        if bold:
            r.bold = True


def add_table(headers, rows, caption=None, col_widths=None):
    if caption:
        cap = doc.add_paragraph()
        cap.paragraph_format.space_before = Pt(10)
        cap.paragraph_format.space_after = Pt(4)
        add_markup(cap, caption)
        for run in cap.runs:
            run.font.size = Pt(9.5)
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    for i, htext in enumerate(headers):
        set_cell_text(t.rows[0].cells[i], htext, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            align = WD_ALIGN_PARAGRAPH.LEFT if i == 0 else WD_ALIGN_PARAGRAPH.CENTER
            set_cell_text(cells[i], str(val), align=align)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return t


def page_break():
    doc.add_page_break()


_EQ_COUNTER = {"n": 0}


def add_equation(*elements):
    """Insert a native, editable Word equation (OMML), centered, with a
    right-aligned equation number (1), (2)... matching the PDF."""
    from docx.enum.text import WD_TAB_ALIGNMENT
    _EQ_COUNTER["n"] += 1
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    # text width = A4 21cm − 2×2.5cm margins = 16cm
    tabs = p.paragraph_format.tab_stops
    tabs.add_tab_stop(Cm(8.0), WD_TAB_ALIGNMENT.CENTER)
    tabs.add_tab_stop(Cm(16.0), WD_TAB_ALIGNMENT.RIGHT)
    p.add_run("\t")
    m.build_oMath(p, *elements)
    p.add_run(f"\t({_EQ_COUNTER['n']})")
    return p


class IM:
    """Marker for inline math inside a mixed text paragraph."""
    def __init__(self, *elements):
        self.elements = elements


def _fill_mixed(p, parts, font_size=None):
    """Append text (markup) and IM inline-math parts to paragraph p in
    order. Both add_run and oMath append to the same w:p, so document
    order is preserved."""
    for part in parts:
        if isinstance(part, IM):
            m.build_oMath(p, *part.elements)
        else:
            add_markup(p, part)
    if font_size is not None:
        for run in p.runs:
            run.font.size = Pt(font_size)


def P_mix(*parts, align=None):
    """Paragraph with mixed text and inline native equations."""
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    _fill_mixed(p, parts)
    return p


def add_figure(path, caption, width_in=6.0):
    """caption may be a plain markup string or a list of parts (strings
    and IM inline-math markers)."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(10)
    run = p.add_run()
    run.add_picture(path, width=Inches(width_in))
    cap = doc.add_paragraph()
    cap.paragraph_format.space_after = Pt(10)
    parts = caption if isinstance(caption, (list, tuple)) else [caption]
    _fill_mixed(cap, parts, font_size=9.5)


# ==================================================== TITLE / FRONT MATTER =
title_p = doc.add_paragraph()
title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = title_p.add_run(
    "NLP-Augmented Multimodal Attention Fusion for Immunotherapy "
    "Response Prediction in Non-Small Cell Lung Cancer"
)
r.bold = True
r.font.size = Pt(16)
title_p.paragraph_format.space_after = Pt(12)

auth_p = doc.add_paragraph()
auth_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
add_markup(auth_p, "First Author^{1,\u2020}, Second Author^{2}, Third Author^{1,*}")
auth_p.paragraph_format.space_after = Pt(4)

aff_p = doc.add_paragraph()
aff_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
add_markup(
    aff_p,
    "^{1}Department of [Department], [University], [City], [Country]; "
    "^{2}Department of [Department], [University], [City], [Country]"
)
for r in aff_p.runs:
    r.font.size = Pt(9.5)
aff_p.paragraph_format.space_after = Pt(2)

corr_p = doc.add_paragraph()
corr_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
add_markup(corr_p, "**Correspondence:** third@email.com  |  ^{\u2020}These authors contributed equally to this work.")
for r in corr_p.runs:
    r.font.size = Pt(9.5)
corr_p.paragraph_format.space_after = Pt(16)

# ---------------------------------------------------------------- abstract -
H("Abstract", level=2)
P(
    "Predicting response to immune checkpoint inhibitors (ICIs) in non-small "
    "cell lung cancer (NSCLC) remains challenging, as established biomarkers "
    "such as PD-L1 expression provide only modest discrimination. We extended "
    "the Dynamic Attention Multimodal (DyAM) fusion framework, which combines "
    "CT radiomics, pathology, genomics, PD-L1, and clinical data via learned "
    "per-modality attention, with two contributions: a competitive "
    "one-vs-others (OvO) attention mechanism, and a natural language "
    "processing (NLP) encoding of 13 numeric clinical variables using a "
    "pretrained sentence transformer. In a 247-patient discovery cohort, "
    "repeated-seed evaluation (5 seeds \u00d7 10-fold, 21 combinations) placed "
    "OvO within 0.011 AUC of cooperative attention at every primary "
    "benchmark (AUC = 0.773 vs. 0.760, *p* \u2265 0.17), exceeding it in 12 of "
    "21 combinations (best single-partition AUC = 0.800), showing OvO's "
    "simpler formulation matches or exceeds cooperative attention at no "
    "accuracy cost. NLP-encoded clinical features improved AUC over raw "
    "numeric encoding in both pathology arms (best AUC = 0.813 "
    "[0.753\u20130.874], \u0394 = +0.030), a gain that also held for OvO. Survival "
    "analysis showed NLP encoding yielded the highest Cox hazard ratios "
    "(up to 6.07 vs. 5.30) and improved time-dependent AUC at 12\u201318 months, "
    "with all Kaplan\u2013Meier stratifications reaching *p* < 0.005. "
    "NLP-based clinical encoding offers a practical, architecture-agnostic "
    "improvement to multimodal ICI response prediction."
)

kw_p = doc.add_paragraph()
add_markup(
    kw_p,
    "**Keywords:** non-small cell lung cancer; immunotherapy response "
    "prediction; multimodal deep learning; attention mechanism; natural "
    "language processing; sentence embedding; survival analysis; radiomics"
)
kw_p.paragraph_format.space_after = Pt(16)

print("front matter done")

# ======================================================== 1. INTRODUCTION =
H("1. Introduction", level=1)

P(
    "Lung cancer remains the leading cause of cancer-related mortality "
    "worldwide, with non-small cell lung cancer (NSCLC) accounting for "
    "approximately 85% of cases [1]. The introduction of immune checkpoint "
    "inhibitors (ICIs) targeting the PD-1/PD-L1 axis has transformed "
    "treatment for advanced NSCLC, producing durable responses in a subset "
    "of patients [2,3]. However, only 20–30% of unselected patients achieve "
    "an objective response, and a substantial proportion experience primary "
    "or acquired resistance [4]. Reliable pre-treatment identification of "
    "likely responders remains an unresolved clinical need, as it would "
    "allow clinicians to prioritise ICI therapy for patients most likely to "
    "benefit and to consider alternative or combination strategies for "
    "those unlikely to respond."
)

P(
    "PD-L1 tumour proportion score (TPS) and tumour mutational burden (TMB) "
    "are the most widely used biomarkers for ICI response prediction, but "
    "both have well-documented limitations: PD-L1 expression is spatially "
    "heterogeneous and assay-dependent, while TMB alone provides only "
    "modest discriminative performance [5,4]. This has motivated growing "
    "interest in multimodal approaches that integrate radiomics from CT "
    "imaging, pathomics from digitised histology, genomic alterations, and "
    "clinical variables into a single predictive model [6–9]. Such "
    "approaches are appealing because they can capture complementary "
    "aspects of tumour biology and host physiology that no single modality "
    "fully represents; however, they introduce new methodological "
    "challenges, including how to weight modalities of differing "
    "reliability and how to handle missing modalities without discarding "
    "patients [10]. The Dynamic Attention Multimodal (DyAM) framework [11] "
    "addresses these challenges through a learned, per-patient attention "
    "mechanism that adaptively weights each available modality and "
    "naturally accommodates missing data via masking, but its original "
    "formulation used only a cooperative (softmax/L1-normalised) attention "
    "mechanism and represented clinical variables as raw numeric features."
)

P(
    "A largely unexplored question in this setting is how structured "
    "clinical variables — demographics, laboratory values, performance "
    "status — should be represented for fusion with imaging- and "
    "molecular-derived features. The conventional approach treats each "
    "clinical variable as an independent numeric input, requiring the model "
    "to learn inter-variable relationships (e.g., the joint prognostic "
    "implication of advanced age, low albumin, and poor performance status) "
    "from a relatively small training set. Recent advances in natural "
    "language processing (NLP) have shown that pretrained sentence encoders "
    "can represent complex, multi-attribute entities as dense embeddings "
    "that preserve semantic similarity [12,13], and domain-specific "
    "clinical language models have been developed for free-text clinical "
    "notes [14]. However, whether converting *structured, tabular* clinical "
    "variables into natural-language sentences and encoding them with a "
    "general-purpose sentence transformer can improve multimodal fusion "
    "performance — as opposed to encoding free-text clinical narratives "
    "— has not, to our knowledge, been systematically evaluated for ICI "
    "response prediction."
)

P(
    "A second open question concerns the form of the attention mechanism "
    "used to combine modalities. Cooperative attention mechanisms, which "
    "normalise modality weights to sum to one via softmax or L1 "
    "normalisation, implicitly assume that all modalities contribute "
    "complementary, additive information. An alternative, competitive "
    "formulation — in which each modality's weight is determined relative "
    "to the others via a one-vs-others (OvO) comparison — may better "
    "capture settings where modalities are partially redundant and the "
    "most informative modality should dominate. Whether such a competitive "
    "mechanism improves performance, and under what modality "
    "configurations, remains untested in the multimodal ICI response "
    "prediction literature."
)

P(
    "In this study, we extend the DyAM framework with two contributions and "
    "evaluate them on a discovery cohort of 247 patients with advanced "
    "NSCLC treated with ICIs. First, we introduce an OvO competitive "
    "attention variant (AttentionMatrixOvO) and compare it against the "
    "original cooperative attention across 20 modality combinations using "
    "10-fold cross-validation, with a confirmatory repeated-seed evaluation "
    "(5 seeds × 10-fold) across 21 modality combinations to test the "
    "reproducibility of any observed difference. Second, we convert 13 "
    "numeric clinical variables into natural-language sentences and encode "
    "them with a pretrained sentence transformer (MiniLM-L6-v2), comparing "
    "this NLP-based representation against raw numeric encoding, a "
    "domain-specific clinical language model (BioClinBERT), and a "
    "PCA-compressed variant, evaluated using both binary classification "
    "(AUC) and survival analysis (Cox regression, time-dependent AUC, "
    "Harrell's C-index, and Kaplan–Meier stratification). Together, these "
    "contributions aim to clarify (i) whether competitive attention offers "
    "a reproducible accuracy advantage over cooperative attention in "
    "multimodal fusion, or whether the two are functionally interchangeable "
    "at this sample size, and (ii) whether NLP-based encoding of structured "
    "clinical data provides a practical, fine-tuning-free improvement over "
    "conventional numeric representations for both response prediction and "
    "survival risk stratification."
)

print("intro done")

# =================================================== 2. MATERIALS/METHODS =
H("2. Materials and Methods", level=1)

H("2.1. Study Design and Patient Cohort", level=2)
P(
    "This retrospective study used data from the MSK-MIND (Memorial Sloan "
    "Kettering Molecular-Informed Neoadjuvant/Definitive therapy) programme "
    "[11], comprising patients with advanced non-small cell lung cancer "
    "(NSCLC) treated with immune checkpoint inhibitor (ICI) therapy between "
    "2017 and 2021. Inclusion criteria were: (i) histologically confirmed "
    "stage IIIB/IV NSCLC, (ii) first- or second-line ICI therapy "
    "(monotherapy or combination), and (iii) available baseline CT imaging. "
    "The study dataset comprised three independent cohorts: a *discovery "
    "cohort* (n = 247) used for model development and 10-fold "
    "cross-validation; a *radiomics validation cohort* (n = 50) and a "
    "*pathology validation cohort* (n = 71), both held out entirely for "
    "external evaluation. Patient characteristics are summarised in "
    "Table 1."
)

add_table(
    ["Characteristic", "Discovery (n = 247)", "Validation (rad n=50; path n=71)"],
    [
        ["Age, years, mean (range)", "66.9 (38–93)", "64.4 (45–86); 68.0 (30–89)"],
        ["Male sex, n (%)", "113 (45.7%)", "24 (48.0%); 32 (45.1%)"],
        ["Adenocarcinoma", "184 (74.5%)", "39 (78.0%); 47 (66.2%)"],
        ["Squamous cell", "36 (14.6%)", "7 (14.0%); 13 (18.3%)"],
        ["Other / NOS", "27 (10.9%)", "4 (8.0%); 11 (15.5%)"],
        ["ECOG 0", "28 (11.3%)", "10 (20.0%); 8 (11.3%)"],
        ["ECOG 1", "194 (78.5%)", "39 (78.0%); 58 (81.7%)"],
        ["ECOG ≥2", "25 (10.1%)", "1 (2.0%); 5 (7.0%)"],
        ["Anti-PD-1 therapy*", "197 (79.8%)", "not recorded"],
        ["Anti-PD-L1 therapy*", "50 (20.2%)", "not recorded"],
        ["Combination therapy*", "12 (4.9%)", "not recorded"],
        ["Response (PR/CR), label=0", "62 (25.1%)", "11 (22.0%); 21 (29.6%)"],
        ["No response (SD/PD), label=1", "185 (74.9%)", "39 (78.0%); 50 (70.4%)"],
        ["PFS events (progression/death), n (%)", "209 (84.6%)", "46 (92.0%); 55 (77.5%)"],
        ["Median PFS, months (range)", "2.7 (0.1–49.1)", "2.6 (1.0–59.7); 2.7 (0.1–28.4)"],
        ["CT radiomics available (PC/PL/LN)", "187 (75.7%)", "46/50; —"],
        ["Pathology IHC available", "105 (42.5%)", "—; 52/71"],
        ["Genomics (NGS) available", "247 (100%)", "—"],
        ["PD-L1 TPS score available", "201 (81.4%)", "—"],
        ["Clinical labs (13 vars) available", "247 (100%)", "—"],
    ],
    caption=(
        "**Table 1.** Patient and cohort characteristics. The discovery "
        "cohort was used for model development with 10-fold "
        "cross-validation; validation cohorts were held out entirely. "
        "Validation column reports radiomics-evaluable; "
        "pathology-evaluable cohort values, separated by a semicolon. "
        "*ICI categories are recorded as independent flags and are not "
        "mutually exclusive (patients on combination therapy are counted "
        "in more than one row). PR/CR, partial/complete response; SD/PD, "
        "stable/progressive disease; PFS, progression-free survival."
    ),
)

P(
    "The primary binary endpoint was best overall response to ICI therapy, "
    "classified as *response* (partial or complete response, PR/CR; "
    "label = 0) or *no response* (stable or progressive disease, SD/PD; "
    "label = 1). The response-to-no-response ratio in the discovery cohort "
    "was approximately 1:2.98, reflecting the known clinical rate of ICI "
    "non-response in unselected NSCLC populations [3]. A secondary "
    "time-to-event endpoint, progression-free survival (PFS), was defined "
    "as time from ICI initiation to disease progression or death from any "
    "cause. Patients without a recorded event were censored at the date of "
    "last follow-up."
)
P(
    "This study was conducted in accordance with the Declaration of "
    "Helsinki and approved by the Institutional Review Board of "
    "[Institution] (Protocol No. [IRB Number]). All patients provided "
    "written informed consent."
)

add_figure(
    f"{FIGDIR}\\fig1_overview_demo.png",
    "**Figure 1.** Study overview. (A) The discovery cohort (n = 247) and "
    "two validation cohorts (radiomics-evaluable, n = 50; "
    "pathology-evaluable, n = 71) provide six data sources: CT radiomics "
    "(modelled separately per lesion site: PC, PL, LN), pathology image "
    "texture (IHC-A or IHC-G), tumour genomics, PD-L1 TPS, and structured "
    "clinical variables, giving up to seven input modalities per model. "
    "(B) The DyAM "
    "architecture computes a per-modality risk score and combines "
    "modalities using either cooperative (AttentionMatrix) or competitive "
    "one-vs-others (AttentionMatrixOvO) attention to produce a fused risk "
    "prediction ŷ. (C) Models are evaluated on both binary response "
    "classification (AUC-ROC with DeLong 95% CI) and survival endpoints "
    "(Harrell's C-index, multivariate Cox hazard ratios, and Kaplan–Meier "
    "log-rank stratification).",
)

H("2.2. Data Modalities", level=2)
P(
    "Six data sources were assembled for each patient (Table 2). Because "
    "CT radiomics is modelled separately per lesion site (PC, PL, LN) and "
    "the two pathology feature sets (IHC-A, IHC-G) are used as alternative "
    "arms rather than jointly, a single model receives up to seven input "
    "modalities. CT "
    "radiomics features were extracted from baseline scans using the "
    "MIRPON pipeline [7] at 1.0 mm isotropic spacing with a window of "
    "[−1350, 250] HU, yielding 1,688 features per lesion type. Lesions "
    "were categorised by anatomical site: primary tumour (PC), pleural "
    "lesion (PL), and lymph node (LN). Up to three lesions per site were "
    "included, ranked by volume. Pathology features were derived from "
    "PD-L1 immunohistochemistry (IHC) slides: *IHC-A* comprised 18 "
    "pixel-intensity aggregation features (area, mean intensity, "
    "percentiles), while *IHC-G* comprised 150 grey-level co-occurrence "
    "matrix (GLCM) texture features across six aggregation levels [11]. "
    "Genomic features (11 variables) included tumour mutational burden "
    "(TMB) and the status of eight driver mutations and amplifications "
    "derived from FoundationOne CDx next-generation sequencing (NGS). The "
    "PD-L1 tumour proportion score (TPS) was obtained from clinical "
    "pathology reports (range 0–100%). Clinical laboratory features "
    "comprised 13 variables: age, smoking pack-years, Eastern Cooperative "
    "Oncology Group (ECOG) performance status, serum albumin, derived "
    "neutrophil-to-lymphocyte ratio (dNLR), initial tumour burden, brain "
    "and liver metastasis status, line of therapy, combination therapy, "
    "primary lung site, prior PD-L1 therapy, and adenocarcinoma histology."
)
P_mix(
    "For radiomics modalities, features with an intraclass correlation "
    "coefficient (ICC) below 0.15 across test–retest perturbation scans "
    "were removed (robustness filter), and features with a z-score "
    "exceeding 6 in any patient were flagged as outliers and excluded. All "
    "modalities except the NLP embedding (Section 2.3) were standardised "
    "with RobustScaler (median centring, IQR scaling). Missing modalities "
    "were handled via a binary indicator mask ", IM(
        m.run("m ∈ "),
        m.sup(m.delim([m.run("0,1")], beg="{", end="}"), "N"),
    ), "; masked "
    "modalities contribute zero attention and zero risk to the fused "
    "prediction (Section 2.4)."
)

add_table(
    ["Modality", "Source", "Features", "Clinical meaning"],
    [
        ["CT Radiomics (PC, PL, LN)", "MIRPON pipeline", "≤1,688", "Tumour shape, size, and texture from baseline CT"],
        ["Pathology IHC-A", "PD-L1 IHC slides", "18", "Pixel-intensity aggregation of PD-L1 expression"],
        ["Pathology IHC-G", "PD-L1 IHC slides", "150", "GLCM texture of PD-L1 signal"],
        ["Genomics", "FoundationOne CDx", "11", "TMB + driver mutation/amplification status"],
        ["PD-L1 TPS", "Pathology report", "1", "Tumour proportion score (0–100%)"],
        ["Clinical labs", "EHR", "13", "Demographics, performance status, labs"],
        ["**NLP embedding (new)**", "all-MiniLM-L6-v2", "384", "Sentence embedding of clinical lab text"],
        ["**NLP-PCA16 (new)**", "PCA of NLP", "16", "PCA-compressed NLP (approx. 94% variance)"],
    ],
    caption=(
        "**Table 2.** Summary of data modalities used in this study. All "
        "feature counts are post-selection (after ICC and outlier "
        "filtering for radiomics)."
    ),
)

H("2.3. NLP Encoding of Tabular Clinical Features", level=2)
P(
    "Standard encoding of clinical variables as a 13-dimensional numeric "
    "vector treats each feature independently and loses contextual "
    "relationships between variables (e.g., an elderly patient with low "
    "albumin and ECOG 2 implies a different prognosis than the same values "
    "in isolation). To capture inter-feature context, we converted each "
    "patient's clinical profile into an English-language sentence and "
    "extracted a dense semantic embedding."
)
P(
    "**Text prompt construction.** For each patient, the 13 clinical "
    "variables were formatted into a structured natural-language sentence, "
    "for example: “The patient is a 69-year-old. Smoking history in "
    "pack-years is 48.0. ECOG performance status is 1.0. Blood albumin "
    "concentration is 4.00. Derived neutrophil-to-lymphocyte ratio (dNLR) "
    "is 2.40. Initial tumor burden stands at 52.00. The patient is "
    "diagnosed with adenocarcinoma at the lung site. Metastatic status: "
    "without brain metastasis and without liver metastasis. Currently "
    "undergoing therapy line 2, utilizing combination therapy. The patient "
    "received prior PD-L1 immunotherapy.” Binary variables were expanded "
    "to descriptive phrases to exploit the pretrained vocabulary of the "
    "encoder."
)
P(
    "**Sentence encoder.** Prompts were encoded with "
    "sentence-transformers/all-MiniLM-L6-v2 [12,13], a 6-layer transformer "
    "distilled for semantic sentence similarity and trained on over one "
    "billion sentence pairs. Each patient was represented as a "
    "384-dimensional L2-normalised vector. No fine-tuning was performed; "
    "the model was applied in zero-shot transfer. We compared this "
    "general-purpose encoder against the domain-specific Bio_ClinicalBERT "
    "[14]; the domain-specific model performed inferiorly (ΔAUC = −0.017 "
    "versus MiniLM-L6-v2 in the IHC-A arm), consistent with its design for "
    "masked language modelling rather than sentence-similarity tasks."
)
P(
    "**Dimensionality reduction.** Principal component analysis (PCA) was "
    "applied to the 384-dimensional embeddings, retaining the top 16 "
    "components (≈94% of variance explained). Both the raw 384-dimensional "
    "(*NLP raw*) and PCA-compressed 16-dimensional (*NLP-PCA16*) variants "
    "were evaluated. Crucially, the NLP modality was passed to the model "
    "with the RobustScaler disabled (no_scale=True), as the embeddings are "
    "already L2-normalised and scaling would distort the cosine-similarity "
    "geometry of the embedding space."
)

add_figure(
    f"{FIGDIR}\\fig2_nlp_pipeline_demo.png",
    "**Figure 2.** NLP-based encoding of structured clinical features. "
    "Thirteen numeric clinical variables per patient are converted into a "
    "natural-language sentence describing the patient's clinical profile "
    "(example shown), which is then encoded into a 384-dimensional "
    "embedding using a pretrained sentence transformer (all-MiniLM-L6-v2). "
    "The 384-dimensional embedding (“NLP raw”) is used directly, or "
    "further reduced to 16 dimensions via PCA (“NLP-PCA16”, retaining "
    "≈94% of variance) prior to fusion with other modalities. "
    "no_scale=True is applied to NLP embeddings to preserve their "
    "cosine-normalised geometry.",
)

H("2.4. Model Architecture", level=2)
P_mix(
    "**Dynamic Attention Multimodal (DyAM) model.** The base model, DyAM "
    "[11], fuses ", IM(m.run("N")), " modalities through per-modality "
    "risk scores and a learned attention mechanism, in the spirit of "
    "attention-based architectures broadly [15]. Let ",
    IM(m.sub("x", "i"), m.run(" ∈ "), m.sup("ℝ", [m.sub("d", "i")])),
    " denote the feature vector of modality ", IM(m.run("i")),
    " after optional RobustScaling. A scalar risk score is computed as:",
)
add_equation(
    m.sub("r", "i"), m.run(" = "),
    m.func("tanh", [m.subsup("W", "r,i", "T"), m.run(" "), m.sub("x", "i")]),
)
P_mix(
    "where ", IM(m.sub("W", "r,i"), m.run(" ∈ "), m.sup("ℝ", [m.sub("d", "i")])),
    " are learnable weights. An attention score is produced by an ",
    IM(m.run("N × N")), " grid of linear layers with softplus activation, ",
    IM(m.sub("ℓ", "2")), "-normalised by the modality vector norm:",
)
add_equation(
    m.sub("s", "i"), m.run(" = "),
    m.frac(
        [m.func("softplus", [m.subsup("W", "a,ij", "T"), m.run(" "), m.sub("x", "i")])],
        [m.sub(m.delim([m.sub("x", "i")], beg="‖", end="‖"), "2")],
    ),
)
P("Attention weights are obtained by L1 normalisation across modalities:")
add_equation(
    m.sub("a", "i"), m.run(" = "),
    m.frac([m.sub("s", "i")], [m.nary_sum("j = 1", "N", [m.sub("s", "j")])]),
)

P_mix(
    "**OvO competitive attention.** We propose an alternative attention "
    "mechanism inspired by one-versus-others (OvO) classification [16], in "
    "which each modality *competes* against the mean score of all "
    "remaining modalities rather than sharing a fixed budget. The mean "
    "score of the ", IM(m.run("N − 1")), " competing modalities is:",
)
add_equation(
    m.sub("μ", "−i"), m.run(" = "),
    m.frac(["1"], [m.run("N − 1")]),
    m.nary_sum("j ≠ i", None, [m.sub("s", "j")]),
)
P("and the OvO attention score is:")
add_equation(
    m.sub("ovo", "i"), m.run(" = σ"),
    m.delim([m.sub("s", "i"), m.run(" − "), m.sub("μ", "−i")]),
    m.run(",    σ(z) = "),
    m.sup(m.delim([m.run("1 + "), m.sup("e", "−z")]), "−1"),
)
P_mix(
    "The sigmoid function yields an independent probability that modality ",
    IM(m.run("i")),
    " outperforms the others, rather than a jointly normalised score. "
    "After applying the binary missingness mask ",
    IM(m.sub("m", "i"), m.run(" ∈ "),
       m.delim([m.run("0,1")], beg="{", end="}")),
    ", the weights are L1-normalised: ",
    IM(
        m.sub("a", "i"), m.run(" = "),
        m.frac(
            [m.sub("ovo", "i"), m.run(" · "), m.sub("m", "i")],
            [m.nary_sum("j", None, [m.sub("ovo", "j"), m.run(" · "), m.sub("m", "j")])],
        ),
    ),
    ".",
)
P(
    "**Fused prediction and training.** For both variants, the final risk "
    "prediction is:"
)
add_equation(
    m.acc("y"), m.run(" = "),
    m.nary_sum("i = 1", "N", [m.sub("r", "i"), m.run(" · "), m.sub("a", "i")]),
)

add_figure(
    f"{FIGDIR}\\fig_attention_arch_demo.png",
    [
        "**Figure 3.** Architecture of the two attention mechanisms. Both "
        "variants compute a per-modality risk score ",
        IM(m.sub("r", "i"), m.run(" = "),
           m.func("tanh", [m.subsup("W", "r,i", "T"), m.run(" "), m.sub("x", "i")])),
        " and produce the fused prediction ",
        IM(m.acc("y"), m.run(" = "),
           m.nary_sum("i", None, [m.sub("r", "i"), m.run(" · "), m.sub("a", "i")])),
        "; they differ only in how the attention weights ",
        IM(m.sub("a", "i")),
        " are obtained. (A) Cooperative attention (AttentionMatrix): an ",
        IM(m.run("N × N")),
        " grid of linear layers with softplus activation produces pairwise "
        "scores that are column-summed into ",
        IM(m.sub("s", "j")),
        " and L1-normalised so that the modalities share a fixed attention "
        "budget. (B) Competitive one-vs-others attention (AttentionMatrixOvO): "
        "each modality produces a single score ",
        IM(m.sub("s", "i")),
        " that is compared against the mean score of the remaining modalities ",
        IM(m.sub("μ", "−i")),
        " through a sigmoid, ",
        IM(m.run("σ"), m.delim([m.sub("s", "i"), m.run(" − "), m.sub("μ", "−i")])),
        ", yielding an independent competitive weight before L1 "
        "normalisation. Heatmap bars denote modality feature vectors; the "
        "diagram is drawn for ",
        IM(m.run("N = 4")),
        " modalities for clarity.",
    ],
)

P_mix(
    "The model was trained with binary cross-entropy loss "
    "(BCEWithLogitsLoss) with automatic inverse-frequency class weighting "
    "to address the ≈1:3 class imbalance, ",
    IM(m.sub("ℓ", "1")),
    " regularisation on attention weights (", IM(m.run("α = 1.0")),
    "), and ", IM(m.sub("ℓ", "2")), " weight decay (",
    IM(m.run("β = 1.0")),
    "). The Adam optimiser "
    "was used with a learning rate of 0.01 for 125 epochs. All experiments "
    "used 10-fold stratified cross-validation on the discovery cohort; "
    "predictions were pooled across held-out folds for AUC computation.",
)

H("2.5. Statistical Analysis and Evaluation", level=2)
P(
    "**Discriminative performance.** Area under the receiver operating "
    "characteristic curve (AUC-ROC) was computed on pooled out-of-fold "
    "predictions from 10-fold cross-validation. Ninety-five percent "
    "confidence intervals (CIs) were estimated using the DeLong structural "
    "components method [17]. Two models were considered statistically "
    "significantly different when their 95% CIs did not overlap, "
    "consistent with standard practice in oncology imaging studies."
)
P(
    "**Survival analysis.** Harrell's concordance index (C-index) [18] was "
    "computed from pooled out-of-fold risk scores against PFS. Bootstrap "
    "95% CIs were derived from 1,000 resamples with replacement. For "
    "pairwise comparisons, per-fold C-indices (10 values) were compared "
    "with a two-sided Wilcoxon signed-rank test (α = 0.05). Time-dependent "
    "AUC was computed at 6, 12, and 18 months using the cumulative/dynamic "
    "estimator [18]. Multivariate Cox proportional hazards regression was "
    "fitted with the DyAM risk score as the primary predictor, adjusted "
    "for five clinical covariates: age, ECOG performance status, serum "
    "albumin, dNLR, and liver metastasis status. Hazard ratios (HRs) are "
    "reported with 95% CIs and Wald p-values. Model calibration was "
    "assessed via the Integrated Brier Score (IBS); values below 0.25 "
    "indicate better-than-random calibration."
)
P(
    "**Kaplan–Meier survival stratification.** Patients were dichotomised "
    "at risk score = 0, and PFS curves for the two groups were compared "
    "with the log-rank test. A threshold of χ² > 7.88 (p < 0.005) was "
    "applied as the significance criterion, consistent with oncology "
    "reporting standards."
)
P(
    "**Software.** All analyses were performed in Python 3.10 using "
    "PyTorch 2.0 (model training), lifelines 0.27 (Kaplan–Meier, Cox "
    "regression, C-index), scikit-survival 0.21 (time-dependent AUC, Brier "
    "score), and scipy 1.11 (Wilcoxon test)."
)

print("methods done")

# ============================================================ 3. RESULTS =
H("3. Results", level=1)

H("3.1. Multimodal Fusion Outperforms Single-Modality Baselines", level=2)
P(
    "Within the DyAM framework, models restricted to a single data source "
    "achieved repeated-seed (5 seeds × 10-fold) AUC values of 0.616 ± "
    "0.001 (TMB), 0.635 ± 0.021 (pathology IHC), 0.673 ± 0.020 (CT "
    "radiomics), 0.676 ± 0.013 (genomics), and 0.720 ± 0.012 (PD-L1 TPS) "
    "on the discovery cohort. PD-L1 TPS was the strongest individual "
    "predictor, consistent with its role as the biomarker in routine "
    "clinical use."
)
P(
    "Fusing four data sources (CT radiomics, pathology IHC-G, genomics, "
    "and PD-L1 TPS) under the same protocol raised the AUC to 0.760 ± "
    "0.022, exceeding the best single source by 0.040 AUC; adding "
    "NLP-encoded clinical variables raised it further to 0.783 ± 0.016 "
    "(Section 3.3). Evaluated on the single cross-validation partition "
    "used for the DeLong confidence intervals reported in Tables 5 and 6, "
    "the same four-source configuration reached AUC = 0.784 [95% CI: "
    "0.717–0.850] (IHC-G arm; Table 6). Multimodal fusion is therefore the "
    "primary driver of predictive performance in this cohort, with no "
    "individual source approaching the discrimination of the fused model."
)

H(
    "3.2. OvO Competitive Attention Achieves Performance Comparable to "
    "Cooperative Attention with a Simpler Parameterization",
    level=2,
)
P(
    "In an initial exploratory screen across 20 modality configurations "
    "evaluated with a single 10-fold cross-validation partition, the OvO "
    "attention variant outperformed the original cooperative attention in "
    "7 of 20 cases (35%), while the original model was superior in 9 of 20 "
    "cases (45%), with 4 identical results (single-modality, where OvO "
    "reduces to the original). The mean AUC difference was −0.32% in "
    "favour of the original model overall. The largest single-partition "
    "gains for OvO were observed for PDL1+Gen (ΔAUC = +3.27%, AUC 0.716 "
    "vs. 0.693) and Rad+Gen (ΔAUC = +2.23%; Table 3), while the largest "
    "deficits occurred for configurations combining three or more "
    "modalities, particularly when IHC-G or clinical labs were included "
    "(largest deficit: Rad+IHC-A+Gen, ΔAUC = −4.94%)."
)
P(
    "Because a single CV partition can favour either model by chance, we "
    "repeated this comparison as a confirmatory analysis using 5 "
    "independent random partitions (5 seeds × 10-fold, patient labels "
    "re-shuffled per seed) across the same 21 modality combinations used "
    "in the primary paper-matched configuration set (Section 3.1), with "
    "paired bootstrap significance testing on the resulting patient-level "
    "scores. At the primary benchmark configuration (Rad+IHC-G+Gen+PDL1), "
    "OvO reached a repeated-seed mean AUC of 0.773 ± 0.020 versus 0.760 ± "
    "0.022 for the original cooperative model (ΔAUC = +0.009, 95% "
    "bootstrap CI [−0.008, 0.027], p = 0.313); across all four primary "
    "benchmarks the two models differed by at most 0.011 AUC (p ≥ 0.17). "
    "Extending the comparison to all 21 modality combinations (Table 4), "
    "OvO's repeated-seed mean AUC exceeded the original model's in 12 of "
    "21 configurations and was lower in 5 of 21, with 4 identical "
    "(single-modality, where the two formulations coincide algebraically)."
)
P(
    "This advantage was concentrated in the many-source regime: for "
    "configurations fusing k ≥ 5 input sources, OvO matched or exceeded "
    "cooperative attention in 7 of 8 cases (mean ΔAUC = +0.009, range "
    "−0.001 to +0.018), whereas for k = 2–4 it led in only 5 of 9 cases "
    "with a far wider spread (mean +0.003, range −0.028 to +0.045). "
    "Notably, this is the opposite of the pattern suggested by the "
    "single-partition screen, which had favoured OvO at low modality "
    "counts; the repeated-seed evaluation instead places OvO's small but "
    "consistent edge in precisely the multi-source configurations the "
    "fusion model is intended for. These per-k counts are descriptive: "
    "formal significance testing was performed only at the four primary "
    "benchmarks, and the effect sizes involved (≤ 0.018 AUC) remain below "
    "the seed-to-seed standard deviation (≈ 0.02), so the trend should be "
    "read as a consistency pattern rather than a demonstrated superiority."
)
P(
    "The single-partition “best-performing configuration” identified in "
    "the initial screen (OvO Rad+IHC-G+Gen+PDL1, AUC = 0.8003 [95% CI: "
    "0.739–0.862]) illustrates how favourable a single partition can be: "
    "the same configuration's repeated-seed mean AUC was 0.773, and OvO's "
    "rank relative to the cooperative model reversed once averaged across "
    "seeds."
)
P(
    "As a further robustness check, we combined OvO attention with the "
    "NLP-clinical encoding described in Section 3.3, repeating the same "
    "21-combination, 5-seed evaluation with clinical text embeddings added "
    "to each configuration. OvO+NLP reproduced the same fusion-dependent "
    "improvement pattern observed in the primary NLP-clinical analysis "
    "(Section 3.3): AUC improved relative to the no-clinical baseline in "
    "15 of 21 configurations (an identical set of configurations), and "
    "declined in the same six single- or near-single-modality "
    "configurations where any clinical addition diluted a strong "
    "standalone signal (PD-L1, radiomics-only, and similar). At the "
    "primary benchmark, OvO+NLP reached a repeated-seed mean AUC = 0.782 ± "
    "0.017, matching the 0.783 ± 0.016 obtained by the repeated-seed "
    "NLP-clinical confirmation at the same configuration (Section 3.3). "
    "Under the single-partition protocol, "
    "OvO+NLP reached AUC = 0.8133 at the Rad+IHC-A+Gen+TMB+PDL1 "
    "configuration, the highest value obtained under either evaluation "
    "protocol in this study."
)
P(
    "This shows that OvO's simpler, N-parameter formulation (independent "
    "per-modality scores compared against the mean of the others) delivers "
    "the same NLP-clinical benefit as the cooperative model without the "
    "N × N weight matrix, and that the clinical-encoding intervention, not "
    "the choice of attention mechanism, is the source of that gain. "
    "Table 3 summarises the exploratory single-partition screen and Table 4 "
    "gives the repeated-seed AUC of all three models across all 21 "
    "configurations; the corresponding single-partition values are provided "
    "in Supplementary Table S3."
)


add_table(
    ["Configuration", "Original AUC", "OvO AUC", "ΔAUC"],
    [
        ["PDL1+Gen", "0.693", "**0.716**", "+3.27%"],
        ["Rad+Gen", "0.738", "**0.755**", "+2.23%"],
        ["Rad+IHC-G+Gen+PDL1", "0.784", "**0.800**", "+2.10%"],
        ["Rad+IHC-A+Gen+PDL1", "0.764", "**0.777**", "+1.63%"],
        ["TMB+PDL1", "0.705", "**0.715**", "+1.42%"],
        ["Rad+IHC-A+Gen", "**0.757**", "0.719", "−4.94%"],
        ["IHC-G+Gen", "**0.756**", "0.723", "−4.33%"],
        ["Rad+IHC-A+Gen+PDL1+Labs", "**0.768**", "0.747", "−2.83%"],
        ["Rad+IHC-G+Gen", "**0.787**", "0.766", "−2.66%"],
        ["Rad+IHC-G+Gen+PDL1+Labs", "**0.788**", "0.783", "−0.56%"],
    ],
    caption=(
        "**Table 3.** Top configurations from the exploratory, "
        "single-partition screen comparing OvO competitive attention and "
        "original cooperative attention. ΔAUC is computed as OvO minus "
        "Original. As described in the text, these single-partition "
        "differences were not reproducible under repeated-seed evaluation "
        "(Table 4) and should be interpreted as an initial screen rather "
        "than a confirmed effect."
    ),
)

add_table(
    ["#", "Configuration", "k", "DyAM", "OvO", "OvO+NLP"],
    [
        ["1", "TMB", "1", "**0.616**", "**0.616**", "0.606"],
        ["2", "PDL1", "1", "**0.720**", "**0.720**", "0.694"],
        ["3", "IHC-A", "1", "**0.635**", "**0.635**", "0.625"],
        ["4", "Gen", "1", "**0.676**", "**0.676**", "0.661"],
        ["5", "Rad", "3", "**0.673**", "0.659", "0.651"],
        ["6", "Rad-LU", "4", "0.637", "**0.646**", "0.629"],
        ["7", "TMB+PDL1", "2", "0.664", "0.709", "**0.714**"],
        ["8", "PDL1+Gen (BM4)", "2", "0.707", "0.718", "**0.747**"],
        ["9", "Rad+IHC-A", "4", "0.671", "0.668", "**0.712**"],
        ["10", "Rad+IHC-G", "4", "0.706", "0.703", "**0.736**"],
        ["11", "Rad+Gen (BM3)", "4", "0.709", "**0.713**", "0.708"],
        ["12", "IHC-A+Gen", "2", "0.693", "0.697", "**0.702**"],
        ["13", "IHC-G+Gen", "2", "0.747", "0.719", "**0.752**"],
        ["14", "Rad+IHC-A+Gen", "5", "0.725", "0.735", "**0.744**"],
        ["15", "Rad+IHC-G+Gen", "5", "0.751", "0.752", "**0.760**"],
        ["16", "Rad+IHC-A+MutAmp", "6", "0.734", "0.744", "**0.747**"],
        ["17", "Rad+IHC-A+Gen+PDL1", "6", "0.743", "0.753", "**0.757**"],
        ["18", "Rad+IHC-G+Gen+PDL1 (BM1)", "6", "0.760", "0.773", "**0.782**"],
        ["19", "Rad+IHC-A+Gen+TMB+PDL1", "7", "0.749", "**0.766**", "0.765"],
        ["20", "Rad+IHC-A+Gen+PDL1+Labs", "7", "0.736", "0.745", "**0.757**"],
        ["21", "Rad+IHC-G+Gen+PDL1+Labs (BM2)", "7", "0.764", "0.763", "**0.782**"],
    ],
    caption=(
        "**Table 4.** Repeated-seed (5 seeds × 10-fold) AUC across all 21 "
        "paper-matched modality combinations, for the original cooperative "
        "attention (DyAM), OvO competitive attention, and OvO combined "
        "with NLP-clinical encoding. k is the number of fused input "
        "sources (CT radiomics contributes one source per lesion site). "
        "Bold marks the highest value in each row. For k = 1 OvO reduces "
        "algebraically to the cooperative formulation, hence the identical "
        "values. Configurations 8, 11, 18 and 21 are the primary "
        "benchmarks BM4, BM3, BM1 and BM2; paired-bootstrap testing (2000 "
        "resamples) was performed at those four only, giving ΔAUC "
        "(OvO − DyAM) of +0.011 [−0.005, 0.026], p = 0.172 (BM4); +0.000 "
        "[−0.012, 0.013], p = 0.980 (BM3); +0.009 [−0.008, 0.027], "
        "p = 0.313 (BM1); and −0.004 [−0.020, 0.009], p = 0.536 (BM2). "
        "Single-partition values for the same 21 configurations are given "
        "in Supplementary Table S3."
    ),
)

H(
    "3.3. NLP Encoding Consistently Improves AUC over Numeric Clinical "
    "Features",
    level=2,
)
P(
    "Seven NLP encoding variants were evaluated against numeric clinical "
    "labs as a reference. In the IHC-A arm, NLP-PCA16 achieved the highest "
    "AUC among clinical encoding strategies (AUC = 0.784, 95% CI: "
    "0.719–0.848; ΔAUC = +0.020 over no-clinical baseline), compared to "
    "+0.004 for numeric labs (Table 5). In the IHC-G arm, NLP raw "
    "(384-dimensional) achieved the best result (AUC = 0.813, 95% CI: "
    "0.753–0.874; ΔAUC = +0.030), with NLP-PCA16 performing comparably "
    "(ΔAUC = +0.028; Table 6)."
)
P(
    "However, all pairwise DeLong confidence intervals overlapped "
    "substantially, meaning no individual NLP variant reached statistical "
    "significance compared to the no-clinical baseline. The largest "
    "separation was observed in the IHC-G arm (NLP raw vs. no-clinical: "
    "lower CI 0.753 vs. upper CI 0.850; overlap ≈0.10). As discussed in "
    "the Discussion, this is attributable to the limited cohort size "
    "(n = 247) rather than an absent effect."
)
P(
    "*Repeated-seed confirmation.* To verify that the advantage of NLP "
    "encoding is not specific to a single CV partition, we repeated the "
    "NLP-versus-labs comparison under the repeated-seed protocol of "
    "Section 3.2 (5 seeds × 10-fold) across all 21 paper-matched modality "
    "combinations, using masked averaging of the per-modality risk scores "
    "as the fusion rule to isolate the encoding effect from attention "
    "learning. NLP encoding exceeded numeric labs in 19 of 21 "
    "combinations; the two exceptions (PD-L1 alone and the four-lesion "
    "radiomics-only configuration) are single-source configurations in "
    "which adding any clinical modality dilutes a strong standalone "
    "signal. At the primary IHC-G-arm configuration (Rad+IHC-G+Gen+PDL1), "
    "repeated-seed mean AUC was 0.766 ± 0.013 with numeric labs versus "
    "0.783 ± 0.016 with NLP encoding (Δ = +0.017); the corresponding "
    "IHC-A-arm values were 0.742 ± 0.014 versus 0.758 ± 0.013 "
    "(Δ = +0.016). The same NLP gain also held under OvO attention "
    "(Section 3.2), indicating the improvement is robust to both the CV "
    "partition and the fusion mechanism."
)
P(
    "*Domain-specific versus general-purpose encoder.* Bio_ClinicalBERT "
    "[14], PCA-compressed to 16 dimensions, performed below MiniLM-L6-v2 "
    "in the IHC-A arm (AUC = 0.767 versus 0.784; ΔAUC = −0.017), "
    "consistent with its pretraining objective of masked language "
    "modelling on clinical notes rather than semantic sentence similarity."
)
P_mix(
    "*Combined encoding.* Concatenating numeric labs and NLP-PCA16 as "
    "separate modalities (8 modalities total) degraded performance "
    "(AUC = 0.753, Δ = −0.011 in IHC-A), likely due to unstable attention "
    "learning with ",
    IM(m.frac(["n"], ["N"]), m.run(" = "), m.frac(["247"], ["8"]),
       m.run(" ≈ 31")),
    ". NLP encoding is therefore recommended as a *replacement* for, "
    "rather than an addition to, numeric clinical features.",
)

add_table(
    ["Model", "AUC", "95% CI", "ΔAUC"],
    [
        ["No clinical (baseline)", "0.764", "[0.695–0.833]", "(ref)"],
        ["+ Labs (13-dim numeric)", "0.768", "[0.700–0.837]", "+0.004"],
        ["+ NLP raw (384-dim)", "0.781", "[0.717–0.845]", "+0.017"],
        ["**+ NLP-PCA16 (16-dim)**", "**0.784**", "**[0.719–0.848]**", "**+0.020**"],
        ["+ Labs + NLP (combined)", "0.767", "[0.700–0.835]", "+0.003"],
        ["+ Labs + NLP-PCA16 (combined)", "0.753", "[0.682–0.823]", "−0.011"],
        ["+ BioClinBERT-PCA16", "0.767", "[0.701–0.833]", "+0.003"],
    ],
    caption=(
        "**Table 5.** AUC comparison across clinical encoding strategies "
        "— IHC-A arm. All models use the DyAM architecture with CT "
        "radiomics (PC, PL, LN), pathology IHC-A, genomics, and PD-L1 TPS "
        "as base modalities; clinical encoding is the varied factor. AUC "
        "computed on pooled out-of-fold predictions from 10-fold "
        "cross-validation (n = 247). 95% CI by DeLong structural "
        "components method."
    ),
)

add_table(
    ["Model", "AUC", "95% CI", "ΔAUC"],
    [
        ["No clinical (baseline)", "0.784", "[0.717–0.850]", "(ref)"],
        ["+ Labs (13-dim numeric)", "0.788", "[0.723–0.853]", "+0.004"],
        ["**+ NLP raw (384-dim)**", "**0.813**", "**[0.753–0.874]**", "**+0.030**"],
        ["+ NLP-PCA16 (16-dim)", "0.812", "[0.751–0.873]", "+0.028"],
    ],
    caption=(
        "**Table 6.** AUC comparison across clinical encoding strategies "
        "— IHC-G arm. Base modalities: CT radiomics (PC, PL, LN), "
        "pathology IHC-G (GLCM texture), genomics, and PD-L1 TPS. All "
        "other details as in Table 5."
    ),
)

add_figure(
    f"{FIGDIR_CONV}\\fig3a_auc_ihca_demo.png",
    "**Figure 4A.** AUC comparison across clinical encoding strategies, "
    "IHC-A arm (7 variants). Bars show pooled out-of-fold AUC from "
    "10-fold cross-validation (n = 247); error bars denote DeLong 95% "
    "confidence intervals. The dashed horizontal line at AUC = 0.80 "
    "indicates the conventional threshold for clinically useful "
    "discrimination.",
    width_in=5.0,
)
add_figure(
    f"{FIGDIR_CONV}\\fig3b_auc_ihcg_demo.png",
    "**Figure 4B.** AUC comparison across clinical encoding strategies, "
    "IHC-G arm (4 variants). Same conventions as Figure 4A. "
    "Best-performing model overall: DyAM Rad+IHC-G+Gen+PDL1+NLP raw, "
    "AUC = 0.813.",
    width_in=5.0,
)

H(
    "3.4. Survival Analysis: NLP Encoding Enhances Independent Prognostic "
    "Value",
    level=2,
)
P(
    "The DyAM risk score was a statistically significant independent "
    "prognostic factor for PFS across all eight model variants in "
    "multivariate Cox regression (all p < 0.001, adjusted for age, ECOG "
    "performance status, serum albumin, dNLR, and liver metastasis "
    "status; Table 7)."
)
P(
    "**Cox hazard ratios.** NLP-PCA16 yielded the highest hazard ratio in "
    "the IHC-A arm (HR = 6.07, 95% CI: 3.50–10.52), compared with HR = "
    "5.30 [3.10–9.07] for the no-clinical baseline and HR = 5.31 "
    "[2.88–9.79] for numeric labs. In the IHC-G arm, NLP raw achieved "
    "HR = 4.97 [2.98–8.31] versus HR = 4.74 [2.82–7.97] for no-clinical "
    "and HR = 4.49 [2.56–7.89] for numeric labs. Notably, numeric labs did "
    "not improve the hazard ratio in either arm, whereas NLP encoding "
    "consistently increased it, suggesting that sentence-embedded "
    "clinical context captures prognostic information not represented by "
    "the five Cox covariates or by raw numeric features."
)
P(
    "**Time-dependent AUC.** The advantage of NLP encoding was "
    "time-dependent: in the IHC-A arm, NLP-PCA16 improved time-dependent "
    "AUC by +0.024 at 12 months and +0.020 at 18 months relative to the "
    "no-clinical baseline, with smaller differences at 6 months (+0.002). "
    "This pattern, concentrated at late timepoints, is consistent with the "
    "hypothesis that sentence embeddings primarily encode long-term "
    "survival characteristics rather than short-term binary response."
)
P(
    "**C-index.** Harrell's C-index showed a consistent but modest "
    "improvement with NLP variants (ΔC = +0.005 in both IHC arms; "
    "Wilcoxon p = 0.492 and p = 0.375 for IHC-A and IHC-G, respectively), "
    "which did not reach statistical significance. The high inter-fold "
    "variance (range 0.52–0.78 across 10 folds) and the small fold size "
    "(≈25 patients per fold) explain the non-significant result "
    "independently of the true effect size."
)
P(
    "**Calibration.** All models were well-calibrated, with Integrated "
    "Brier Scores of 0.172–0.175, well below the 0.25 random-classifier "
    "baseline. NLP-PCA16 in the IHC-A arm achieved the lowest IBS "
    "(0.1716), indicating marginally superior probabilistic calibration."
)
P(
    "**Kaplan–Meier stratification.** Dichotomising patients at risk "
    "score = 0 produced statistically significant separation between "
    "high- and low-risk groups for all eight models (p < 0.005 by "
    "log-rank test; Table 8), with the highest χ² statistic observed for "
    "the IHC-G arm with NLP raw clinical encoding (χ² = 28.92)."
)

add_table(
    ["Model", "C-index [95% CI]", "Cox HR [95% CI]", "IBS"],
    [
        ["IHC-A arm", "", "", ""],
        ["No clinical", "0.623 [0.580–0.665]", "5.30 [3.10–9.07]", "0.1745"],
        ["+ Labs", "0.626 [0.583–0.669]", "5.31 [2.88–9.79]", "0.1739"],
        ["+ NLP raw", "0.625 [0.583–0.663]", "5.91 [3.40–10.25]", "0.1723"],
        ["**+ NLP-PCA16**", "**0.628 [0.584–0.668]**", "**6.07 [3.50–10.52]**", "**0.1716**"],
        ["IHC-G arm", "", "", ""],
        ["No clinical", "0.626 [0.583–0.665]", "4.74 [2.82–7.97]", "0.1748"],
        ["+ Labs", "**0.632 [0.587–0.675]**", "4.49 [2.56–7.89]", "0.1746"],
        ["**+ NLP raw**", "**0.632 [0.591–0.672]**", "**4.97 [2.98–8.31]**", "**0.1736**"],
        ["+ NLP-PCA16", "0.631 [0.590–0.672]", "4.69 [2.86–7.68]", "0.1736"],
    ],
    caption=(
        "**Table 7.** Survival analysis results across all model "
        "variants. Discovery cohort (n = 247; 209 events, 84.6%; median "
        "PFS 2.7 months). C-index: Harrell's concordance index with "
        "bootstrap 95% CI (1,000 resamples). Cox HR: hazard ratio from "
        "multivariate Cox proportional hazards regression adjusted for "
        "age, ECOG, albumin, dNLR, and liver metastases (all Cox "
        "p < 0.001). IBS: Integrated Brier Score (lower is better; "
        "< 0.25 indicates better-than-random calibration). Bold: best "
        "value per column within each IHC arm."
    ),
)

add_table(
    ["Model", "IHC arm", "χ²", "p-value"],
    [
        ["DyAM No clinical", "IHC-A", "28.17", "< 0.005"],
        ["DyAM + NLP-PCA16", "IHC-A", "23.74", "< 0.005"],
        ["DyAM No clinical", "IHC-G", "20.07", "< 0.005"],
        ["**DyAM + NLP-PCA16**", "**IHC-G**", "**28.92***", "**< 0.005**"],
    ],
    caption=(
        "**Table 8.** Kaplan–Meier log-rank statistics for survival "
        "stratification. Patients were dichotomised at risk score = 0. "
        "Threshold for significance: χ² > 7.88 (p < 0.005). All models "
        "achieve p < 0.005. * Highest χ² across all tested models."
    ),
)

add_figure(
    f"{FIGDIR_CONV}\\fig4a_km_ihca_noclin.png",
    "**Figure 5A.** Kaplan–Meier PFS curves, IHC-A arm, no clinical "
    "encoding (χ² = 28.17).", width_in=4.2,
)
add_figure(
    f"{FIGDIR_CONV}\\fig4b_km_ihca_nlppca16.png",
    "**Figure 5B.** Kaplan–Meier PFS curves, IHC-A arm, + NLP-PCA16 "
    "clinical encoding (χ² = 23.74).", width_in=4.2,
)
add_figure(
    f"{FIGDIR_CONV}\\fig4c_km_ihcg_noclin.png",
    "**Figure 5C.** Kaplan–Meier PFS curves, IHC-G arm, no clinical "
    "encoding (χ² = 20.07).", width_in=4.2,
)
add_figure(
    f"{FIGDIR_CONV}\\fig4d_km_ihcg_nlpraw.png",
    "**Figure 5D.** Kaplan–Meier PFS curves, IHC-G arm, + NLP raw "
    "clinical encoding (χ² = 28.92, the highest log-rank statistic among "
    "all tested models). Panels A–D: patients stratified at risk "
    "score = 0 into high-risk (red) and low-risk (blue) groups, with "
    "number-at-risk tables. All comparisons reach p < 0.005 by the "
    "log-rank test.", width_in=4.2,
)

add_figure(
    f"{FIGDIR_CONV}\\fig5a_cindex.png",
    "**Figure 6A.** Harrell's C-index (bootstrap 95% CI) for IHC-A and "
    "IHC-G arms; differences between encoding strategies were not "
    "statistically significant (paired Wilcoxon p > 0.05).", width_in=4.2,
)
add_figure(
    f"{FIGDIR_CONV}\\fig5b_tdauc.png",
    "**Figure 6B.** Time-dependent AUC at 6, 12, and 18 months; "
    "NLP-encoded models show the largest advantage at 12–18 months.",
    width_in=4.2,
)
add_figure(
    f"{FIGDIR_CONV}\\fig5c_cox_forest.png",
    "**Figure 6C.** Forest plot of multivariate Cox proportional hazards "
    "ratios for the fused risk score, adjusted for age, ECOG performance "
    "status, albumin, derived neutrophil-to-lymphocyte ratio, and liver "
    "metastases (all p < 0.001).", width_in=4.2,
)
add_figure(
    f"{FIGDIR_CONV}\\fig5d_ibs.png",
    "**Figure 6D.** Integrated Brier Score (IBS) for each model; all "
    "values fall well below the 0.25 random-guess threshold (range "
    "0.172–0.175), indicating good calibration across all encoding "
    "strategies.", width_in=4.2,
)

print("results done")

# ========================================================= 4. DISCUSSION =
H("4. Discussion", level=1)

P(
    "In this study, we extended the DyAM multimodal attention framework "
    "with two complementary contributions: a competitive (OvO) attention "
    "mechanism and an NLP-based encoding of tabular clinical features, and "
    "evaluated both using binary classification and survival endpoints."
)
P(
    "The best-performing model, DyAM Rad+IHC-G+Gen+PDL1+NLP raw, achieved "
    "AUC = 0.813 [95% CI: 0.753–0.874], placing it at the "
    "“good-to-excellent” discrimination threshold (AUC > 0.80) commonly "
    "used in oncology decision support [11]. For context, this exceeds "
    "the single-source models evaluated in the same cohort under the DyAM "
    "framework — PD-L1 TPS alone (AUC = 0.720) and TMB alone "
    "(AUC = 0.616) — consistent with the modest discrimination reported "
    "for these markers in the literature [4,2]. This "
    "confirms that multimodal fusion, augmented with NLP-encoded clinical "
    "context, approaches the upper range of discrimination achievable with "
    "currently available biomarkers in this disease setting."
)
P(
    "Why does NLP encoding outperform raw numeric clinical features? We "
    "propose that sentence embeddings capture non-linear interactions "
    "between clinical variables — for example, the combined prognostic "
    "implication of advanced age, low albumin, and poor performance status "
    "— that are not explicitly represented in a 13-dimensional numeric "
    "vector. The MiniLM-L6-v2 embedding space [12,13] is pretrained to "
    "place semantically similar sentences close together; patients with "
    "similar overall clinical profiles are therefore mapped to nearby "
    "embedding vectors even when no single numeric feature dominates. Raw "
    "numeric encoding, by contrast, presents each variable to the "
    "attention mechanism independently, requiring the model to learn these "
    "interactions de novo from a relatively small training set."
)
P(
    "The AUC improvements associated with NLP encoding (+0.017 to +0.030) "
    "did not reach statistical significance under DeLong confidence "
    "interval testing, as all CIs overlapped substantially. This is best "
    "interpreted as a power limitation rather than an absent effect. With "
    "n = 247, the 95% CI width for AUC is approximately ±0.07; detecting a "
    "true ΔAUC ≈ 0.02 at 80% power would require approximately 950–1,250 "
    "patients (Supplementary Note S3). The directional consistency of the improvement across both "
    "the IHC-A and IHC-G arms, and across raw and PCA-compressed NLP "
    "variants, supports a true but modest effect size that is currently "
    "underpowered rather than null."
)
P(
    "The survival analysis provides an important reframing of this "
    "finding. While binary AUC improvements were modest and "
    "non-significant, multivariate Cox regression showed that NLP-PCA16 "
    "achieved the highest hazard ratio in the IHC-A arm (HR = 6.07 vs. "
    "HR = 5.30 for the no-clinical baseline), and NLP raw achieved the "
    "highest hazard ratio in the IHC-G arm (HR = 4.97 vs. HR = 4.74), "
    "whereas numeric clinical labs provided no comparable improvement in "
    "either arm. Furthermore, the time-dependent AUC advantage of NLP "
    "encoding was concentrated at 12 and 18 months (+0.024 and +0.020 in "
    "the IHC-A arm) rather than at 6 months. Taken together, these results "
    "suggest that NLP encoding of tabular clinical features predominantly "
    "captures *prognostic (time-to-event)* information rather than "
    "*predictive (binary response)* information — a distinction with "
    "direct implications for how such encodings should be evaluated in "
    "future multimodal studies."
)
P(
    "The OvO competitive attention mechanism matched the accuracy of the "
    "original cooperative attention while using a substantially simpler "
    "parameterization. An initial single-partition screen suggested a "
    "context-dependent pattern, with OvO favoured in low-modality "
    "configurations (up to +3.27% for PDL1+Gen) and the original model "
    "favoured as more modalities were added; the overall best-performing "
    "single-partition configuration (Rad+IHC-G+Gen+PDL1, AUC = 0.8003) "
    "used OvO attention. A confirmatory repeated-seed analysis (5 seeds × "
    "10-fold) placed the two attention mechanisms within 0.011 AUC of each "
    "other at all four primary benchmarks (p ≥ 0.17), with OvO reaching a "
    "repeated-seed mean AUC of 0.773 versus 0.760 for the cooperative "
    "model at the primary configuration — directionally favouring OvO, "
    "though the single-partition “best” configuration's rank relative to "
    "the cooperative model reversed once averaged across seeds, "
    "underscoring the value of repeated-seed evaluation over a single "
    "split."
)
P(
    "Across all 21 configurations the repeated-seed data also reversed the "
    "*direction* of the context-dependence seen in the screen: OvO matched "
    "or exceeded cooperative attention in 7 of 8 configurations fusing "
    "five or more sources, but in only 5 of 9 configurations with two to "
    "four sources, where the spread between the two mechanisms was several "
    "times larger. A competitive one-vs-others weighting is plausibly "
    "better conditioned when many partially redundant sources compete, "
    "whereas with few sources the comparison against the mean of the "
    "remaining modalities is estimated from one or two values and becomes "
    "unstable; we note, however, that the effect sizes involved "
    "(≤ 0.018 AUC) sit below the seed-to-seed variability, so this remains "
    "a hypothesis generated by the data rather than a tested claim."
)
P(
    "When combined with NLP-clinical encoding, OvO reproduced the "
    "identical fusion-dependent improvement pattern seen in the primary "
    "NLP-clinical analysis (improvement in the same 15 of 21 "
    "configurations, decline in the same six single-modality "
    "configurations), reaching "
    "AUC = 0.782 at the primary benchmark under the repeated-seed protocol "
    "(repeated-seed NLP-clinical confirmation: 0.783) and "
    "AUC = 0.8133 under the single-partition "
    "protocol at the Rad+IHC-A+Gen+TMB+PDL1 configuration — the highest "
    "value observed under either protocol in this study. OvO's practical "
    "value lies in its simpler parameterization — N independent "
    "per-modality scores compared against the mean of the remaining "
    "modalities, rather than the N × N weight matrix required by "
    "cooperative attention — delivering matching accuracy, and its "
    "improvement pattern under NLP-clinical encoding generalises across "
    "both attention mechanisms, indicating that the clinical-encoding "
    "intervention, not the choice of attention mechanism, is the source "
    "of the observed gain."
)
P(
    "From a clinical implementation perspective, NLP encoding of "
    "structured clinical data offers practical advantages: encoding a "
    "patient profile requires less than one second on CPU, requires no "
    "labelled clinical text, and can be applied to any structured "
    "electronic health record (EHR) field that can be rendered as a "
    "sentence. This makes the approach readily transferable to other "
    "structured clinical datasets without requiring a fine-tuned clinical "
    "language model."
)
P(
    "Several limitations should be considered. First, this was a "
    "single-institution retrospective cohort, which may limit "
    "generalisability and introduce selection bias. Second, the "
    "non-significant AUC improvement reflects the modest sample size "
    "(n = 247) rather than a definitive negative result, and larger "
    "multi-institutional cohorts are needed to confirm the observed "
    "trends. Third, the sentence encoder was applied in a zero-shot manner "
    "without fine-tuning on NSCLC-specific clinical text, which may limit "
    "its sensitivity to domain-specific terminology. Finally, PD-L1 TPS "
    "was scored using the Sauter method in this cohort, which may differ "
    "from the SP142 or 22C3 assays used at other institutions, and the "
    "generalisability of PD-L1-derived features should be interpreted "
    "with this in mind."
)

# ========================================================= 5. CONCLUSIONS =
H("5. Conclusions", level=1)
P(
    "We extended the DyAM multimodal attention framework for ICI response "
    "prediction in NSCLC with a competitive (OvO) attention mechanism and "
    "an NLP-based encoding of structured clinical variables. Under a "
    "repeated-seed evaluation protocol, OvO attention matched the original "
    "cooperative attention at every primary benchmark configuration "
    "(within 0.011 AUC, p ≥ 0.17; AUC = 0.773 vs. 0.760 at the primary "
    "benchmark), offering a simpler, N-parameter alternative to the N × N "
    "cooperative weight matrix without a measurable accuracy cost; an "
    "initial single-partition screen had suggested a larger OvO advantage "
    "in low-modality configurations (best single-partition AUC = 0.800, "
    "later 0.8133 when combined with NLP-clinical encoding), and averaging "
    "across five independent CV partitions confirmed OvO's accuracy while "
    "showing this particular single-partition margin does not reproduce "
    "exactly."
)
P(
    "NLP encoding of clinical features using a general-purpose sentence "
    "transformer consistently improved AUC over raw numeric encoding (up "
    "to ΔAUC = +0.030) and provided the strongest independent prognostic "
    "value in survival analysis, with the highest Cox hazard ratios and "
    "time-dependent AUC among all clinical encoding strategies, despite "
    "AUC differences not reaching statistical significance at n = 247. "
    "This improvement was confirmed under repeated-seed evaluation (NLP "
    "encoding exceeded numeric labs in 19 of 21 modality combinations) "
    "and generalised across fusion mechanisms: combining NLP-clinical "
    "encoding with OvO attention reproduced the same fusion-dependent "
    "gain, indicating that the clinical-encoding strategy, rather than "
    "the choice of attention mechanism, drives the improvement."
)
P(
    "These findings suggest that sentence-embedding-based representations "
    "of tabular clinical data are a practical and generalisable strategy "
    "for multimodal biomarker discovery, robust to the underlying "
    "attention architecture. Validation in larger, multi-institutional "
    "cohorts is needed to confirm these findings and to establish their "
    "clinical utility for treatment selection."
)

# ======================================================== DECLARATIONS ===
H("Declarations", level=1)
P("**Author Contributions:** Conceptualization, F.A. and T.A.; methodology, "
  "F.A. and S.A.; software, F.A.; formal analysis, F.A.; writing—original "
  "draft preparation, F.A.; writing—review and editing, S.A. and T.A.; "
  "supervision, T.A. All authors have read and agreed to the published "
  "version of the manuscript.")
P("**Funding:** [Funding — e.g., This research received no external "
  "funding.]")
P("**Institutional Review Board Statement:** This study was approved by "
  "the Institutional Review Board of [Institution] (Protocol No. "
  "[IRB Number]).")
P("**Informed Consent Statement:** Informed consent was obtained from all "
  "subjects involved in the study.")
P("**Data Availability Statement:** The code used in this study is "
  "available at https://github.com/[repo]. Patient-level data are "
  "subject to institutional data-sharing agreements and available upon "
  "reasonable request.")
P("**Conflicts of Interest:** The authors declare no conflicts of "
  "interest.")

print("discussion/conclusion/declarations done")

# =========================================================== REFERENCES ===
H("References", level=1)

REFERENCES = [
    "Siegel RL, Giaquinto AN, Jemal A. Cancer Statistics, 2024. CA Cancer "
    "J Clin. 2024;74:12–49. doi:10.3322/caac.21820",
    "Reck M, Rodríguez-Abreu D, Robinson AG, et al. Pembrolizumab versus "
    "Chemotherapy for PD-L1-Positive Non-Small-Cell Lung Cancer. N Engl J "
    "Med. 2016;375:1823–1833. doi:10.1056/NEJMoa1606774",
    "Borghaei H, Paz-Ares L, Horn L, et al. Nivolumab versus Docetaxel in "
    "Advanced Nonsquamous Non-Small-Cell Lung Cancer. N Engl J Med. "
    "2015;373:1627–1639. doi:10.1056/NEJMoa1507643",
    "Rizvi NA, Hellmann MD, Snyder A, et al. Mutational landscape "
    "determines sensitivity to PD-1 blockade in non-small cell lung "
    "cancer. Science. 2015;348:124–128. doi:10.1126/science.aaa1348",
    "Herbst RS, Baas P, Kim DW, et al. Pembrolizumab versus docetaxel for "
    "previously treated, PD-L1-positive, advanced non-small-cell lung "
    "cancer (KEYNOTE-010). Lancet. 2016;387:1540–1550. "
    "doi:10.1016/S0140-6736(15)01281-7",
    "Bodalal Z, Trebeschi S, Nguyen-Kim TDL, et al. Radiogenomics: "
    "bridging imaging and genomics. Abdom Radiol. 2019;44:1960–1984. "
    "doi:10.1007/s00261-019-02028-w",
    "Dercle L, Fronheiser M, Lu L, et al. Identification of Non-Small "
    "Cell Lung Cancer Sensitive to Systemic Cancer Therapies Using "
    "Radiomics. Clin Cancer Res. 2020;26:2151–2162. "
    "doi:10.1158/1078-0432.CCR-19-2942",
    "Trebeschi S, Drago SG, Birkbak NJ, et al. Predicting Response to "
    "Cancer Immunotherapy Using Noninvasive Radiomic Biomarkers. Ann "
    "Oncol. 2019;30:998–1004. doi:10.1093/annonc/mdz108",
    "Cheerla A, Gevaert O. Deep Learning with Multimodal Representation "
    "for Pancancer Prognosis Prediction. Bioinformatics. "
    "2019;35:i446–i454. doi:10.1093/bioinformatics/btz342",
    "Ma M, Ren J, Zhao L, et al. SMIL: Multimodal Learning with Severely "
    "Missing Modality. Proc AAAI Conf Artif Intell. 2021;35(3):2302–2310. "
    "doi:10.1609/aaai.v35i3.16330",
    "Vanguri RS, Luo J, Aukerman AT, et al. Multimodal integration of "
    "radiology, pathology and genomics for prediction of response to "
    "PD-(L)1 blockade in patients with non-small cell lung cancer. Nat "
    "Cancer. 2022;3(10):1151–1164. doi:10.1038/s43018-022-00416-8",
    "Reimers N, Gurevych I. Sentence-BERT: Sentence Embeddings using "
    "Siamese BERT-Networks. Proc 2019 Conf Empir Methods Nat Lang Process "
    "(EMNLP). 2019:3982–3992. doi:10.18653/v1/D19-1410",
    "Wang W, Wei F, Dong L, et al. MiniLM: Deep Self-Attention "
    "Distillation for Task-Agnostic Compression of Pre-Trained "
    "Transformers. Adv Neural Inf Process Syst (NeurIPS). "
    "2020;33:5776–5788.",
    "Alsentzer E, Murphy JR, Boag W, et al. Publicly Available Clinical "
    "BERT Embeddings. Proc 2nd Clinical Nat Lang Process Workshop. "
    "2019:72–78. doi:10.18653/v1/W19-1909",
    "Vaswani A, Shazeer N, Parmar N, et al. Attention Is All You Need. "
    "Adv Neural Inf Process Syst (NeurIPS). 2017;30.",
    "Hastie T, Tibshirani R. Classification by Pairwise Coupling. Ann "
    "Stat. 1998;26:451–471. doi:10.1214/aos/1028144844",
    "DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the Areas under "
    "Two or More Correlated Receiver Operating Characteristic Curves: A "
    "Nonparametric Approach. Biometrics. 1988;44:837–845. "
    "doi:10.2307/2531595",
    "Harrell FE, Lee KL, Mark DB. Multivariable Prognostic Models: Issues "
    "in Developing Models, Evaluating Assumptions and Adequacy, and "
    "Measuring and Reducing Errors. Stat Med. 1996;15:361–387. "
    "doi:10.1002/(SICI)1097-0258(19960229)15:4<361::AID-SIM168>3.0.CO;2-4",
]

for i, ref in enumerate(REFERENCES, 1):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.75)
    p.paragraph_format.first_line_indent = Cm(-0.75)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(f"{i}.\t")
    add_markup(p, ref)
    for run in p.runs:
        run.font.size = Pt(10)

print("references done")
# Lưu; nếu file đích đang mở trong Word (bị khoá) thì fallback _v2, _v3...
_target = Path(OUT_PATH)
for attempt in range(10):
    try:
        doc.save(str(_target))
        print("SAVED:", _target)
        break
    except PermissionError:
        _target = _target.with_name(
            f"{Path(OUT_PATH).stem}_v{attempt + 2}{Path(OUT_PATH).suffix}")
else:
    raise RuntimeError("Không lưu được: mọi tên file đều bị khoá.")
