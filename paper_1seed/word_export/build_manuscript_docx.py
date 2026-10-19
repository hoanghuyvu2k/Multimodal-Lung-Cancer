import os
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
   Output: paper_1seed/manuscript_word_1seed_YYYYMMDD_HHMM.docx (nếu file đang mở trong Word sẽ bị
   khoá — script tự fallback sang manuscript_word_1seed_v2.docx, v3, ...).
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
import datetime as _dt, shutil as _sh
_TAG = "1seed"
# every build gets a new name with date and time; earlier versions move to word_archive/
OUT_PATH = str(PAPER_DIR / f"manuscript_word_{_TAG}_{_dt.datetime.now():%Y%m%d_%H%M}.docx")
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
            _add_text_with_cites(paragraph, chunk)


def _add_ref_link(paragraph, num):
    """Internal hyperlink to the bookmark of reference <num> (Ctrl+click in Word)."""
    h = OxmlElement("w:hyperlink")
    h.set(qn("w:anchor"), f"ref_{num}")
    h.set(qn("w:history"), "1")
    r = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    col = OxmlElement("w:color")
    col.set(qn("w:val"), "0563C1")
    rpr.append(col)
    r.append(rpr)
    t = OxmlElement("w:t")
    t.text = str(num)
    r.append(t)
    h.append(r)
    paragraph._p.append(h)


CITE_RE = re.compile(r"\[(\d+(?:\s*[,\u2013-]\s*\d+)*)\]")


def _add_text_with_cites(paragraph, chunk):
    pos = 0
    for m in CITE_RE.finditer(chunk):
        if m.start() > pos:
            paragraph.add_run(chunk[pos:m.start()])
        paragraph.add_run("[")
        for part in re.split(r"(\s*[,\u2013-]\s*)", m.group(1)):
            if part.strip().isdigit():
                _add_ref_link(paragraph, int(part))
            elif part:
                paragraph.add_run(part)
        paragraph.add_run("]")
        pos = m.end()
    if pos < len(chunk):
        paragraph.add_run(chunk[pos:])


def _add_bookmark(paragraph, name, bid):
    s = OxmlElement("w:bookmarkStart")
    s.set(qn("w:id"), str(bid))
    s.set(qn("w:name"), name)
    e = OxmlElement("w:bookmarkEnd")
    e.set(qn("w:id"), str(bid))
    paragraph._p.insert(0, s) if paragraph._p.pPr is None else paragraph._p.pPr.addnext(s)
    s.addnext(e)


_EXTRA_URL = {
    "MiniLM: Deep Self-Attention": "https://arxiv.org/abs/2002.10957",
    "Attention Is All You Need": "https://arxiv.org/abs/1706.03762",
    "In Defense of One-Vs-All": "https://www.jmlr.org/papers/v5/rifkin04a.html",
}


def _add_external_link(paragraph, url, text):
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    rid = paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True)
    h = OxmlElement("w:hyperlink")
    h.set(qn("r:id"), rid)
    h.set(qn("w:history"), "1")
    r = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    col = OxmlElement("w:color")
    col.set(qn("w:val"), "0563C1")
    u = OxmlElement("w:u")
    u.set(qn("w:val"), "single")
    rpr.append(col)
    rpr.append(u)
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), "20")
    rpr.append(sz)
    r.append(rpr)
    t = OxmlElement("w:t")
    t.text = text
    r.append(t)
    h.append(r)
    paragraph._p.append(h)


def _add_ref_text_online(paragraph, ref):
    """Reference text; the DOI (or a known URL) becomes a clickable online link."""
    from urllib.parse import quote
    m = re.search(r"doi:(\S+)$", ref.strip())
    if m:
        add_markup(paragraph, ref[:ref.rindex("doi:")])
        doi = m.group(1)
        _add_external_link(paragraph, "https://doi.org/" + quote(doi, safe="/:()-._;"), "doi:" + doi)
        return
    add_markup(paragraph, ref)
    for key, url in _EXTRA_URL.items():
        if key in ref:
            paragraph.add_run(" ")
            _add_external_link(paragraph, url, url)
            return


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


import re as _re


def tex_rows(name):
    """Rows of a generated paper/tables/<name>.tex (between midrule and bottomrule) as lists of cell strings."""
    txt = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tables", name + ".tex"), encoding="utf-8").read()
    body = txt.split(r"\midrule", 1)[1].split(r"\bottomrule")[0]
    out = []
    for line in body.split("\n"):
        line = line.strip()
        if not line.endswith(r"\\"):
            continue
        cells = []
        for c in line[:-2].split("&"):
            c = _re.sub(r"\\textbf\{([^}]*)\}", r"**\1**", c.strip())
            c = c.replace("$", "").replace(r"\pm", "±")
            c = _re.sub(r"(?<![\w.])-(?=\d)", "−", c)
            cells.append(c.strip())
        out.append(cells)
    return out


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


# ==================================================================================================
# LaTeX -> Word renderer for the single-partition draft (reads ../sections/*.tex, ../tables/*.tex, ../main.tex)
# ==================================================================================================
import re as _re
import subprocess as _sp

_TEXDIR = PAPER_DIR
_REFS = {}          # label -> number string
_BIB = {}           # cite key -> number
_FIGNO = [3]        # figures 1-3 are in the methods
_TABNO = [2]        # tables 1-2 are in the methods


def _read(p):
    return open(str(p), encoding="utf-8").read().replace("\r\n", "\n")


def _balanced(s, i):
    """s[i] == '{' -> (content, index after the closing brace)"""
    depth, j = 0, i
    while j < len(s):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise ValueError("unbalanced")


def _strip_comments(s):
    return "\n".join(_re.sub(r"(?<!\\)%.*$", "", l) for l in s.split("\n"))


def _bib_numbers():
    bbl = _read(_TEXDIR / "main.bbl")
    for n, m in enumerate(_re.finditer(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}", bbl), 1):
        _BIB[m.group(1)] = n


def _assign_labels(texts):
    """Numbers for \\label{sec:..}, tab:.., fig:.. in order of appearance across the given section texts."""
    sec = [0, 0]
    for kind, txt in texts:
        sec[0] = {"results": 3, "discussion": 4, "conclusion": 5}[kind]
        sec[1] = 0
        pos = 0
        for m in _re.finditer(r"\\subsection\{|\\begin\{table\}|\\begin\{figure\}|\\input\{tables/([^}]+)\}|\\label\{([^}]+)\}", txt):
            t = m.group(0)
            if t.startswith("\\subsection"):
                sec[1] += 1
                pend = ("sec", f"{sec[0]}.{sec[1]}")
                lm = _re.search(r"\\label\{([^}]+)\}", txt[m.end():m.end() + 2000])
                # label that follows the subsection title
                j = m.end() - 1
                _, k = _balanced(txt, j)
                lm = _re.match(r"\s*\\label\{([^}]+)\}", txt[k:])
                if lm:
                    _REFS[lm.group(1)] = pend[1]
            elif t.startswith("\\input"):
                tex = _strip_comments(_read(_TEXDIR / "tables" / (m.group(1) + ".tex")))
                if "\\begin{table}" in tex:
                    _TABNO[0] += 1
                    lm = _re.search(r"\\label\{([^}]+)\}", tex)
                    if lm:
                        _REFS[lm.group(1)] = str(_TABNO[0])
            elif t.startswith("\\begin{table}"):
                _TABNO[0] += 1
                e = txt.index("\\end{table}", m.end())
                lm = _re.search(r"\\label\{([^}]+)\}", txt[m.end():e])
                if lm:
                    _REFS[lm.group(1)] = str(_TABNO[0])
            elif t.startswith("\\begin{figure}"):
                _FIGNO[0] += 1
                e = txt.index("\\end{figure}", m.end())
                body = txt[m.end():e]
                for lm in _re.finditer(r"\\label\{(fig:[^}]+)\}", body):
                    if lm.group(1) not in _REFS:
                        _REFS[lm.group(1)] = str(_FIGNO[0])
    _TABNO[0], _FIGNO[0] = 2, 3


_MATH = [(r"\\pm", "±"), (r"\\Delta", "Δ"), (r"\\times", "×"), (r"\\geq", "≥"), (r"\\leq", "≤"), (r"\\approx", "≈"),
         (r"\\rightarrow", "→"), (r"\\chi\^2", "χ²"), (r"\\chi", "χ"), (r"\\sim", "~"), (r"\\le(?![a-z])", "≤"), (r"\\ge(?![a-z])", "≥"),
         (r"\\%", "%"), (r"\\,", " "), (r"\\;", " "), (r"\\!", "")]


def _math(m):
    t = m.group(1)
    t = _re.sub(r"\\text\{([^}]*)\}", r"\1", t)
    for a, b in _MATH:
        t = _re.sub(a, b, t)
    t = t.replace("^2", "²").replace("{", "").replace("}", "")
    t = _re.sub(r"(?<![\w.])-(?=\d)", "−", t)
    t = _re.sub(r"\s*=\s*", " = ", t) if "Δ" in t else t
    return t


def _clean(t):
    """LaTeX inline text -> Word markup text."""
    t = _re.sub(r"\\(?:cite[a-z]*)\{([^}]*)\}", lambda m: "[" + ",".join(str(_BIB.get(k.strip(), "?")) for k in m.group(1).split(",")) + "]", t)
    t = _re.sub(r"\\label\{[^}]*\}", "", t)
    t = _re.sub(r"\\ref\{([^}]*)\}", lambda m: _REFS.get(m.group(1), "?"), t)
    t = _re.sub(r"\$([^$]*)\$", _math, t)
    t = _re.sub(r"\\textbf\{([^{}]*)\}", r"**\1**", t)
    t = _re.sub(r"\\(?:emph|textit)\{([^{}]*)\}", r"\1", t)
    t = _re.sub(r"\\(?:texttt|text|mbox)\{([^{}]*)\}", r"\1", t)
    t = _re.sub(r"\\boldmath\{([^{}]*)\}", r"\1", t)
    t = t.replace("\\#", "#").replace("\\%", "%").replace("\\&", "&").replace("\\_", "_").replace("{,}", ",").replace("\\ ", " ").replace("\\@", "")
    t = t.replace("---", "—").replace("--", "–").replace("``", "“").replace("''", "”").replace("~", " ")
    t = _re.sub(r"\\[a-zA-Z]+\*?", "", t) if False else t
    t = _re.sub(r"[ \t]*\n[ \t]*", " ", t)
    t = _re.sub(r" {2,}", " ", t)
    return t.strip()


def _table(tex):
    """\\begin{table}...\\end{table} text -> add_table(...) call."""
    cap_i = tex.index("\\caption{")
    cap, _ = _balanced(tex, cap_i + len("\\caption"))
    _TABNO[0] += 1
    n = _TABNO[0]
    caption = f"**Table {n}.** " + _clean(cap)
    a = tex.index("\\begin{tabular}")
    body = tex[a:tex.index("\\end{tabular}")]
    body = _re.sub(r"\\begin\{tabular\}(\{[^\n]*\})", "", body, count=1) if False else body
    body = body.split("\n", 1)[1] if body.startswith("\\begin{tabular}") else body
    # tabular spec may contain braces on the same line (p{..}); drop first line only when it is the spec
    lines = [l.strip() for l in body.split("\n")]
    top = next(i for i, l in enumerate(lines) if l.startswith("\\toprule"))
    mid = next(i for i, l in enumerate(lines) if l.startswith("\\midrule"))
    hdr_lines = [l for l in lines[top + 1:mid] if l and not l.startswith("\\cmidrule")]
    rows = []
    for l in lines[mid + 1:]:
        if l.startswith("\\bottomrule"):
            break
        if not l or l.startswith("\\midrule") or l.startswith("\\cmidrule"):
            continue
        rows.append(l)

    def split_row(l):
        l = l.rstrip()
        if l.endswith("\\\\"):
            l = l[:-2]
        cells, depth, cur = [], 0, ""
        for ch in l:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            if ch == "&" and depth == 0:
                cells.append(cur); cur = ""
            else:
                cur += ch
        cells.append(cur)
        return [c.strip() for c in cells]

    def expand(cells):
        out = []
        for c in cells:
            m = _re.match(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}$", c, _re.S)
            if m:
                out += [m.group(2)] + [""] * (int(m.group(1)) - 1)
            else:
                out.append(c)
        return out

    H = [expand(split_row(l)) for l in hdr_lines]
    header = H[0]
    if len(H) == 2:
        header, cur = [], ""
        for g, s2 in zip(H[0], H[1]):
            cur = g or cur
            header.append((cur + " " + s2).strip() if (g or (cur and s2 and header and header[-1] != "")) and cur and s2 else (s2 or g))
    nrow = []
    for l in rows:
        cs = split_row(l)
        if len(cs) == 1 and "\\multicolumn" in cs[0]:
            cs = expand(cs)
            cs = [cs[0]] + [""] * (len(header) - 1)
        cs = [_clean(c) for c in expand(cs)] if len(cs) > 1 else [_clean(c) for c in cs]
        cs = (cs + [""] * len(header))[:len(header)]
        nrow.append(cs)
    header = [_clean(h) for h in header]
    add_table(header, nrow, caption=caption)


def _figure(tex):
    cap_i = tex.index("\\caption{")
    cap, _ = _balanced(tex, cap_i + len("\\caption"))
    _FIGNO[0] += 1
    n = _FIGNO[0]
    panels = []
    for m in _re.finditer(r"\\subfloat\[", tex):
        j = m.end() - 1
        # title in [...] possibly containing \label{..}
        depth, k = 0, j
        while True:
            if tex[k] == "[":
                depth += 1
            elif tex[k] == "]":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        title = _clean(_re.sub(r"\\label\{[^}]*\}", "", tex[j + 1:k]))
        im = _re.search(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex[k:])
        panels.append((title, im.group(1)))
    if not panels:
        for im in _re.finditer(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex):
            panels.append(("", im.group(1)))
    pngs = []
    for idx, (title, path) in enumerate(panels):
        png = os.path.join(FIGDIR_CONV, os.path.basename(path).replace(".pdf", ".png"))
        # always re-render from the PDF so that no stale PNG (e.g. copied from the multi-seed paper) is used
        _sp.run(["pdftocairo", "-png", "-r", "200", "-singlefile", os.path.join(str(PAPER_DIR), path),
                 png[:-4]], check=True)
        pngs.append(png)
    cap_txt = _clean(cap)
    if len(panels) == 1:
        add_figure(pngs[0], f"**Figure {n}.** {cap_txt}", width_in=5.0)
    else:
        # all sub-figures of one LaTeX figure -> ONE composite image (2 columns, panel letters A, B, ...), one caption
        add_figure(_compose_panels(pngs, os.path.join(FIGDIR_CONV, f"composite_fig{n}.png")),
                   f"**Figure {n}.** {cap_txt}", width_in=6.3)


def _compose_panels(pngs, out, ncols=2, pad=30):
    from PIL import Image, ImageDraw, ImageFont
    ims = [Image.open(p).convert("RGB") for p in pngs]
    w = max(i.width for i in ims)
    ims = [i.resize((w, round(i.height * w / i.width))) if i.width != w else i for i in ims]
    nrows = -(-len(ims) // ncols)
    rowh = [max(i.height for i in ims[r * ncols:(r + 1) * ncols]) for r in range(nrows)]
    canvas = Image.new("RGB", (ncols * w + (ncols + 1) * pad, sum(rowh) + (nrows + 1) * pad), "white")
    try:
        font = ImageFont.truetype("arialbd.ttf", 64)
    except OSError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    y = pad
    for r in range(nrows):
        for c in range(ncols):
            k = r * ncols + c
            if k >= len(ims):
                break
            x = pad + c * (w + pad)
            canvas.paste(ims[k], (x, y))
            draw.text((x + 8, y + 4), "ABCDEFGH"[k], fill="black", font=font)
        y += rowh[r] + pad
    canvas.save(out)
    return out


def render_tex(kind):
    txt = _strip_comments(_read(_TEXDIR / "sections" / f"{kind}.tex"))
    txt = _re.sub(r"\\section\{[^}]*\}\s*(\\label\{[^}]*\})?", "", txt)
    sub = [0]
    secno = {"results": 3, "discussion": 4, "conclusion": 5}[kind]
    # split into blocks at environments / commands
    pos, buf = 0, []

    def flush(text):
        text = text.strip()
        if not text:
            return
        for para in _re.split(r"\n\s*\n", text):
            para = para.strip()
            if not para:
                continue
            m = _re.match(r"\\paragraph\{([^}]*)\}\s*(.*)$", para, _re.S)
            if m:
                P(f"**{_clean(m.group(1))}** " + _clean(m.group(2)))
            else:
                P(_clean(para))

    pat = _re.compile(r"\\subsection\{|\\begin\{table\}|\\begin\{figure\}|\\input\{tables/([^}]+)\}")
    while True:
        m = pat.search(txt, pos)
        if not m:
            flush(txt[pos:])
            break
        flush(txt[pos:m.start()])
        t = m.group(0)
        if t.startswith("\\subsection"):
            title, k = _balanced(txt, m.end() - 1)
            sub[0] += 1
            H(f"{secno}.{sub[0]}. {_clean(title)}", level=2)
            pos = k
            lm = _re.match(r"\s*\\label\{[^}]*\}", txt[pos:])
            if lm:
                pos += lm.end()
        elif t.startswith("\\begin{table}"):
            e = txt.index("\\end{table}", m.end()) + len("\\end{table}")
            _table(txt[m.start():e]); pos = e
        elif t.startswith("\\begin{figure}"):
            e = txt.index("\\end{figure}", m.end()) + len("\\end{figure}")
            _figure(txt[m.start():e]); pos = e
        else:
            tex = _strip_comments(_read(_TEXDIR / "tables" / (m.group(1) + ".tex")))
            if "\\begin{table}" in tex:
                _table(tex)
            pos = m.end()


def abstract_text():
    s = _strip_comments(_read(_TEXDIR / "main.tex"))
    i = s.index("\\abstract{")
    body, _ = _balanced(s, i + len("\\abstract"))
    return _clean(body)


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
add_markup(auth_p, "V\u0169 Huy Ho\u00e0ng^{1,*}")
auth_p.paragraph_format.space_after = Pt(4)

aff_p = doc.add_paragraph()
aff_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
add_markup(
    aff_p,
    "^{1}Master's student (Student ID 20242052M), School of Information and Communication Technology, Hanoi University of Science and Technology, Hanoi, Vietnam"
)
for r in aff_p.runs:
    r.font.size = Pt(9.5)
aff_p.paragraph_format.space_after = Pt(2)

corr_p = doc.add_paragraph()
corr_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
add_markup(corr_p, "**Correspondence:** Hoang.VH242052M@sis.hust.edu.vn; Tel.: +84-327-633-274")
for r in corr_p.runs:
    r.font.size = Pt(9.5)
corr_p.paragraph_format.space_after = Pt(16)

# ---------------------------------------------------------------- abstract -
H("Abstract", level=2)
P(abstract_text())

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
    "or acquired resistance [3,4]. Reliable pre-treatment identification of "
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
    "modest discriminative performance [5–8]. This has motivated growing "
    "interest in multimodal approaches that integrate radiomics from CT "
    "imaging, pathomics from digitised histology, genomic alterations, and "
    "clinical variables into a single predictive model [9–13]. Such "
    "approaches are appealing because they can capture complementary "
    "aspects of tumour biology and host physiology that no single modality "
    "fully represents; however, they introduce new methodological "
    "challenges, including how to weight modalities of differing "
    "reliability and how to handle missing modalities without discarding "
    "patients [14]. The Dynamic Attention Multimodal (DyAM) framework [15] "
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
    "that preserve semantic similarity [16,17], and domain-specific "
    "clinical language models have been developed for free-text clinical "
    "notes [18]. However, whether converting *structured, tabular* clinical "
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
    "In this study, we extend the DyAM framework with two "
    "contributions and evaluate them on a discovery cohort of 247 "
    "patients with advanced NSCLC treated with ICIs. First, we "
    "introduce an OvO competitive attention variant "
    "(AttentionMatrixOvO) and compare it against the original "
    "cooperative attention across 21 modality combinations using "
    "10-fold cross-validation, without and with the clinical "
    "encoding below. Second, we convert 13 numeric "
    "clinical variables into natural-language sentences and encode "
    "them with a pretrained sentence transformer (MiniLM-L6-v2), "
    "comparing this NLP-based representation against raw numeric "
    "encoding, a domain-specific clinical language model "
    "(BioClinBERT), and a PCA-compressed variant, evaluated using "
    "both binary classification (AUC) and survival analysis (Cox "
    "regression, Harrell's C-index, and Kaplan–Meier "
    "stratification). Because 247 patients are few for "
    "hypothesis tests of differences of the size at stake, we "
    "report the direction and magnitude of differences with the "
    "number of configurations favouring each model and interval "
    "estimates, and we state where they are within the confidence "
    "intervals. Together, "
    "these contributions aim to show that (i) competitive "
    "attention can match or exceed cooperative attention in "
    "multimodal fusion with fewer parameters, and in which "
    "configurations, and (ii) that NLP-based encoding of "
    "structured clinical data is a practical, "
    "fine-tuning-free alternative to conventional numeric "
    "representations for response prediction and survival risk "
    "stratification."
)

print("intro done")

# =================================================== 2. MATERIALS/METHODS =
H("2. Materials and Methods", level=1)

H("2.1. Study Design and Patient Cohort", level=2)
P(
    "This retrospective study used data from the MSK MIND (Multi-Modal "
    "Integration of Data) initiative at Memorial Sloan Kettering Cancer "
    "Center [15], comprising patients with advanced non-small cell lung cancer "
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
    "radiomics features were extracted from baseline scans using "
    "PyRadiomics [19], following the preprocessing of [15], at 1.0 mm isotropic spacing with a window of "
    "[−1350, 250] HU, yielding 1,688 features per lesion type. Lesions "
    "were categorised by anatomical site: primary tumour (PC), pleural "
    "lesion (PL), and lymph node (LN). Up to three lesions per site were "
    "included, ranked by volume. Pathology features were derived from "
    "PD-L1 immunohistochemistry (IHC) slides: *IHC-A* comprised 18 "
    "pixel-intensity aggregation features (area, mean intensity, "
    "percentiles), while *IHC-G* comprised 150 grey-level co-occurrence "
    "matrix (GLCM) texture features across six aggregation levels [15]. "
    "Genomic features (11 variables) included tumour mutational burden "
    "(TMB) and the status of eight driver mutations and amplifications "
    "derived from MSK-IMPACT targeted next-generation sequencing (NGS) [20]. The "
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
        ["CT Radiomics (PC, PL, LN)", "PyRadiomics", "≤1,688", "Tumour shape, size, and texture from baseline CT"],
        ["Pathology IHC-A", "PD-L1 IHC slides", "18", "Pixel-intensity aggregation of PD-L1 expression"],
        ["Pathology IHC-G", "PD-L1 IHC slides", "150", "GLCM texture of PD-L1 signal"],
        ["Genomics", "MSK-IMPACT", "11", "TMB + driver mutation/amplification status"],
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
    "sentence-transformers/all-MiniLM-L6-v2 [16,17], a 6-layer transformer "
    "distilled for semantic sentence similarity and trained on over one "
    "billion sentence pairs. Each patient was represented as a "
    "384-dimensional L2-normalised vector. No fine-tuning was performed; "
    "the model was applied in zero-shot transfer. We compared this "
    "general-purpose encoder against the domain-specific Bio_ClinicalBERT "
    "[18]; the domain-specific model performed inferiorly (ΔAUC = −0.012 "
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
    "[15], fuses ", IM(m.run("N")), " modalities through per-modality "
    "risk scores and a learned attention mechanism, in the spirit of "
    "attention-based architectures broadly [21]. Let ",
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
    "mechanism inspired by one-versus-others (OvO) classification [22], in "
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

P(
    "The model was trained with binary cross-entropy loss "
    "(BCEWithLogitsLoss) with automatic inverse-frequency class "
    "weighting to address the ≈1:3 class imbalance, an ℓ2 penalty "
    "on the weights (coefficient α = 0.001); the attention-norm "
    "penalty was disabled (β = 0). The Adam optimiser was used with "
    "a learning rate of 0.01 for 125 epochs. All experiments used "
    "10-fold cross-validation (shuffled, not stratified) on the "
    "discovery cohort; predictions were pooled across held-out "
    "folds for AUC computation."
)

H("2.5. Statistical Analysis and Evaluation", level=2)
P(
    "**Discriminative performance.** Area under the receiver "
    "operating characteristic curve (AUC-ROC) was computed on "
    "pooled out-of-fold predictions from 10-fold cross-validation. "
    "Ninety-five percent confidence intervals (CIs) were estimated "
    "using the DeLong structural components method [23]. "
    "All results come from a single 10-fold cross-validation "
    "partition (seed 42, natural patient order, the protocol of the "
    "original publication). Differences between models are reported "
    "as the difference in AUC and the number of configurations "
    "favouring each model. Because the cohort is small, hypothesis "
    "tests were not used to claim that one model is superior; "
    "DeLong intervals are reported for single AUC values."
)
P(
    "**Survival analysis.** Harrell's concordance index (C-index) "
    "[24] was computed from pooled out-of-fold risk scores against "
    "PFS, from the same partition. Multivariate Cox "
    "proportional hazards regression was fitted with the DyAM risk "
    "score as the primary predictor, adjusted for five clinical "
    "covariates: age, ECOG performance status, serum albumin, dNLR, "
    "and liver metastasis status; hazard ratios are per unit of "
    "risk score. Time-dependent AUC (6, 12 and 18 months; "
    "cumulative/dynamic estimator [25,26]) was computed from the "
    "same partition (scikit-survival 0.28 with numpy 2.5.4 in "
    "a separate environment, from the stored out-of-fold risk "
    "scores)."
)
P(
    "**Kaplan–Meier survival stratification.** Patients were "
    "dichotomised at the median risk score, and PFS curves for the "
    "two groups were compared with the log-rank test. The log-rank "
    "χ² is reported with χ² > 3.84 (p < 0.05) and χ² > 7.88 (p < "
    "0.005) as reference values."
)
P(
    "**Software.** All numerical results in Tables 3–8 were "
    "computed in one pinned environment (Linux under WSL2, Ubuntu "
    "22.04; Python 3.8.15, PyTorch 1.8.1 (CPU), scikit-learn "
    "0.24.2, scipy 1.6.3, numpy 1.20.3, pandas 1.2.5, and lifelines "
    "0.26.0 for Kaplan–Meier, Cox regression and the C-index). In "
    "this environment the original DyAM code reproduced the "
    "single-partition AUC of the original publication at its "
    "primary benchmark (0.798) to within one patient pair. The same "
    "code and library versions on another operating system gave "
    "AUCs that differed in the second or third decimal place, so "
    "results were never compared across platforms; the radiomics "
    "feature selection used the original procedure, without removal "
    "of degenerate columns. The sentence-encoder embeddings "
    "(MiniLM-L6-v2 and Bio_ClinicalBERT) were computed once in a "
    "separate environment and cached, and time-dependent AUC was "
    "computed from the stored out-of-fold risk scores with "
    "scikit-survival 0.28 (numpy 2.5.4)."
)

print("methods done")


# ============================================================ 3. RESULTS (rendered from the LaTeX sections) =
_bib_numbers()
_assign_labels([("results", _strip_comments(_read(_TEXDIR / "sections" / "results.tex"))),
                ("discussion", _strip_comments(_read(_TEXDIR / "sections" / "discussion.tex"))),
                ("conclusion", _strip_comments(_read(_TEXDIR / "sections" / "conclusion.tex")))])
H("3. Results", level=1)
render_tex("results")
H("4. Discussion", level=1)
render_tex("discussion")
H("5. Conclusions", level=1)
render_tex("conclusion")

# ======================================================== DECLARATIONS ===
H("Declarations", level=1)
P("**Author Contributions:** Conceptualization, methodology, software, formal analysis, investigation, writing—original draft preparation, and writing—review and editing, V.H.H. The author has read and agreed to the published version of the manuscript.")
P("**Data Availability Statement:** The code used in this study is "
  "available at https://github.com/hoanghuyvu2k/Multimodal-Lung-Cancer. Patient-level data are "
  "subject to institutional data-sharing agreements and available upon "
  "reasonable request.")
P("**Conflicts of Interest:** The author declares no conflicts of "
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
    "Sharma P, Hu-Lieskovan S, Wargo JA, Ribas A. Primary, Adaptive, and "
    "Acquired Resistance to Cancer Immunotherapy. Cell. 2017;168(4):707–723. "
    "doi:10.1016/j.cell.2017.01.017",
    "Herbst RS, Baas P, Kim DW, et al. Pembrolizumab versus docetaxel for "
    "previously treated, PD-L1-positive, advanced non-small-cell lung "
    "cancer (KEYNOTE-010). Lancet. 2016;387:1540–1550. "
    "doi:10.1016/S0140-6736(15)01281-7",
    "Hirsch FR, McElhinny A, Stanforth D, et al. PD-L1 Immunohistochemistry "
    "Assays for Lung Cancer: Results from Phase 1 of the Blueprint PD-L1 IHC "
    "Assay Comparison Project. J Thorac Oncol. 2017;12(2):208–222. "
    "doi:10.1016/j.jtho.2016.11.2228",
    "Rizvi NA, Hellmann MD, Snyder A, et al. Mutational landscape "
    "determines sensitivity to PD-1 blockade in non-small cell lung "
    "cancer. Science. 2015;348:124–128. doi:10.1126/science.aaa1348",
    "Lu S, Stein JE, Rimm DL, et al. Comparison of Biomarker Modalities for "
    "Predicting Response to PD-1/PD-L1 Checkpoint Blockade: A Systematic "
    "Review and Meta-analysis. JAMA Oncol. 2019;5(8):1195–1204. "
    "doi:10.1001/jamaoncol.2019.1549",
    "Boehm KM, Khosravi P, Vanguri R, Gao J, Shah SP. Harnessing multimodal "
    "data integration to advance precision oncology. Nat Rev Cancer. "
    "2022;22(2):114–126. doi:10.1038/s41568-021-00408-3",
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
    "and 9th Int Jt Conf Nat Lang Process (EMNLP-IJCNLP). 2019:3982–3992. doi:10.18653/v1/D19-1410",
    "Wang W, Wei F, Dong L, et al. MiniLM: Deep Self-Attention "
    "Distillation for Task-Agnostic Compression of Pre-Trained "
    "Transformers. Adv Neural Inf Process Syst (NeurIPS). "
    "2020;33:5776–5788.",
    "Alsentzer E, Murphy JR, Boag W, et al. Publicly Available Clinical "
    "BERT Embeddings. Proc 2nd Clinical Nat Lang Process Workshop. "
    "2019:72–78. doi:10.18653/v1/W19-1909",
    "van Griethuysen JJM, Fedorov A, Parmar C, et al. Computational "
    "Radiomics System to Decode the Radiographic Phenotype. Cancer Res. "
    "2017;77(21):e104–e107. doi:10.1158/0008-5472.CAN-17-0339",
    "Cheng DT, Mitchell TN, Zehir A, et al. Memorial Sloan Kettering-"
    "Integrated Mutation Profiling of Actionable Cancer Targets (MSK-IMPACT): "
    "A Hybridization Capture-Based Next-Generation Sequencing Clinical Assay "
    "for Solid Tumor Molecular Oncology. J Mol Diagn. 2015;17(3):251–264. "
    "doi:10.1016/j.jmoldx.2014.12.006",
    "Vaswani A, Shazeer N, Parmar N, et al. Attention Is All You Need. "
    "Adv Neural Inf Process Syst (NeurIPS). 2017;30:5998–6008.",
    "Rifkin R, Klautau A. In Defense of One-Vs-All Classification. J Mach "
    "Learn Res. 2004;5:101–141.",
    "DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the Areas under "
    "Two or More Correlated Receiver Operating Characteristic Curves: A "
    "Nonparametric Approach. Biometrics. 1988;44:837–845. "
    "doi:10.2307/2531595",
    "Harrell FE, Lee KL, Mark DB. Multivariable Prognostic Models: Issues "
    "in Developing Models, Evaluating Assumptions and Adequacy, and "
    "Measuring and Reducing Errors. Stat Med. 1996;15:361–387. "
    "doi:10.1002/(SICI)1097-0258(19960229)15:4<361::AID-SIM168>3.0.CO;2-4",
    "Uno H, Cai T, Tian L, Wei LJ. Evaluating Prediction Rules for t-Year "
    "Survivors With Censored Regression Models. J Am Stat Assoc. "
    "2007;102(478):527–537. doi:10.1198/016214507000000149",
    "Hung H, Chiang CT. Estimation methods for time-dependent AUC models "
    "with survival data. Can J Stat. 2010;38(1):8–26. doi:10.1002/cjs.10046",
]

for i, ref in enumerate(REFERENCES, 1):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.75)
    p.paragraph_format.first_line_indent = Cm(-0.75)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(f"{i}.\t")
    _add_bookmark(p, f"ref_{i}", 9000 + i)
    _add_ref_text_online(p, ref)
    for run in p.runs:
        run.font.size = Pt(10)

print("references done")
# Lưu; nếu file đích đang mở trong Word (bị khoá) thì fallback _v2, _v3...
_target = Path(OUT_PATH)
for _old in list(PAPER_DIR.glob(f"manuscript_word_{_TAG}*.docx")):
    if _old != _target:
        try:
            (PAPER_DIR / "word_archive").mkdir(exist_ok=True)
            _sh.move(str(_old), str(PAPER_DIR / "word_archive" / _old.name))
        except OSError:
            pass          # locked (open in Word): leave it where it is
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
