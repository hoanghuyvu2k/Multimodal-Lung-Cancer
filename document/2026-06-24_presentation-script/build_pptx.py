# -*- coding: utf-8 -*-
"""
Dựng bộ slide PowerPoint (.pptx) tiếng Việt cho bài báo
NLP-Augmented Multimodal Attention Fusion (NSCLC ICI response).

Chạy:  python build_pptx.py
Xuất:  slides.pptx  (cùng thư mục)
Yêu cầu: python-pptx; các hình PNG trong ../../paper/figures/
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image

FIG = os.path.join(os.path.dirname(__file__), '..', '..', 'paper', 'figures')

# ── Bảng màu (khớp với hình trong bài báo) ───────────────────────────────────
GREEN  = RGBColor(0x1E, 0x84, 0x49)
GREEN_L= RGBColor(0xE5, 0xF3, 0xEB)
PURPLE = RGBColor(0x6C, 0x34, 0x83)
PURPLE_L=RGBColor(0xF0, 0xE8, 0xF5)
ORANGE = RGBColor(0xB7, 0x46, 0x0A)
ORANGE_L=RGBColor(0xFB, 0xEC, 0xDF)
SLATE  = RGBColor(0x2C, 0x3E, 0x50)
GRAY   = RGBColor(0x5D, 0x6D, 0x7E)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
FONT = 'Arial'

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]


def slide():
    return prs.slides.add_slide(BLANK)


def rect(s, l, t, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE, line_w=1.0):
    sp = s.shapes.add_shape(shape, l, t, w, h)
    sp.fill.solid(); sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line; sp.line.width = Pt(line_w)
    sp.shadow.inherit = False
    return sp


def textbox(s, l, t, w, h, anchor=MSO_ANCHOR.TOP):
    tb = s.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = Pt(2); tf.margin_right = Pt(2)
    tf.margin_top = Pt(2); tf.margin_bottom = Pt(2)
    return tb, tf


def setpara(p, text, size, color=SLATE, bold=False, italic=False,
            align=PP_ALIGN.LEFT, font=FONT, space_after=6):
    p.text = text
    p.alignment = align
    p.space_after = Pt(space_after)
    r = p.runs[0]
    r.font.size = Pt(size); r.font.bold = bold; r.font.italic = italic
    r.font.color.rgb = color; r.font.name = font
    return p


def title_bar(s, title):
    """Thanh tiêu đề xanh ở đầu slide."""
    bar = rect(s, 0, 0, SW, Inches(0.92), GREEN)
    tf = bar.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.45)
    p = tf.paragraphs[0]
    setpara(p, title, 26, WHITE, bold=True, space_after=0)


def footer(s, idx):
    tb, tf = textbox(s, Inches(0.4), Inches(7.05), Inches(9.5), Inches(0.4))
    setpara(tf.paragraphs[0],
            'DyAM + OvO + NLP  —  Dự đoán đáp ứng ICI trong NSCLC',
            10, GRAY, space_after=0)
    tb2, tf2 = textbox(s, Inches(11.8), Inches(7.05), Inches(1.3), Inches(0.4))
    setpara(tf2.paragraphs[0], f'{idx} / {TOTAL}', 10, GRAY,
            align=PP_ALIGN.RIGHT, space_after=0)


def bullets(tf, items, size=17, base_color=SLATE):
    """items: list of (text, level, color, bold)."""
    for i, it in enumerate(items):
        text, level, color, bold = it
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = level
        prefix = ('• ' if level == 0 else '– ')
        setpara(p, prefix + text, size - (level*1.5), color, bold=bold,
                space_after=7)


def block(s, l, t, w, h, title, body, accent, accent_l):
    """Hộp nhấn mạnh kiểu Beamer block."""
    box = rect(s, l, t, w, h, accent_l, line=accent, shape=MSO_SHAPE.ROUNDED_RECTANGLE, line_w=1.25)
    tf = box.text_frame; tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Pt(10); tf.margin_right = Pt(10)
    tf.margin_top = Pt(6); tf.margin_bottom = Pt(6)
    if title:
        setpara(tf.paragraphs[0], title, 15, accent, bold=True, space_after=3)
        p = tf.add_paragraph()
        setpara(p, body, 14, SLATE, space_after=0)
    else:
        setpara(tf.paragraphs[0], body, 14, SLATE, space_after=0)
    return box


def image(s, name, left, top, width=None, height=None, center_x=None):
    path = os.path.join(FIG, name)
    iw, ih = Image.open(path).size
    ar = iw / ih
    if width is not None:
        w = width; h = Emu(int(width / ar))
    else:
        h = height; w = Emu(int(height * ar))
    if center_x is not None:
        left = int(center_x - w / 2)
    s.shapes.add_picture(path, int(left), int(top), int(w), int(h))
    return w, h


# ── Render công thức toán bằng matplotlib mathtext → PNG trong suốt ──────────
FORM_DIR = os.path.join(os.path.dirname(__file__), '_formulas')
os.makedirs(FORM_DIR, exist_ok=True)


_fcount = [0]
def render_math(tex, name=None, fontsize=26, color='#2C3E50'):
    if name is None:
        _fcount[0] += 1; name = f'formula_{_fcount[0]}'
    path = os.path.join(FORM_DIR, name + '.png')
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0.5, 0.5, f'${tex}$', fontsize=fontsize, ha='center', va='center',
             color=color)
    fig.savefig(path, dpi=220, bbox_inches='tight', pad_inches=0.06,
                transparent=True)
    plt.close(fig)
    return path


def img_abs(s, path, left, top, width=None, height=None, center_x=None):
    iw, ih = Image.open(path).size
    ar = iw / ih
    if width is not None:
        w = width; h = Emu(int(width / ar))
    else:
        h = height; w = Emu(int(height * ar))
    if center_x is not None:
        left = int(center_x - w / 2)
    s.shapes.add_picture(path, int(left), int(top), int(w), int(h))
    return w, h


def metric_card(s, l, t, w, h, accent, accent_l, title, formula_path,
                meaning, formula_h, title_size=14, meaning_size=12):
    rect(s, l, t, w, h, accent_l, line=accent,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, line_w=1.25)
    tb, tf = textbox(s, l + Inches(0.18), t + Inches(0.06),
                     w - Inches(0.36), Inches(0.42))
    setpara(tf.paragraphs[0], title, title_size, accent, bold=True, space_after=0)
    img_abs(s, formula_path, None, t + Inches(0.5), height=formula_h,
            center_x=l + w / 2)
    mb, mf = textbox(s, l + Inches(0.22), t + h - Inches(0.62),
                     w - Inches(0.44), Inches(0.58))
    setpara(mf.paragraphs[0], meaning, meaning_size, SLATE, space_after=0)


TOTAL = 22

# ═══════════════════════════════════════════════════════════════════════════
# Slide 1 — Tiêu đề
s = slide()
rect(s, 0, 0, SW, SH, WHITE)
banner = rect(s, Inches(0.6), Inches(0.7), Inches(12.13), Inches(2.6), GREEN,
              shape=MSO_SHAPE.ROUNDED_RECTANGLE)
tf = banner.text_frame; tf.word_wrap = True
tf.vertical_anchor = MSO_ANCHOR.MIDDLE
setpara(tf.paragraphs[0],
        'Tích hợp đa phương thức dựa trên cơ chế Attention',
        30, WHITE, bold=True, align=PP_ALIGN.CENTER, space_after=2)
setpara(tf.add_paragraph(),
        'tăng cường NLP để dự đoán đáp ứng miễn dịch',
        30, WHITE, bold=True, align=PP_ALIGN.CENTER, space_after=2)
setpara(tf.add_paragraph(),
        'trong ung thư phổi không tế bào nhỏ',
        30, WHITE, bold=True, align=PP_ALIGN.CENTER, space_after=8)
setpara(tf.add_paragraph(),
        'Phát triển trên nền tảng DyAM (Vanguri et al., Nature Cancer 2022)',
        15, WHITE, italic=True, align=PP_ALIGN.CENTER, space_after=0)
tb, tf = textbox(s, Inches(0.6), Inches(4.0), Inches(12.13), Inches(2.2),
                 anchor=MSO_ANCHOR.TOP)
setpara(tf.paragraphs[0], 'Học viên thực hiện: [Tên học viên]', 18, SLATE,
        bold=True, align=PP_ALIGN.CENTER, space_after=10)
setpara(tf.add_paragraph(), 'Người hướng dẫn: [Tên GVHD]   •   [Đơn vị]',
        16, GRAY, align=PP_ALIGN.CENTER, space_after=10)
setpara(tf.add_paragraph(), 'Cohort khám phá: 247 bệnh nhân NSCLC điều trị ICI',
        14, GRAY, italic=True, align=PP_ALIGN.CENTER, space_after=0)

# ═══════════════════════════════════════════════════════════════════════════
# Slide 2 — Nội dung
s = slide(); title_bar(s, 'Nội dung trình bày'); footer(s, 2)
tb, tf = textbox(s, Inches(1.2), Inches(1.4), Inches(11), Inches(5.4))
bullets(tf, [
    ('1.  Đặt vấn đề — bài toán dự đoán đáp ứng miễn dịch', 0, GREEN, True),
    ('2.  Phương pháp — DyAM, hai cơ chế Attention, NLP encoding', 0, GREEN, True),
    ('3.  Kết quả — phân loại (AUC) và phân tích sống còn', 0, GREEN, True),
    ('4.  Bàn luận & Kết luận', 0, GREEN, True),
], size=22)

# ═══════════════════════════════════════════════════════════════════════════
# Slide 3 — Bài toán lâm sàng
s = slide(); title_bar(s, 'Bài toán lâm sàng'); footer(s, 3)
tb, tf = textbox(s, Inches(0.6), Inches(1.2), Inches(12.1), Inches(3.4))
bullets(tf, [
    ('Ung thư phổi: nguyên nhân tử vong do ung thư hàng đầu thế giới; NSCLC chiếm ≈ 85% số ca.', 0, SLATE, False),
    ('Liệu pháp miễn dịch ICI (ức chế trục PD-1/PD-L1) tạo đột phá điều trị, đáp ứng bền vững ở một nhóm bệnh nhân.', 0, SLATE, False),
    ('Nhưng chỉ 20–30% bệnh nhân đáp ứng — phần lớn kháng thuốc nguyên phát hoặc mắc phải.', 0, ORANGE, True),
], size=18)
block(s, Inches(0.6), Inches(5.0), Inches(12.1), Inches(1.5),
      'Nhu cầu lâm sàng',
      'Dự đoán TRƯỚC điều trị bệnh nhân nào có khả năng đáp ứng ⇒ ưu tiên đúng người, '
      'tránh điều trị tốn kém và độc tính cho người ít khả năng đáp ứng.',
      GREEN, GREEN_L)

# ═══════════════════════════════════════════════════════════════════════════
# Slide 4 — Hạn chế biomarker
s = slide(); title_bar(s, 'Hạn chế của biomarker hiện tại'); footer(s, 4)
tb, tf = textbox(s, Inches(0.6), Inches(1.2), Inches(12.1), Inches(3.6))
bullets(tf, [
    ('Hai biomarker phổ biến: PD-L1 TPS và TMB — khả năng phân biệt còn hạn chế.', 0, SLATE, True),
    ('PD-L1: không đồng nhất về không gian, phụ thuộc loại xét nghiệm.', 1, GRAY, False),
    ('TMB đơn lẻ: chỉ phân biệt ở mức trung bình.', 1, GRAY, False),
    ('⇒ Xu hướng tích hợp đa phương thức: radiomics (CT) + bệnh học (mô) + genomics + lâm sàng.', 0, GREEN, True),
], size=18)
block(s, Inches(0.6), Inches(5.0), Inches(12.1), Inches(1.5),
      'Thách thức của hướng đa phương thức',
      '(1) Gán trọng số cho các phương thức có độ tin cậy khác nhau;   '
      '(2) Xử lý dữ liệu thiếu mà không loại bỏ bệnh nhân.',
      ORANGE, ORANGE_L)

# ═══════════════════════════════════════════════════════════════════════════
# Slide 5 — DyAM & khoảng trống
s = slide(); title_bar(s, 'Mô hình nền DyAM & khoảng trống nghiên cứu'); footer(s, 5)
tb, tf = textbox(s, Inches(0.6), Inches(1.15), Inches(12.1), Inches(1.5))
setpara(tf.paragraphs[0],
        'DyAM (Vanguri 2022): attention học theo từng bệnh nhân, tự gán trọng số mỗi '
        'phương thức, xử lý dữ liệu thiếu bằng masking.', 17, SLATE, space_after=0)
block(s, Inches(0.6), Inches(2.7), Inches(12.1), Inches(3.6),
      'Hai khoảng trống chưa giải quyết',
      '1)  Biểu diễn biến lâm sàng: DyAM dùng SỐ THÔ → bỏ mất quan hệ giữa các biến '
      '(tuổi cao + albumin thấp + ECOG kém).\n\n'
      '2)  Dạng attention: DyAM chỉ dùng attention HỢP TÁC (cooperative) → ngầm giả định '
      'mọi phương thức bổ sung, cộng gộp.',
      ORANGE, ORANGE_L)

# ═══════════════════════════════════════════════════════════════════════════
# Slide 6 — Mục tiêu & đóng góp
s = slide(); title_bar(s, 'Mục tiêu & đóng góp'); footer(s, 6)
block(s, Inches(0.6), Inches(1.3), Inches(12.1), Inches(1.7),
      'Đóng góp 1 — OvO competitive attention',
      'Cơ chế attention CẠNH TRANH (one-vs-others); so sánh với attention hợp tác gốc '
      'trên 20 tổ hợp phương thức.', GREEN, GREEN_L)
block(s, Inches(0.6), Inches(3.2), Inches(12.1), Inches(1.7),
      'Đóng góp 2 — NLP encoding',
      'Chuyển 13 biến lâm sàng số → câu tiếng Anh → embedding bằng sentence-transformer '
      '(MiniLM-L6-v2).', PURPLE, PURPLE_L)
tb, tf = textbox(s, Inches(0.6), Inches(5.2), Inches(12.1), Inches(1.6))
setpara(tf.paragraphs[0],
        'Đánh giá trên CẢ HAI trục: phân loại nhị phân (AUC) VÀ phân tích sống còn '
        '(Cox, C-index, Kaplan–Meier). Cohort khám phá: 247 bệnh nhân NSCLC điều trị ICI.',
        15, GRAY, italic=True, space_after=0)

# ═══════════════════════════════════════════════════════════════════════════
# Slide 7 — Tổng quan thiết kế (fig1)
s = slide(); title_bar(s, 'Tổng quan thiết kế nghiên cứu'); footer(s, 7)
image(s, 'fig1_overview.png', None, Inches(1.2), height=Inches(5.0),
      center_x=SW/2)
tb, tf = textbox(s, Inches(0.6), Inches(6.35), Inches(12.1), Inches(0.6))
setpara(tf.paragraphs[0],
        '(A) Nguồn dữ liệu   •   (B) Kiến trúc DyAM: risk score + attention   •   '
        '(C) Đánh giá AUC & sống còn',
        13, GRAY, italic=True, align=PP_ALIGN.CENTER, space_after=0)

# ═══════════════════════════════════════════════════════════════════════════
# Slide 8 — Hai cơ chế attention (fig_attention + text)
s = slide(); title_bar(s, 'Hai cơ chế Attention — điểm cốt lõi'); footer(s, 8)
image(s, 'fig_attention_arch.png', Inches(0.35), Inches(1.55),
      width=Inches(8.0))
tb, tf = textbox(s, Inches(8.55), Inches(1.1), Inches(4.5), Inches(0.9))
setpara(tf.paragraphs[0],
        'Cả hai:  r_i = tanh(W_r·x_i),   ŷ = Σ r_i·a_i', 13, SLATE, bold=True,
        space_after=0)
block(s, Inches(8.55), Inches(2.05), Inches(4.45), Inches(2.0),
      '(A) Cooperative',
      'Ma trận N×N + softplus → L1-norm. Các phương thức CHIA SẺ NGÂN SÁCH (tổng = 1).',
      GREEN, GREEN_L)
block(s, Inches(8.55), Inches(4.2), Inches(4.45), Inches(2.4),
      '(B) Competitive OvO',
      'σ(s_i − μ): mỗi phương thức so điểm với TRUNG BÌNH các phương thức CÒN LẠI (μ). '
      'Trọng số ĐỘC LẬP, kiểu "best-of".',
      PURPLE, PURPLE_L)

# ═══════════════════════════════════════════════════════════════════════════
# Slide 9 — NLP encoding (fig2)
s = slide(); title_bar(s, 'Đóng góp 2: NLP encoding cho biến lâm sàng'); footer(s, 9)
image(s, 'fig2_nlp_pipeline.png', None, Inches(1.25), height=Inches(4.2),
      center_x=SW/2)
tb, tf = textbox(s, Inches(0.6), Inches(5.65), Inches(12.1), Inches(1.3))
setpara(tf.paragraphs[0],
        '13 biến số → câu tiếng Anh → MiniLM-L6-v2 → embedding 384-d ("NLP raw") / PCA 16-d.',
        14, SLATE, align=PP_ALIGN.CENTER, space_after=4)
setpara(tf.add_paragraph(),
        'Quan trọng: no_scale=True để giữ hình học cosine. Không cần fine-tune, < 1s/bệnh nhân trên CPU.',
        14, ORANGE, bold=True, align=PP_ALIGN.CENTER, space_after=0)

# ═══════════════════════════════════════════════════════════════════════════
# Slide 10 — Dữ liệu & đánh giá
s = slide(); title_bar(s, 'Dữ liệu & phương pháp đánh giá'); footer(s, 10)
tb, tf = textbox(s, Inches(0.6), Inches(1.2), Inches(6.0), Inches(4.2))
setpara(tf.paragraphs[0], 'Dữ liệu & huấn luyện', 17, GREEN, bold=True, space_after=8)
bullets_data = [
    ('Cohort khám phá: 247 BN NSCLC', 0, SLATE, False),
    ('Nhãn: đáp ứng (PR/CR) vs không (SD/PD); mất cân bằng ≈ 1:3', 0, SLATE, False),
    ('BCE loss + cân bằng lớp tự động; Adam; 10-fold CV', 0, SLATE, False),
]
for it in bullets_data:
    p = tf.add_paragraph(); setpara(p, '• ' + it[0], 15, it[2], space_after=7)
tb2, tf2 = textbox(s, Inches(6.9), Inches(1.2), Inches(5.9), Inches(4.2))
setpara(tf2.paragraphs[0], 'Đánh giá', 17, GREEN, bold=True, space_after=8)
for txt2 in [
    'Nhị phân: AUC-ROC + CI DeLong 95%',
    'Sống còn (PFS): Cox HR, time-dependent AUC, C-index, IBS, Kaplan–Meier log-rank',
]:
    p = tf2.add_paragraph(); setpara(p, '• ' + txt2, 15, SLATE, space_after=7)
block(s, Inches(0.6), Inches(5.5), Inches(12.2), Inches(1.2),
      'Điểm nhấn',
      'Đánh giá cả PREDICTIVE (đáp ứng) lẫn PROGNOSTIC (sống còn) → về sau cho một phát hiện thú vị.',
      GREEN, GREEN_L)

# ═══════════════════════════════════════════════════════════════════════════
# Slide 11 — KQ1
s = slide(); title_bar(s, 'KQ1: Đa phương thức vượt đơn phương thức'); footer(s, 11)
tb, tf = textbox(s, Inches(0.6), Inches(1.3), Inches(12.1), Inches(2.4))
setpara(tf.paragraphs[0], 'Đơn phương thức (logistic regression):', 18, SLATE,
        bold=True, space_after=8)
setpara(tf.add_paragraph(),
        '   AUC  0,570 (lâm sàng)  •  0,640 (radiomics)  •  0,650 (genomics)  •  0,729 (PD-L1)',
        17, GRAY, space_after=0)
block(s, Inches(0.6), Inches(4.2), Inches(12.1), Inches(2.0),
      'DyAM tích hợp 4 phương thức:  AUC = 0,784  [0,717–0,850]',
      'Tích hợp đa phương thức là ĐỘNG LỰC CHÍNH của hiệu năng, vượt xa mọi phương thức đơn lẻ.',
      ORANGE, ORANGE_L)

# ═══════════════════════════════════════════════════════════════════════════
# Slide 12 — KQ2 OvO
s = slide(); title_bar(s, 'KQ2: OvO attention — lợi ích phụ thuộc ngữ cảnh'); footer(s, 12)
tb, tf = textbox(s, Inches(0.6), Inches(1.2), Inches(12.1), Inches(2.8))
bullets(tf, [
    ('20 tổ hợp: OvO thắng 7, Original thắng 9, hòa 4.', 0, SLATE, False),
    ('2 phương thức: OvO tốt hơn 3/6 — lớn nhất PDL1+Gen +3,27%, Rad+Gen +2,23%.', 0, SLATE, False),
    ('≥3 phương thức: Original tốt hơn 6/10 (nhất là khi có IHC-G / lâm sàng).', 0, SLATE, False),
], size=18)
block(s, Inches(0.6), Inches(4.4), Inches(12.1), Inches(2.0),
      'Cấu hình tốt nhất toàn bộ:  OvO Rad+IHC-G+Gen+PDL1 — AUC = 0,800  [0,739–0,862]',
      'OvO hợp với phương thức TRÙNG LẶP (chọn "best-of"); cooperative hợp khi nhiều phương thức BỔ SUNG.',
      GREEN, GREEN_L)

# ═══════════════════════════════════════════════════════════════════════════
# Slide 13 — KQ3 NLP AUC (fig3a/b)
s = slide(); title_bar(s, 'KQ3: NLP cải thiện AUC ổn định'); footer(s, 13)
image(s, 'fig3a_auc_ihca_p.png', Inches(1.0), Inches(1.15), height=Inches(4.0))
image(s, 'fig3b_auc_ihcg_p.png', Inches(7.3), Inches(1.15), height=Inches(4.0))
tb, tf = textbox(s, Inches(0.6), Inches(5.4), Inches(12.2), Inches(1.5))
bullets(tf, [
    ('IHC-A: NLP-PCA16 tốt nhất (AUC 0,784; Δ=+0,020) vs +0,004 cho số thô.', 0, SLATE, False),
    ('IHC-G: NLP raw tốt nhất (AUC 0,813; Δ=+0,030). Cải thiện nhất quán, nhưng CI còn chồng lấn.', 0, SLATE, False),
    ('BioClinBERT (chuyên y khoa) kém hơn MiniLM tổng quát.', 0, SLATE, False),
], size=13)

# ═══════════════════════════════════════════════════════════════════════════
# Slide 14 — Vì sao thực hiện phân tích sống còn? (động lực)
s = slide(); title_bar(s, 'Vì sao thực hiện phân tích sống còn?'); footer(s, 14)
block(s, Inches(0.5), Inches(1.1), Inches(12.3), Inches(1.78),
      '1. Bài toán nhị phân bị đặt sai (góc nhìn lâm sàng)',
      'Phân loại chỉ hỏi "có đáp ứng không?" — nhưng SD ở tháng 3 KHÁC HẲN SD ở tháng 12. '
      'Câu hỏi thật của bác sĩ là "đáp ứng được BAO LÂU?". Vì vậy thử nghiệm ung thư dùng '
      'PFS/OS làm primary endpoint, không phải tỉ lệ đáp ứng.',
      GREEN, GREEN_L)
block(s, Inches(0.5), Inches(3.0), Inches(12.3), Inches(1.95),
      '2. Giả thuyết trung tâm — vì sao AUC nhị phân "bỏ lỡ" tín hiệu NLP',
      'Embedding đặt các bệnh nhân hồ sơ giống nhau (già, albumin thấp, di căn gan…) gần nhau '
      '— nhóm SỐNG NGẮN HƠN nhưng chưa chắc SD/PD nhiều hơn ở n=247. ⇒ NLP mã hóa thông tin '
      '"bao lâu" (time-to-event) tốt hơn "có/không" (binary) ⇒ AUC nhị phân cải thiện ít.',
      PURPLE, PURPLE_L)
block(s, Inches(0.5), Inches(5.07), Inches(12.3), Inches(1.6),
      '3. Lợi thế thống kê (quan trọng với n = 247)',
      'AUC nhị phân: mỗi BN = 1 nhãn → 247 đơn vị thông tin. C-index sống còn: so trên MỌI CẶP '
      'bệnh nhân ≈ 209 × 38 ≈ 7.900 cặp → power cao hơn nhiều trên cùng cohort.',
      ORANGE, ORANGE_L)

# ═══════════════════════════════════════════════════════════════════════════
# Slide 15 — KQ4 sống còn (fig5c/b)
s = slide(); title_bar(s, 'KQ4: Phân tích sống còn — tín hiệu mạnh nhất'); footer(s, 15)
image(s, 'fig5c_cox_forest_p.png', Inches(0.45), Inches(1.35), height=Inches(3.5))
image(s, 'fig5b_tdauc_p.png', Inches(6.05), Inches(1.35), height=Inches(3.5))
tb, tf = textbox(s, Inches(0.6), Inches(5.25), Inches(12.2), Inches(1.5))
bullets(tf, [
    ('Risk score DyAM: yếu tố tiên lượng độc lập (Cox p<0,001, cả 8 biến thể).', 0, SLATE, False),
    ('HR cao nhất luôn thuộc NLP: 6,07 (IHC-A) / 4,97 (IHC-G) — số thô KHÔNG cải thiện HR.', 0, ORANGE, True),
    ('Lợi thế NLP tập trung ở 12–18 tháng; Kaplan–Meier tất cả p<0,005.', 0, SLATE, False),
], size=13)

# ═══════════════════════════════════════════════════════════════════════════
# Slide 16 — Bảng kết quả survival (Table 4 của bài báo + tdAUC)
def survival_table(s, left, top, width, height):
    gs = s.shapes.add_table(11, 5, int(left), int(top), int(width), int(height))
    tbl = gs.table
    tbl.first_row = False
    tbl.horz_banding = False
    for c, wv in enumerate([3.7, 2.15, 2.15, 2.15, 2.15]):
        tbl.columns[c].width = Inches(wv)

    header = ['Mô hình', 'C-index', 'Cox HR', 'tdAUC 12m', 'IBS ↓']
    arms = {1: 'Nhánh IHC-A', 6: 'Nhánh IHC-G'}
    data = {
        2: ('No clinical', '0,623', '5,30', '0,718', '0,1745', set()),
        3: ('+ Labs',      '0,626', '5,31', '0,707', '0,1739', set()),
        4: ('+ NLP raw',   '0,625', '5,91', '0,739', '0,1723', set()),
        5: ('+ NLP-PCA16', '0,628', '6,07', '0,742', '0,1716', {0, 1, 2, 3, 4}),
        7: ('No clinical', '0,626', '4,74', '0,691', '0,1748', set()),
        8: ('+ Labs',      '0,632', '4,49', '0,686', '0,1746', {1}),
        9: ('+ NLP raw',   '0,632', '4,97', '0,699', '0,1736', {0, 1, 2, 3, 4}),
        10: ('+ NLP-PCA16','0,631', '4,69', '0,694', '0,1736', set()),
    }

    def setcell(cell, text, size=10.5, color=SLATE, bold=False, italic=False,
                bg=WHITE, align=PP_ALIGN.CENTER):
        cell.fill.solid(); cell.fill.fore_color.rgb = bg
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_left = Pt(5); cell.margin_right = Pt(4)
        cell.margin_top = Pt(1); cell.margin_bottom = Pt(1)
        p = cell.text_frame.paragraphs[0]; p.text = text; p.alignment = align
        r = p.runs[0]; r.font.size = Pt(size); r.font.name = FONT
        r.font.bold = bold; r.font.italic = italic; r.font.color.rgb = color

    for c, h in enumerate(header):
        setcell(tbl.cell(0, c), h, size=11, color=WHITE, bold=True, bg=GREEN,
                align=(PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER))
    for r, name in arms.items():
        tbl.cell(r, 0).merge(tbl.cell(r, 4))
        setcell(tbl.cell(r, 0), name, color=GREEN, bold=True, italic=True,
                bg=GREEN_L, align=PP_ALIGN.LEFT)
    for r, vals in data.items():
        bolds = vals[5]
        setcell(tbl.cell(r, 0), vals[0], color=(GREEN if 0 in bolds else SLATE),
                bold=(0 in bolds), align=PP_ALIGN.LEFT)
        for ci, val in enumerate(vals[1:5], start=1):
            b = ci in bolds
            setcell(tbl.cell(r, ci), val, color=(GREEN if b else SLATE), bold=b)
    for r in range(11):
        tbl.rows[r].height = Inches(0.31)


s = slide(); title_bar(s, 'Bảng kết quả phân tích sống còn (8 mô hình)')
footer(s, 16)
survival_table(s, Inches(0.5), Inches(1.05), Inches(12.3), Inches(3.45))
tb, tf = textbox(s, Inches(0.55), Inches(4.85), Inches(6.3), Inches(1.4))
setpara(tf.paragraphs[0],
        'C-index: ≈0,62–0,63 (đều ≫0,5) — nền mạnh; NLP nhỉnh nhẹ, chưa có ý nghĩa.',
        12, SLATE, space_after=9)
setpara(tf.add_paragraph(),
        'Cox HR: NLP cao nhất (6,07/4,97); số thô KHÔNG tăng → tín hiệu mạnh nhất.',
        12, ORANGE, bold=True, space_after=0)
tb2, tf2 = textbox(s, Inches(7.0), Inches(4.85), Inches(5.9), Inches(1.4))
setpara(tf2.paragraphs[0],
        'tdAUC 12m: NLP tốt nhất cả 2 nhánh; +Labs thấp nhất.',
        12, SLATE, space_after=9)
setpara(tf2.add_paragraph(),
        'IBS: đều <0,25 (hiệu chuẩn tốt); NLP-PCA16 thấp nhất.',
        12, GREEN, bold=True, space_after=0)
tb3, tf3 = textbox(s, Inches(0.55), Inches(6.4), Inches(12.3), Inches(0.5))
setpara(tf3.paragraphs[0],
        'Cox đa biến hiệu chỉnh tuổi/ECOG/albumin/dNLR/di căn gan, mọi p<0,001. '
        'In đậm = tốt nhất mỗi cột trong từng nhánh. IBS càng thấp càng tốt.',
        10, GRAY, italic=True, space_after=0)

# ═══════════════════════════════════════════════════════════════════════════
# Slide 17 — Công thức chỉ số sống còn (1): phân biệt
f_cindex = render_math(
    r'C=\frac{n_{\mathrm{concordant}}}{n_{\mathrm{comparable}}}'
    r'=P(\hat{r}_i>\hat{r}_j \mid T_i<T_j)')
f_tdauc = render_math(
    r'\mathrm{AUC}(t)=P(\hat{r}_i>\hat{r}_j \mid T_i\leq t<T_j)')
s = slide(); title_bar(s, 'Chỉ số sống còn (1): Phân biệt — công thức & ý nghĩa')
footer(s, 17)
metric_card(s, Inches(0.5), Inches(1.15), Inches(12.3), Inches(2.35),
            GREEN, GREEN_L,
            'Harrell’s C-index — "AUC cho dữ liệu sống còn"  (lifelines)',
            f_cindex,
            'Xét mọi CẶP bệnh nhân so sánh được → tỉ lệ model xếp ĐÚNG thứ tự nguy cơ '
            '(event sớm hơn ⇒ risk cao hơn). 0,5 = ngẫu nhiên; 1,0 = hoàn hảo. CI 95%: bootstrap 1000 lần.',
            Inches(0.62))
metric_card(s, Inches(0.5), Inches(3.65), Inches(12.3), Inches(2.35),
            GREEN, GREEN_L,
            'Time-dependent AUC — phân biệt tại từng mốc t  (scikit-survival)',
            f_tdauc,
            'Tại t = 6/12/18 tháng: "case" = đã có event trước t, "control" = còn lại sau t; '
            'hiệu chỉnh kiểm duyệt bằng IPCW. Cho biết model mạnh ở NGẮN HẠN hay DÀI HẠN.',
            Inches(0.55))
tb, tf = textbox(s, Inches(0.55), Inches(6.2), Inches(12.2), Inches(0.6))
setpara(tf.paragraphs[0],
        'Trong bài: ΔC = +0,005 (nhất quán, chưa có ý nghĩa);  td-AUC: +0,024 @12m, +0,020 @18m.',
        13, ORANGE, italic=True, space_after=0)

# ═══════════════════════════════════════════════════════════════════════════
# Slide 16 — Công thức chỉ số sống còn (2): hồi quy & hiệu chuẩn
f_cox = render_math(
    r'h(t\mid\mathbf{x})=h_0(t)\,e^{\beta_1 x_1+\cdots+\beta_p x_p},'
    r'\quad \mathrm{HR}_{\mathrm{score}}=e^{\beta_{\mathrm{score}}}')
f_ibs = render_math(
    r'\mathrm{BS}(t)=\frac{1}{N}\sum_i w_i\,(\hat{S}(t\mid\mathbf{x}_i)-'
    r'\mathbf{1}\{T_i>t\})^2,\quad \mathrm{IBS}=\frac{1}{t_{\max}-t_{\min}}\int \mathrm{BS}(t)\,dt')
f_km = render_math(
    r'\hat{S}(t)=\prod_{t_i\leq t}\left(1-\frac{d_i}{n_i}\right),\quad '
    r'\chi^2_{\mathrm{logrank}}=\frac{(\sum_k(O_k-E_k))^2}{\sum_k \mathrm{Var}(O_k)}')
s = slide(); title_bar(s, 'Chỉ số sống còn (2): Hồi quy nguy cơ & hiệu chuẩn')
footer(s, 18)
metric_card(s, Inches(0.5), Inches(1.05), Inches(12.3), Inches(1.72),
            PURPLE, PURPLE_L,
            'Cox proportional hazards — Hazard Ratio (HR)',
            f_cox,
            'Đa biến: score (scale [0,1]) + tuổi, ECOG, albumin, dNLR, di căn gan. '
            'HR>1 ⇒ nguy cơ cao hơn; p (Wald) đo giá trị tiên lượng ĐỘC LẬP.  HR = 6,07 vs 5,30.',
            Inches(0.46), title_size=13, meaning_size=11)
metric_card(s, Inches(0.5), Inches(2.92), Inches(12.3), Inches(1.78),
            ORANGE, ORANGE_L,
            'Integrated Brier Score (IBS) — hiệu chuẩn',
            f_ibs,
            'Sai số bình phương giữa XÁC SUẤT SỐNG dự đoán và thực tế (IPCW). '
            'Càng thấp càng tốt; 0,25 = ngẫu nhiên.  IBS = 0,172–0,175.',
            Inches(0.5), title_size=13, meaning_size=11)
metric_card(s, Inches(0.5), Inches(4.85), Inches(12.3), Inches(1.72),
            GREEN, GREEN_L,
            'Kaplan–Meier + log-rank — phân tầng nguy cơ',
            f_km,
            'Chia 2 nhóm tại score = 0; log-rank so số event quan sát O_k vs kỳ vọng E_k. '
            'χ² lớn / p nhỏ ⇒ tách nhóm rõ.  Mọi p<0,005; χ² cao nhất = 28,92.',
            Inches(0.5), title_size=13, meaning_size=11)

# ═══════════════════════════════════════════════════════════════════════════
# Slide 19 — Bàn luận
s = slide(); title_bar(s, 'Bàn luận'); footer(s, 19)
block(s, Inches(0.6), Inches(1.25), Inches(12.1), Inches(1.55),
      'Vì sao NLP > số thô?',
      'Sentence embedding nắm bắt TƯƠNG TÁC PHI TUYẾN giữa các biến; bệnh nhân hồ sơ giống nhau '
      '→ vector gần nhau. Số thô buộc mô hình tự học từ tập nhỏ.', PURPLE, PURPLE_L)
block(s, Inches(0.6), Inches(2.95), Inches(12.1), Inches(1.75),
      'Vì sao chưa đạt ý nghĩa thống kê?',
      'Giới hạn CỠ MẪU: n=247 → CI rộng ≈ ±0,07; cần ~1.200–1.500 BN để phát hiện ΔAUC≈0,02 '
      '(power 80%). Hiệu ứng nhất quán về hướng → thật nhưng thiếu power.', ORANGE, ORANGE_L)
block(s, Inches(0.6), Inches(4.85), Inches(12.1), Inches(1.45),
      'Khi nào dùng OvO?',
      'Phương thức TRÙNG LẶP (ít phương thức) → OvO; nhiều phương thức BỔ SUNG → cooperative.',
      GREEN, GREEN_L)

# ═══════════════════════════════════════════════════════════════════════════
# Slide 20 — Hạn chế
s = slide(); title_bar(s, 'Hạn chế'); footer(s, 20)
tb, tf = textbox(s, Inches(0.6), Inches(1.3), Inches(12.1), Inches(4.0))
bullets(tf, [
    ('Cohort đơn trung tâm, hồi cứu → có thể sai lệch chọn mẫu, hạn chế khái quát hóa.', 0, SLATE, False),
    ('Cỡ mẫu nhỏ (n=247) → cải thiện AUC chưa đạt ý nghĩa thống kê.', 0, SLATE, False),
    ('Sentence encoder dùng zero-shot, chưa fine-tune cho thuật ngữ NSCLC.', 0, SLATE, False),
    ('PD-L1 chấm theo phương pháp Sauter → có thể khác assay SP142/22C3 ở nơi khác.', 0, SLATE, False),
], size=18)
tb2, tf2 = textbox(s, Inches(0.6), Inches(5.8), Inches(12.1), Inches(0.6))
setpara(tf2.paragraphs[0],
        '⇒ Định hướng: validation trên cohort lớn, đa trung tâm.', 15, GRAY,
        italic=True, space_after=0)

# ═══════════════════════════════════════════════════════════════════════════
# Slide 21 — Kết luận
s = slide(); title_bar(s, 'Kết luận'); footer(s, 21)
tb, tf = textbox(s, Inches(0.6), Inches(1.2), Inches(12.1), Inches(3.4))
bullets(tf, [
    ('Mở rộng DyAM với 2 đóng góp: OvO attention + NLP encoding.', 0, SLATE, True),
    ('OvO: tốt nhất ở cấu hình ít phương thức; cho mô hình tốt nhất toàn bộ (AUC 0,800).', 0, SLATE, False),
    ('NLP: cải thiện AUC ổn định (tới +0,030) VÀ giá trị tiên lượng độc lập mạnh nhất trong sống còn.', 0, SLATE, False),
    ('Thực tiễn: không cần fine-tune; áp dụng cho mọi trường EHR có cấu trúc.', 0, SLATE, False),
], size=17)
block(s, Inches(0.6), Inches(5.0), Inches(12.1), Inches(1.5),
      'Thông điệp',
      'Competitive attention + NLP encoding là chiến lược THỰC TẾ, KHÁI QUÁT HÓA ĐƯỢC cho khám phá '
      'biomarker đa phương thức. Cần validation đa trung tâm để khẳng định.', GREEN, GREEN_L)

# ═══════════════════════════════════════════════════════════════════════════
# Slide 18 — Cảm ơn
s = slide(); rect(s, 0, 0, SW, SH, WHITE)
tb, tf = textbox(s, Inches(0.6), Inches(2.6), Inches(12.1), Inches(2.4),
                 anchor=MSO_ANCHOR.MIDDLE)
setpara(tf.paragraphs[0], 'Xin cảm ơn thầy cô và các bạn đã lắng nghe!',
        30, GREEN, bold=True, align=PP_ALIGN.CENTER, space_after=14)
setpara(tf.add_paragraph(), 'Rất mong nhận được câu hỏi và góp ý.',
        18, SLATE, align=PP_ALIGN.CENTER, space_after=20)
setpara(tf.add_paragraph(),
        'Số liệu chủ chốt:  AUC tốt nhất 0,813 (NLP raw, IHC-G)  •  0,800 (OvO)  •  '
        'HR 6,07 (NLP-PCA16)  •  n = 247',
        13, GRAY, italic=True, align=PP_ALIGN.CENTER, space_after=0)

out = os.path.join(os.path.dirname(__file__), 'slides.pptx')
try:
    prs.save(out)
    print('Saved:', out, '|', len(prs.slides._sldIdLst), 'slides')
except PermissionError:
    alt = os.path.join(os.path.dirname(__file__), 'slides_NEW.pptx')
    prs.save(alt)
    print('!! slides.pptx đang bị khóa (mở trong PowerPoint?). Đã lưu sang:', alt)
    print('   -> Đóng PowerPoint rồi đổi tên slides_NEW.pptx thành slides.pptx.')
