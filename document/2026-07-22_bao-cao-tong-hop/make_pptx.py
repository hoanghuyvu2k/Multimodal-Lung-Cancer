# -*- coding: utf-8 -*-
"""Dựng slide tổng hợp thực nghiệm (.pptx)."""
from pptx import Presentation
from pptx.util import Inches as In, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

C = dict(
    accent=RGBColor(0x1F,0x6F,0x83), ink=RGBColor(0x16,0x21,0x2B),
    muted=RGBColor(0x58,0x68,0x78), faint=RGBColor(0x85,0x95,0xA3),
    neg=RGBColor(0xB3,0x60,0x3A), pos=RGBColor(0x2F,0x7D,0x5B), warn=RGBColor(0xA8,0x79,0x2A),
    line=RGBColor(0xE0,0xE6,0xEC), surf=RGBColor(0xF4,0xF6,0xF8), white=RGBColor(0xFF,0xFF,0xFF),
    accent_soft=RGBColor(0xE3,0xEE,0xF1), neg_soft=RGBColor(0xF5,0xE7,0xDF),
    pos_soft=RGBColor(0xE2,0xEF,0xE9), inkbg=RGBColor(0x11,0x1B,0x24),
)
FONT="Segoe UI"
prs=Presentation(); prs.slide_width=In(13.333); prs.slide_height=In(7.5)
BLANK=prs.slide_layouts[6]
SW,SH=13.333,7.5

def slide(dark_title=False):
    s=prs.slides.add_slide(BLANK)
    bg=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,0,0,prs.slide_width,prs.slide_height)
    bg.fill.solid(); bg.fill.fore_color.rgb=C['inkbg'] if dark_title else C['white']
    bg.line.fill.background(); bg.shadow.inherit=False
    s.shapes._spTree.remove(bg._element); s.shapes._spTree.insert(2,bg._element)
    return s

def _set(p,text,size,color,bold=False,italic=False,spacing=None,font=FONT):
    r=p.add_run(); r.text=text; f=r.font
    f.size=Pt(size); f.bold=bold; f.italic=italic; f.name=font; f.color.rgb=color
    if spacing is not None:
        rPr=r._r.get_or_add_rPr(); rPr.set('spc',str(int(spacing*100)))
    return r

def box(s,l,t,w,h,anchor=MSO_ANCHOR.TOP):
    tb=s.shapes.add_textbox(In(l),In(t),In(w),In(h)); tf=tb.text_frame
    tf.word_wrap=True; tf.vertical_anchor=anchor
    tf.margin_left=0; tf.margin_right=0; tf.margin_top=0; tf.margin_bottom=0
    return tf

def para(tf,first=False,space_after=4,space_before=0,line=1.05):
    p=tf.paragraphs[0] if first else tf.add_paragraph()
    p.space_after=Pt(space_after); p.space_before=Pt(space_before); p.line_spacing=line
    return p

def rect(s,l,t,w,h,fill,line_c=None,line_w=None,round_=False):
    shp=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if round_ else MSO_SHAPE.RECTANGLE,
                           In(l),In(t),In(w),In(h))
    shp.fill.solid(); shp.fill.fore_color.rgb=fill; shp.shadow.inherit=False
    if line_c is None: shp.line.fill.background()
    else: shp.line.color.rgb=line_c; shp.line.width=Pt(line_w or 1)
    if round_:
        try: shp.adjustments[0]=0.08
        except Exception: pass
    return shp

def eyebrow(s,txt,idx=None):
    tf=box(s,0.9,0.62,11.5,0.4); p=para(tf,True)
    if idx: _set(p,idx+"  ",12,C['faint'],bold=True,spacing=1.4)
    _set(p,txt.upper(),12,C['accent'],bold=True,spacing=1.4)

def title(s,txt,size=30,color=None):
    tf=box(s,0.9,1.02,11.5,1.4); p=para(tf,True,line=1.02); _set(p,txt,size,color or C['ink'],bold=True)

def chip(s,l,t,txt,kind='par'):
    fills={'neg':(C['neg_soft'],C['neg']),'pos':(C['pos_soft'],C['pos']),
           'acc':(C['accent_soft'],C['accent']),'par':(C['surf'],C['muted'])}
    bg,fg=fills[kind]
    w=0.13*len(txt)+0.5
    r=rect(s,l,t,w,0.34,bg,line_c=fg,line_w=0.75,round_=True)
    tf=r.text_frame; tf.word_wrap=False
    tf.margin_left=In(0.12); tf.margin_right=In(0.12); tf.margin_top=0; tf.margin_bottom=0
    p=tf.paragraphs[0]; p.alignment=PP_ALIGN.CENTER; _set(p,txt,11.5,fg,bold=True)
    return w

def takeaway(s,label,parts,top=6.35,width=11.53):
    rect(s,0.9,top,width,0.82,C['surf'])
    rect(s,0.9,top,0.06,0.82,C['accent'])
    tf=box(s,1.15,top+0.12,width-0.4,0.62,MSO_ANCHOR.MIDDLE)
    p=para(tf,True,line=1.08); _set(p,label.upper()+"   ",10.5,C['accent'],bold=True,spacing=1.2)
    for txt,bold,col in parts: _set(p,txt,13,col or C['ink'],bold=bold)

def bullets(tf,items,size=13,gap=6):
    for i,(txt,bold) in enumerate(items):
        p=para(tf,i==0,space_after=gap,line=1.12)
        _set(p,"•  ",size,C['accent'],bold=True)
        if isinstance(txt,list):
            for t,b in txt: _set(p,t,size,C['ink'],bold=b)
        else: _set(p,txt,size,C['ink'],bold=bold)

def table(s,l,t,w,rows,colw,header,cell_h=0.34,hdr_h=0.34,fs=12):
    n=len(rows)+1
    gt=s.shapes.add_table(n,len(colw),In(l),In(t),In(w),In(hdr_h+cell_h*len(rows)))
    tb=gt.table
    # remove default style banding
    tbl=tb._tbl;
    for el in tbl.findall(qn('a:tblPr')): el.set('firstRow','0'); el.set('bandRow','0')
    for j,cw in enumerate(colw): tb.columns[j].width=In(cw)
    tb.rows[0].height=In(hdr_h)
    for j,htxt in enumerate(header):
        c=tb.cell(0,j); c.fill.solid(); c.fill.fore_color.rgb=C['surf']
        c.margin_left=In(0.08);c.margin_right=In(0.08);c.margin_top=In(0.02);c.margin_bottom=In(0.02)
        c.vertical_anchor=MSO_ANCHOR.MIDDLE
        p=c.text_frame.paragraphs[0]; p.alignment=PP_ALIGN.RIGHT if j>0 else PP_ALIGN.LEFT
        _set(p,htxt.upper(),10,C['muted'],bold=True,spacing=.4)
    for i,row in enumerate(rows,1):
        tb.rows[i].height=In(cell_h)
        rowfill=row[-1] if isinstance(row[-1],RGBColor) else None
        cells=row[:-1] if rowfill else row
        for j,val in enumerate(cells):
            c=tb.cell(i,j); c.fill.solid()
            c.fill.fore_color.rgb=rowfill if rowfill else C['white']
            c.margin_left=In(0.08);c.margin_right=In(0.08);c.margin_top=In(0.01);c.margin_bottom=In(0.01)
            c.vertical_anchor=MSO_ANCHOR.MIDDLE
            txt,col,bold=val if isinstance(val,tuple) else (val,C['ink'],False)
            p=c.text_frame.paragraphs[0]; p.alignment=PP_ALIGN.RIGHT if j>0 else PP_ALIGN.LEFT
            _set(p,txt,fs,col,bold=bold)
    return gt

def bar(s,l,t,w,label,frac,val,color):
    tf=box(s,l,t-0.02,2.0,0.3,MSO_ANCHOR.MIDDLE); p=para(tf,True); _set(p,label,12,C['muted'])
    track_l=l+2.05; track_w=w-2.05-0.9
    rect(s,track_l,t,track_w,0.22,C['surf'])
    rect(s,track_l,t,max(0.03,track_w*frac),0.22,color)
    tf2=box(s,track_l+track_w+0.05,t-0.02,0.85,0.3,MSO_ANCHOR.MIDDLE); p2=para(tf2,True)
    p2.alignment=PP_ALIGN.RIGHT; _set(p2,val,12,C['ink'],bold=True)

def pbox(s,l,t,w,h,ttl,sub=None,fill=None,tcol=None,border=None):
    r=rect(s,l,t,w,h,fill or C['surf'],line_c=border or C['line'],line_w=1.25,round_=True)
    tf=r.text_frame; tf.word_wrap=True; tf.vertical_anchor=MSO_ANCHOR.MIDDLE
    tf.margin_left=In(0.1); tf.margin_right=In(0.1)
    p=tf.paragraphs[0]; p.alignment=PP_ALIGN.CENTER; _set(p,ttl,13,tcol or C['ink'],bold=True)
    if sub:
        p2=tf.add_paragraph(); p2.alignment=PP_ALIGN.CENTER; p2.space_before=Pt(3); p2.line_spacing=1.0
        _set(p2,sub,11,C['muted'])
    return r

def arrow(s,l,t,w=0.42):
    a=s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,In(l),In(t),In(w),In(0.26))
    a.fill.solid(); a.fill.fore_color.rgb=C['faint']; a.line.fill.background(); a.shadow.inherit=False
    try: a.adjustments[0]=0.55; a.adjustments[1]=0.5
    except Exception: pass

# ---------------- SLIDE 1: TITLE ----------------
s=slide()
rect(s,0,0,SW,0.14,C['accent'])
tf=box(s,0.9,1.5,11.5,0.4); p=para(tf,True); _set(p,"BÁO CÁO NỘI BỘ · 21–22 / 07 / 2026",12,C['accent'],bold=True,spacing=1.5)
tf=box(s,0.9,2.0,11.6,2.0); p=para(tf,True,line=1.05)
_set(p,"Không mô hình hay đặc trưng phức tạp nào vượt được baseline đơn giản",34,C['ink'],bold=True)
tf=box(s,0.9,3.9,11.2,1.1); p=para(tf,True,line=1.3)
_set(p,"Kiểm chứng hệ thống: fusion attention, feature engineering, foundation embedding (ảnh gốc) và tabular foundation model — trên cohort đa nguồn dự đoán đáp ứng miễn dịch NSCLC.",15,C['muted'])
# KPI row
kpis=[("6","nhóm thí nghiệm",C['accent']),("~0.78","trần AUC nội bộ",C['ink']),
      ("1","phát hiện dương (pathology generalize)",C['pos']),("0","phương pháp vượt baseline",C['neg'])]
kx=0.9
for n,c,col in kpis:
    rect(s,kx,5.25,2.75,1.35,C['surf'],line_c=C['line'],line_w=1,round_=True)
    tf=box(s,kx+0.22,5.45,2.4,0.7); p=para(tf,True); _set(p,n,30,col,bold=True)
    tf=box(s,kx+0.22,6.12,2.45,0.55); p=para(tf,True,line=1.0); _set(p,c,11.5,C['muted'])
    kx+=2.9

# ---------------- SLIDE 2: CONTEXT ----------------
s=slide(); eyebrow(s,"Bối cảnh & câu hỏi"); title(s,"Bài toán và cách đo",28)
tf=box(s,0.9,1.95,5.6,4); p=para(tf,True,line=1.25)
_set(p,"Dự đoán đáp ứng liệu pháp PD-(L)1 ở NSCLC bằng cách kết hợp nhiều nguồn: radiomics (CT), pathology (PD-L1 IHC), genomic, điểm PD-L1, xét nghiệm lâm sàng. Mô hình gốc: ",13,C['ink'])
_set(p,"DyAM",13,C['ink'],bold=True); _set(p," (dynamic attention fusion).",13,C['ink'])
p=para(tf,space_before=10); _set(p,"Ba câu hỏi kiểm chứng",14,C['accent'],bold=True)
bl=box(s,0.9,3.35,5.6,2.6)
bullets(bl,[("Attention học của DyAM có hơn trung bình đơn giản?",False),
            ("Foundation embedding từ ảnh gốc có phá được trần / rào generalization?",False),
            ("TabPFN (model hiện đại) có hơn hồi quy tuyến tính trên dữ liệu nhỏ?",False)])
tf=box(s,6.9,1.95,5.5,0.5); p=para(tf,True); _set(p,"Quy tắc đánh giá (chống tự huyễn)",14,C['accent'],bold=True)
bl=box(s,6.9,2.5,5.5,2.6)
bullets(bl,[([("5-seed × 10-fold",True),(" hoán vị phân hoạch — chống cherry-pick 1-lần.",False)],False),
            ([("Paired bootstrap",True),(" trên bệnh nhân để so có ý nghĩa.",False)],False),
            ([("External validation",True),(" trên cohort ngoài — thước đo quan trọng nhất.",False)],False)])
takeaway(s,"Lưu ý",[("AUC nội bộ đã kịch trần ~0.78 từ trước → trọng tâm là ",False,None),
                    ("generalization & không thua baseline",True,None),(", không đua số nội bộ.",False,None)])

# ---------------- SLIDE 2b: MÔ HÌNH uniform_avg ----------------
s=slide(); eyebrow(s,"Mô hình baseline — thứ mọi phương pháp phải vượt")
title(s,"uniform_avg: trung bình có-mask, không attention",25)
IMG=r"D:\code\master\doan\code\paper\figures\fig_uniform_arch.png"
iw=10.6; ih=iw*(4.95/13.6); il=(SW-iw)/2; itop=2.15
s.shapes.add_picture(IMG,In(il),In(itop),In(iw),In(ih))
takeaway(s,"Vì sao quan trọng",[("Bỏ hoàn toàn attention học mà AUC KHÔNG giảm (còn nhỉnh) → ",False,None),
                               ("attention của DyAM không đóng góp",True,None),(". Baseline này là mốc để đánh giá mọi thứ phức tạp hơn.",False,None)])

# ---------------- SLIDE 2c: MÔ HÌNH DyAM + OvO ----------------
s=slide(); eyebrow(s,"Hai biến thể attention HỌC được — mô hình gốc DyAM")
title(s,"DyAM: Cooperative (A)  &  Competitive / OvO (B)",23)
IMG2=r"D:\code\master\doan\code\paper\figures\fig_attention_arch.png"
iw=8.75; ih=iw*(1665.0/2669.0); il=(SW-iw)/2; itop=1.6
s.shapes.add_picture(IMG2,In(il),In(itop),In(iw),In(ih))
tf=box(s,0.9,7.02,11.53,0.38); p=para(tf,True); p.alignment=PP_ALIGN.CENTER
_set(p,"Cả hai HỌC trọng số aᵢ (khác uniform_avg chia đều 1/Σmask) — nhưng thực nghiệm cho thấy không hơn baseline.",11.5,C['muted'])

# ---------------- SLIDE 3: EXP A ----------------
s=slide(); eyebrow(s,"Fusion attention vs trung bình đơn giản","A")
title(s,"uniform_avg ≥ DyAM/OvO trên mọi tổ hợp nguồn",26)
chip(s,0.9,1.75,"Luận điểm “DyAM tốt hơn khi nhiều nguồn”: BÁC BỎ",'neg')
chip(s,6.4,1.75,"26 + 21 tổ hợp · 5-seed",'par')
tf=box(s,0.9,2.4,5.6,0.7); p=para(tf,True,line=1.15)
_set(p,"Δ = uniform − DyAM ",13,C['ink'],bold=True)
_set(p,"tăng theo số nguồn k — ngược kỳ vọng “nhiều nguồn thì attention thắng”.",13,C['ink'])
bar(s,0.9,3.25,5.6,"k = 2 nguồn",0.10,"+0.001",C['accent'])
bar(s,0.9,3.70,5.6,"k = 3 nguồn",0.55,"+0.007",C['accent'])
bar(s,0.9,4.15,5.6,"k = 4 nguồn",0.90,"+0.011",C['accent'])
tf=box(s,0.9,4.65,5.6,1.4); p=para(tf,True,line=1.2)
_set(p,"DyAM chỉ thắng có ý nghĩa ở 2/26 combo — đều tại k=2. OvO gần như trùng uniform_avg (±0.005).",12,C['muted'])
table(s,6.9,2.55,5.5,
      [[("uniform_avg (baseline)",C['ink'],True),("0.7746",C['accent'],True),("0.7850",C['accent'],True),C['accent_soft']],
       ["OvO attention","0.7728","0.7843"],
       ["DyAM attention",("0.7595",C['neg'],False),("0.7752",C['neg'],False)]],
      [2.9,1.3,1.3],["Model","5-seed","ens"],cell_h=0.42,hdr_h=0.36)
tf=box(s,6.9,4.6,5.5,0.5); p=para(tf,True); _set(p,"AUC nội bộ tại combo tốt nhất — BM1",11,C['faint'])
takeaway(s,"Kết luận",[("Attention học không đóng góp — nó tái lập phép chia đều 1/Σmask. Càng nhiều nguồn, baseline càng vượt.",False,None)])

# ---------------- SLIDE 3b: TỔNG QUAN 3 MODEL ----------------
s=slide(); eyebrow(s,"Tổng quan — 3 mô hình fusion")
title(s,"uniform_avg · DyAM · OvO — nhìn tổng thể",25)
A=lambda v:(v,C['accent'],True)   # cao nhất nhóm
table(s,0.82,1.95,11.68,
  [["BM4 · PDL1+Gen",       A("0.7191"),"0.7072","0.7183",  "0.7124","0.6932",A("0.7157")],
   ["BM3 · Rad+Gen",        "0.7110","0.7095",A("0.7132"),  "0.7472",A("0.7549"),"0.7511"],
   ["BM1 · Rad+Path+Gen+PDL1",A("0.7746"),"0.7595","0.7728",  "0.7915",A("0.7976"),"0.7975"],
   ["BM2 · +Labs",          A("0.7665"),"0.7638","0.7632",  "0.7784",A("0.7908"),"0.7780"]],
  [3.68,1.33,1.33,1.33,1.33,1.33,1.33],
  ["Tổ hợp","uni","DyAM","OvO","uni¹","DyAM¹","OvO¹"],cell_h=0.5,hdr_h=0.42,fs=12)
tf=box(s,0.82,4.5,11.68,0.35); p=para(tf,True)
_set(p,"Cột 1–3: 5-seed robust  ·  Cột 4–6 (¹): single-run kiểu bài báo  ·  in đậm xanh = cao nhất mỗi nhóm/hàng",10.5,C['faint'])
bl=box(s,1.0,5.0,11.3,1.3)
bullets(bl,[([("OvO ≈ uniform_avg",True),(" (bám sát ±0.005); cả hai ≥ DyAM ở đánh giá 5-seed robust.",False)],False),
            ("Chênh giữa 3 mô hình chỉ ~0.01 — nằm trong nhiễu, phần lớn KHÔNG có ý nghĩa thống kê.",False),
            ([("Single-run (1 phân hoạch): DyAM/OvO đạt đỉnh ~0.79–0.80",True),(" — đúng con số bài báo, nhưng là phân hoạch thuận lợi.",False)],False)],gap=5)
takeaway(s,"Nhận định",[("Không mô hình attention nào (DyAM/OvO) vượt baseline uniform_avg một cách bền vững — ba mô hình tương đương trong nhiễu.",False,None)])

# ---------------- SLIDE 4: B+C ----------------
s=slide(); eyebrow(s,"Stacked fusion & rà soát cấu hình","B · C")
title(s,"Hai kiểm chứng phụ",26)
tf=box(s,0.9,1.9,5.6,0.5); p=para(tf,True); _set(p,"Stacked late-fusion",16,C['ink'],bold=True)
chip(s,0.9,2.5,"Âm — không hơn baseline",'neg')
tf=box(s,0.9,3.05,5.6,3); p=para(tf,True,line=1.25)
_set(p,"Meta-learner học trọng số modality từ dự đoán out-of-fold. Kỳ vọng: tự hạ radiomics. ",13,C['ink'])
_set(p,"Thực tế ngược lại",13,C['neg'],bold=True)
_set(p," — meta đề cao radiomics (trọng số cao nhất) và bỏ pathology (≈0). Ngang uniform_avg (0.752 vs 0.775) nhưng diễn giải sai lệch.",13,C['ink'])
tf=box(s,6.9,1.9,5.5,0.5); p=para(tf,True); _set(p,"Vì sao bài báo ~0.79 còn tôi ~0.76?",16,C['ink'],bold=True)
chip(s,6.9,2.5,"Cấu hình GIỐNG HỆT",'acc')
tf=box(s,6.9,3.05,5.5,3); p=para(tf,True,line=1.25)
_set(p,"Khác biệt thuần ở cách đo: bài báo báo ",13,C['ink'])
_set(p,"1 lần chạy",13,C['ink'],bold=True); _set(p,"; tôi trung bình ",13,C['ink'])
_set(p,"5 phân hoạch",13,C['ink'],bold=True)
_set(p,". Tái lập kiểu 1-lần: BM4 khớp 4 chữ số (0.6932 vs 0.6931), BM1 = 0.7976. Số 0.79 là một phân hoạch thuận lợi.",13,C['ink'])
takeaway(s,"Ý nghĩa",[("Con số nội bộ dao động ~0.02 theo phân hoạch → ",False,None),
                     ("chênh giữa các model đều trong nhiễu",True,None),("; nên báo cáo 5-seed.",False,None)])

# ---------------- SLIDE 4b: QUY TRÌNH EMBEDDING ----------------
s=slide(); eyebrow(s,"Foundation embedding — quy trình trích đặc trưng","D")
title(s,"Từ ảnh gốc → vector: pipeline pathology & radiology",23)

_PIPE_STYLE={'plain':(C['surf'],C['line'],C['ink']),
             'fm':(C['accent_soft'],C['accent'],C['accent']),
             'weak':(C['neg_soft'],C['neg'],C['neg']),
             'out':(C['pos_soft'],C['pos'],C['pos'])}
def pipe_row(s,top,label,sub_label,steps):
    tf=box(s,0.9,top-0.44,11.5,0.36); p=para(tf,True)
    _set(p,label+"    ",13,C['accent'],bold=True,spacing=.6)
    _set(p,sub_label,11.5,C['muted'])
    n=len(steps); w=1.9; gap=0.48; x=0.95
    for i,(ttl,sub,kind) in enumerate(steps):
        fill,border,tcol=_PIPE_STYLE[kind]
        pbox(s,x,top,w,1.0,ttl,sub,fill=fill,tcol=tcol,border=border)
        if i<n-1: arrow(s,x+w+0.03,top+0.37)
        x+=w+gap

pipe_row(s,2.7,"PATHOLOGY  (WSI)","235 slide · nhuộm PD-L1",[
    ("WSI .svs","slide gốc",'plain'),
    ("Tile 256²@20x","vùng Tumor (HALO)",'plain'),
    ("Phikon ViT-B","768-d / tile",'fm'),
    ("Mean-pool","gộp mọi tile",'weak'),
    ("Vector 768-d","→ LR / fusion",'out')])
pipe_row(s,4.85,"RADIOLOGY  (CT)","237 volume · lesion mask",[
    ("CT .nii + .mha","series khớp size",'plain'),
    ("Window + crop","lát/khối quanh lesion",'plain'),
    ("Encoder CT","BiomedCLIP·MedicalNet·fmcib",'fm'),
    ("Mean-pool","/ volume",'weak'),
    ("Vector 512–4096-d","→ LR / fusion",'out')])
takeaway(s,"Hai lựa chọn thiết kế",[("Encoder ĐÓNG BĂNG (chỉ inference, không fine-tune)  ·  gộp bằng ",False,None),
    ("mean-pool — cách thô nhất",True,C['neg']),(" (chuẩn hơn là ABMIL). Vừa là thiết kế, vừa là giới hạn của thí nghiệm.",False,None)])

# ---------------- SLIDE 5: EXP D ----------------
s=slide(); eyebrow(s,"Foundation embedding từ ảnh gốc — kết quả","D")
title(s,"4 foundation model, 2 modality — không cái nào vượt thủ công",23)
chip(s,0.9,1.7,"Âm nhất quán",'neg'); chip(s,3.4,1.7,"WSI 235 · CT 237 · GPU + Colab",'par')
table(s,0.9,2.35,11.5,
   [[("GLCM (thủ công)",C['ink'],True),"pathology","0.622",("0.765",C['accent'],True),("generalize tốt",C['pos'],True),C['accent_soft']],
    ["Phikon","pathology",("0.680",C['pos'],True),"0.697",("ext không vượt GLCM",C['muted'],False)],
    [("Radiomics (thủ công)",C['ink'],True),"radiology","0.582",("0.461",C['neg'],True),("KHÔNG generalize",C['neg'],True),C['accent_soft']],
    ["BiomedCLIP 2.5D","radiology","0.510","0.530",("≈ ngẫu nhiên",C['muted'],False)],
    ["MedicalNet 3D","radiology","0.514","0.627",("≈ ngẫu nhiên (ext nhiễu)",C['muted'],False)],
    [("fmcib (cancer-CT, Colab)",C['ink'],True),"radiology",("0.503",C['neg'],True),"0.546",("≈ ngẫu nhiên",C['muted'],False)]],
   [3.4,1.7,1.5,1.5,3.4],["Đặc trưng","Modality","nội bộ","external","Đánh giá"],cell_h=0.44,hdr_h=0.36,fs=12)
takeaway(s,"Chốt",[("Kể cả fmcib (model 3D cancer-CT chuyên biệt) cũng ~ngẫu nhiên nội bộ (0.503). Ba model CT hội tụ ~0.51 → ",False,None),
                   ("embedding CT không mang tín hiệu response.",True,None)])

# ---------------- SLIDE 6: EXP E ----------------
s=slide(); eyebrow(s,"Embedding trong FUSION đa nguồn","E")
title(s,"Kết hợp embedding không giúp — mà kéo tụt",26)
chip(s,0.9,1.75,"Hại fusion, tỉ lệ với độ nhiễu",'neg')
tf=box(s,0.9,2.45,5.5,0.9); p=para(tf,True,line=1.2)
_set(p,"Càng thay nhiều imaging bằng embedding, AUC càng giảm — thứ tự khớp “độ yếu” của embedding.",13,C['ink'])
bar(s,0.9,3.55,5.8,"Thủ công (BM1)",0.90,"0.775",C['pos'])
bar(s,0.9,4.05,5.8,"+ fmcib (thêm)",0.78,"0.744",C['accent'])
bar(s,0.9,4.55,5.8,"Embed radiology",0.60,"0.719",C['warn'])
bar(s,0.9,5.05,5.8,"Embed cả 2",0.35,"0.653",C['neg'])
tf=box(s,7.2,2.9,5.2,3); p=para(tf,True,line=1.3)
_set(p,"CT-embed (≈ngẫu nhiên) hại nặng; Phikon (có tín hiệu) hại nhẹ. Thêm fmcib vào combo tốt nhất vẫn ",13,C['ink'])
_set(p,"−0.030",13,C['neg'],bold=True); _set(p,".",13,C['ink'])
p=para(tf,space_before=12,line=1.3); _set(p,"→ Giả thuyết “bù trừ trong fusion” bị bác bỏ. Embedding chỉ đưa nhiễu vào tổ hợp vốn đã tốt.",13,C['ink'],bold=True)

# ---------------- SLIDE 7: EXP F ----------------
s=slide(); eyebrow(s,"TabPFN — tabular foundation model","F")
title(s,"TabPFN không vượt hồi quy tuyến tính / uniform_avg",24)
chip(s,0.9,1.75,"Ngang / không hơn baseline",'par'); chip(s,4.3,1.75,"comparator hiện đại cho paper",'acc')
tf=box(s,0.9,2.4,5.5,0.4);p=para(tf,True);_set(p,"Per-modality: TabPFN vs LR",12,C['faint'])
table(s,0.9,2.8,5.5,
   [["PD-L1",("0.719",C['accent'],True),"0.710"],
    ["Genomic",("0.676",C['accent'],True),"0.656"],
    ["GLCM pathology",("0.576",C['accent'],True),"0.547"],
    ["Radiomics","0.576","0.579"]],
   [3.0,1.25,1.25],["Modality","LR","TabPFN"],cell_h=0.40,hdr_h=0.34)
tf=box(s,6.9,2.4,5.5,0.4);p=para(tf,True);_set(p,"Fusion: TabPFN-late vs uniform_avg",12,C['faint'])
table(s,6.9,2.8,5.5,
   [["BM1",("0.775",C['accent'],True),("0.715",C['neg'],False)],
    ["BM2",("0.767",C['accent'],True),("0.719",C['neg'],False)],
    ["BM3",("0.711",C['accent'],True),("0.667",C['neg'],False)],
    [("BM4 (2 nguồn)",C['ink'],False),"0.719",("0.743",C['pos'],True),C['pos_soft']]],
   [3.0,1.25,1.25],["Combo","uniform","TabPFN"],cell_h=0.40,hdr_h=0.34)
takeaway(s,"Kết luận",[("Thắng 0/5 per-modality, 1/4 fusion (chỉ BM4 — ít nguồn, sạch). Khớp literature 2025 “ML thường ≈ tabular foundation models”. Giữ làm comparator.",False,None)])

# ---------------- SLIDE 8: GENERALIZATION ----------------
s=slide(); eyebrow(s,"Phát hiện xuyên suốt · điểm mạnh nhất cho paper")
title(s,"Generalization là trục có đòn bẩy thật",26)
tf=box(s,0.9,1.95,5.6,1.1); p=para(tf,True,line=1.25)
_set(p,"Xuyên mọi thí nghiệm: ",13,C['ink'])
_set(p,"pathology transfer ra ngoài mẫu, radiomics thì không",13,C['ink'],bold=True)
_set(p," — cả thí nghiệm của tôi lẫn bài báo gốc đều xác nhận.",13,C['ink'])
bar(s,0.9,3.35,5.7,"Pathology (external)",0.80,"0.74–0.77",C['pos'])
bar(s,0.9,3.85,5.7,"Radiomics (external)",0.28,"0.43–0.66",C['neg'])
tf=box(s,0.9,4.4,5.6,1.4);p=para(tf,True,line=1.2)
_set(p,"Radiomics external thường dưới 0.5 → học “quirk” máy chụp, không phải sinh học khối u.",12,C['muted'])
tf=box(s,6.9,1.95,5.5,0.5);p=para(tf,True);_set(p,"Vì sao đây là đóng góp",14,C['accent'],bold=True)
bl=box(s,6.9,2.5,5.5,3.2)
bullets(bl,[("“Modality nào generalize” đang là chủ đề nóng multimodal 2025.",False),
            ([("Ghép với đóng góp NLP-clinical đã có",True),(" → câu chuyện “nguồn nào đáng tin + biểu diễn lâm sàng bằng ngôn ngữ”.",False)],False),
            ([("Các kết quả âm → ",False),("ablation nghiêm túc",True),(": model đơn giản + hiểu dữ liệu > chạy theo con số.",False)],False)],gap=9)

# ---------------- SLIDE 9: CONCLUSION ----------------
s=slide()
rect(s,0,0,SW,SH,C['accent_soft'])
rect(s,0,0,0.16,SH,C['accent'])
eyebrow(s,"Kết luận & hướng đi")
title(s,"Mọi con đường hội tụ về một thông điệp",27)
tf=box(s,0.9,1.95,11.3,1.2); p=para(tf,True,line=1.28)
_set(p,"Sau fusion attention · stacked · 4 foundation embedding · TabPFN — ",15,C['ink'])
_set(p,"không gì vượt uniform_avg / hồi quy tuyến tính",15,C['ink'],bold=True)
_set(p," trên đặc trưng thủ công. Trần nằm ở dữ liệu (n nhỏ, radiomics không transfer), không ở model/feature.",15,C['ink'])
tf=box(s,0.9,3.35,5.6,0.5);p=para(tf,True);_set(p,"Ba đóng góp cho paper",14,C['accent'],bold=True)
bl=box(s,0.9,3.9,5.6,3)
bullets(bl,[([("NLP-clinical embedding",True),(" (đã có) — biểu diễn lâm sàng bằng ngôn ngữ.",False)],False),
            ([("Phát hiện generalization",True),(" — pathology transfer, radiomics không.",False)],False),
            ([("Ablation trung thực",True),(" — model/feature phức tạp không hơn baseline.",False)],False)])
tf=box(s,6.9,3.35,5.5,0.5);p=para(tf,True);_set(p,"Việc còn lại",14,C['accent'],bold=True)
bl=box(s,6.9,3.9,5.5,3)
bullets(bl,[("Chốt phần thực nghiệm — bằng chứng đã đầy đủ, nhất quán.",False),
            ("Tùy chọn: nâng cấp NLP encoder (LLM y khoa).",False),
            ("Viết: generalization + baseline-đơn-giản làm câu chuyện chính.",False)])

# ---------------- SPEAKER NOTES / KỊCH BẢN ----------------
TITLES=[
 "Bìa — thông điệp chính",
 "Bối cảnh & câu hỏi",
 "Mô hình baseline: uniform_avg",
 "Mô hình attention: DyAM (A) + OvO (B)",
 "A · Fusion attention vs uniform_avg",
 "Tổng quan 3 mô hình fusion",
 "B·C · Stacked fusion & rà soát cấu hình",
 "D · Quy trình embedding ảnh gốc",
 "D · Foundation embedding — kết quả",
 "E · Embedding trong fusion",
 "F · TabPFN",
 "Generalization — điểm mạnh nhất",
 "Kết luận & hướng đi",
]
SCRIPTS=[
 "Xin chào. Đây là tổng hợp hai ngày thực nghiệm cho bài toán dự đoán đáp ứng liệu pháp miễn dịch NSCLC từ dữ liệu đa nguồn. Câu hỏi lớn: các phương pháp phức tạp — attention học được, foundation model, embedding từ ảnh gốc — có thực sự hơn một baseline đơn giản không? Tôi nói luôn kết luận: sau sáu nhóm thí nghiệm, không phương pháp nào vượt được baseline uniform_avg. Trần hiệu năng nằm ở bản thân dữ liệu, không phải ở mô hình. Điểm dương duy nhất, và cũng là đóng góp mạnh nhất, là phát hiện về generalization ở cuối bài.",
 "Bài toán: kết hợp năm nguồn — radiomics từ CT, pathology từ nhuộm PD-L1, genomic, điểm PD-L1, và xét nghiệm lâm sàng. Mô hình gốc trong bài báo là DyAM, dynamic attention fusion. Tôi kiểm chứng ba câu hỏi: attention có hơn trung bình đơn giản; foundation embedding có phá được trần; và TabPFN có hơn hồi quy tuyến tính. Về phương pháp, mọi con số đều là trung bình năm seed mười-fold có hoán vị phân hoạch, kèm kiểm định bootstrap trên bệnh nhân — để chống việc chỉ chạy một lần rồi lấy số đẹp. Và tôi luôn kiểm trên cohort ngoài, vì AUC nội bộ đã kịch trần khoảng 0.78.",
 "Đây là baseline uniform_avg. Nó có hai phần. Phần HỌC: mỗi nguồn có một head hồi quy nhỏ — rᵢ bằng tanh của Wᵢ nhân xᵢ — các trọng số W này được huấn luyện bình thường bằng gradient descent. Phần KHÔNG học: trọng số kết hợp aᵢ bằng mask chia tổng mask, tức chia đều cho các nguồn bệnh nhân có, nguồn nào vắng thì tự loại. Về bản chất đây chính là DyAM nhưng cơ chế attention bị vô hiệu hoá, cố định về chia đều. Nó học cách đọc từng nguồn rồi cộng đều, thay vì học tin nguồn nào hơn.",
 "Để so sánh, đây là hai biến thể attention HỌC được của DyAM. Panel A là cooperative attention: một ma trận N nhân N học tương tác giữa các nguồn. Panel B là competitive, gọi là OvO — mỗi nguồn cạnh tranh với trung bình các nguồn còn lại. Cả hai đều học trọng số aᵢ một cách có tham số. Câu hỏi rất tự nhiên: tầng attention học được này có đáng giá không? Slide sau trả lời.",
 "Câu trả lời là không. Tôi chạy đủ mọi tổ hợp nguồn, hơn bốn mươi cấu hình. uniform_avg luôn ngang hoặc hơn DyAM và OvO. Quan trọng hơn: khoảng cách nghiêng về uniform lại TĂNG khi thêm nguồn — ngược hẳn kỳ vọng 'nhiều nguồn thì attention thắng'. DyAM chỉ thắng có ý nghĩa ở hai trên hai mươi sáu tổ hợp, đều ở trường hợp ít nguồn nhất. Kết luận: attention học được không đóng góp gì, nó chỉ tái lập phép chia đều.",
 "Đây là cái nhìn tổng quan ba mô hình fusion trên bốn tổ hợp benchmark. Nhìn bảng: uniform_avg và OvO gần như trùng nhau, bám sát nhau trong khoảng năm phần nghìn, và cả hai đều ngang hoặc hơn DyAM. Chênh lệch giữa ba mô hình chỉ khoảng một phần trăm, phần lớn không đạt ý nghĩa thống kê — tức chúng tương đương trong nhiễu. Con số 0.79 đến 0.80 mà bài báo báo cáo là kết quả single-run của DyAM và OvO trên một phân hoạch thuận lợi. Thông điệp: không mô hình attention nào vượt được baseline đơn giản một cách bền vững, nên ta chọn baseline vì nó đơn giản và ổn định hơn.\n\n"
 "[GIẢI THÍCH single-run vs 5-seed — dùng khi hội đồng hỏi 'sao số thấp hơn bài báo gốc']: "
 "Đánh giá bằng 10-fold cross-validation cần chia ngẫu nhiên bệnh nhân thành các nhóm; cách chia phụ thuộc một 'seed' ngẫu nhiên. "
 "SINGLE-RUN là chạy với MỘT cách chia; 5-SEED là trung bình của NĂM cách chia khác nhau. "
 "Vì cỡ mẫu nhỏ (~250 bệnh nhân), AUC dao động mạnh giữa các cách chia — riêng BM1 dải là 0.726 đến 0.781, "
 "tức chỉ đổi cách chia thôi đã lệch tới 0.055. Con số 0.79 của bài báo rơi vào một lần chia THUẬN LỢI, "
 "thậm chí cao hơn cả đỉnh của dải 5-seed; còn 0.76 là TRUNG BÌNH, đại diện và ổn định hơn. "
 "Bằng chứng cấu hình giống hệt: khi tái lập ĐÚNG kiểu single-run, số của em khớp bài báo tới bốn chữ số ở BM4 "
 "(0.6932 so với 0.6931). Vì vậy em báo cáo 5-seed để trung thực về phương sai và chống cherry-pick. "
 "Ví von: single-run như chấm một đề thi (có thể dễ/khó bất chợt); 5-seed như trung bình năm đề — công bằng hơn.",
 "Hai kiểm chứng phụ. Bên trái: stacked fusion — cho một meta-model học trọng số nguồn từ dự đoán out-of-fold, kỳ vọng nó tự hạ radiomics. Kết quả ngược lại: nó đề cao radiomics và bỏ pathology, vì học từ tín hiệu nội bộ. Bên phải, câu hỏi quan trọng: vì sao bài báo báo 0.79 còn tôi 0.76? Cấu hình giống hệt — khác biệt thuần ở chỗ bài báo báo MỘT lần chạy, tôi trung bình NĂM phân hoạch. Tái lập đúng kiểu một-lần, số của tôi khớp bài báo tới bốn chữ số. Vậy 0.79 chỉ là một phân hoạch may mắn, không phải cấu hình tốt hơn.",
 "Trước khi xem kết quả, đây là cách tôi biến ẢNH GỐC thành vector đặc trưng — pipeline cho cả hai loại ảnh. Với PATHOLOGY: từ slide WSI, tôi cắt thành các tile 256 pixel ở độ phóng 20x, chỉ trong vùng khối u do HALO khoanh; mỗi tile đưa qua Phikon — một Vision Transformer nền tảng — ra vector 768 chiều; rồi mean-pool trung bình mọi tile thành một vector cho mỗi slide. Với RADIOLOGY: từ file CT và mask tổn thương, tôi chọn đúng series theo kích thước, cắt cửa sổ mô mềm, crop quanh lesion, đưa qua ba encoder — BiomedCLIP, MedicalNet 3D và fmcib — rồi cũng mean-pool thành một vector cho mỗi ca. Hai điểm cốt lõi: encoder được ĐÓNG BĂNG, chỉ chạy inference chứ không fine-tune; và cách gộp là mean-pool — đơn giản nhất và cũng là thô nhất, chuẩn mực hơn là ABMIL học trọng số tile. Chính hai lựa chọn này vừa định hình vừa giới hạn kết quả ở slide sau.",
 "Đây là nỗ lực lớn nhất — thay đặc trưng thủ công bằng embedding foundation model chạy trực tiếp trên ảnh gốc, cả WSI pathology lẫn CT. Tôi thử bốn model. Với pathology, Phikon giàu tín hiệu hơn ở nội bộ nhưng không vượt được GLCM thủ công ngoài mẫu. Với radiology, ba model — BiomedCLIP, MedicalNet, và cả fmcib chuyên biệt cho ung thư CT mà tôi phải chạy trên Colab — đều gần như ngẫu nhiên ở nội bộ, quanh 0.51. Kể cả fmcib, model đúng nhất, cũng không mang tín hiệu. Ba model độc lập cùng hội tụ về ngẫu nhiên, nên đây không phải lỗi chọn model.",
 "Câu hỏi tiếp: embedding đơn lẻ yếu, nhưng khi kết hợp với nguồn khác có bổ sung tín hiệu không? Câu trả lời còn tệ hơn — nó kéo tụt fusion. Càng thay nhiều imaging bằng embedding, AUC càng giảm: từ 0.775 của bản thủ công xuống 0.653 khi embed cả hai. Thứ tự mức hại khớp chính xác với độ nhiễu của từng embedding. Kể cả chỉ thêm fmcib vào tổ hợp tốt nhất cũng giảm ba phần trăm. Giả thuyết 'bù trừ trong fusion' bị bác bỏ dứt khoát.",
 "Thí nghiệm cuối: TabPFN, một tabular foundation model chuyên dữ liệu nhỏ — đúng ràng buộc của chúng ta. Nhưng nó cũng không vượt hồi quy tuyến tính: thắng không-trên-năm ở per-modality, và một-trên-bốn ở fusion, chỉ ở BM4 là tổ hợp ít nguồn và sạch nhất. Kết quả khớp đúng literature 2025: các model tabular hiện đại thường chỉ ngang ML truyền thống. Nên tôi giữ TabPFN làm comparator hiện đại trong bài, không phải một cải tiến.",
 "Đây là phát hiện xuyên suốt và là điểm mạnh nhất cho paper. Qua mọi thí nghiệm, một tín hiệu bền vững: pathology transfer được ra ngoài mẫu, đạt 0.74 đến 0.77; còn radiomics thì không, thường dưới 0.5 và rất bất ổn — nghĩa là nó học đặc tính máy chụp, không phải sinh học khối u. 'Modality nào generalize' đang là chủ đề nóng trong literature multimodal 2025. Ghép với đóng góp NLP-clinical đã có, ta có câu chuyện mạch lạc: nguồn nào đáng tin, và biểu diễn dữ liệu lâm sàng bằng ngôn ngữ.",
 "Tổng kết: sau fusion attention, stacked, bốn foundation embedding và TabPFN, mọi con đường hội tụ về một thông điệp — không gì vượt được baseline đơn giản trên đặc trưng thủ công. Trần nằm ở dữ liệu, không ở mô hình hay feature. Ba đóng góp cho paper: NLP-clinical embedding đã có; phát hiện generalization; và một loạt ablation trung thực cho thấy model đơn giản cộng hiểu dữ liệu thắng việc chạy theo con số. Việc còn lại là chốt thực nghiệm và viết, với generalization và baseline đơn giản làm câu chuyện trung tâm. Xin cảm ơn.",
]
for sld,scr in zip(prs.slides,SCRIPTS):
    sld.notes_slide.notes_text_frame.text=scr

out=r"C:\Users\Admin\AppData\Local\Temp\claude\D--code-master-doan-code\87b18592-2446-4da4-b16a-2a37fd51b540\scratchpad\bao-cao-thuc-nghiem.pptx"
prs.save(out); print("SAVED", out, len(prs.slides._sldIdLst), "slides")

# xuất kịch bản ra markdown
md=r"C:\Users\Admin\AppData\Local\Temp\claude\D--code-master-doan-code\87b18592-2446-4da4-b16a-2a37fd51b540\scratchpad\kich-ban-thuyet-trinh.md"
with open(md,"w",encoding="utf-8") as f:
    f.write("# Kịch bản thuyết trình — Tổng hợp thực nghiệm 21–22/07\n\n")
    f.write("_Mỗi mục tương ứng một slide. Thời lượng gợi ý: ~45–60 giây/slide, tổng ~9–11 phút._\n\n")
    for i,(t,s2) in enumerate(zip(TITLES,SCRIPTS),1):
        f.write(f"## Slide {i} — {t}\n\n{s2}\n\n")
print("SAVED", md)
