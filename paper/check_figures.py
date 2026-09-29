# -*- coding: utf-8 -*-
"""Kiểm tra tự động layout các hình sơ đồ của bài báo.

Chạy:  python paper/check_figures.py        (từ thư mục paper/ hoặc code/)

Bắt 3 loại lỗi mà mắt thường hay bỏ sót:
  1. TEXT-OVERLAP : hai nhãn đè lên nhau (lỗi "díu chữ").
  2. BOX-OVERFLOW : nhãn tràn ra ngoài ô/khung chứa nó.
  3. FIG-OVERFLOW : nhãn tràn ra ngoài vùng vẽ của figure.

Cách hoạt động: exec script vẽ hình (cắt bỏ phần savefig), rồi đo
get_window_extent() thật của từng text artist sau khi render — không phải
ước lượng theo số ký tự.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

# Console Windows mặc định cp1252 -> vỡ khi in tiếng Việt / tên nhãn unicode.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

HERE = Path(__file__).resolve().parent

FIG_SCRIPTS = [
    ('Figure 1  (study overview)',      'generate_fig1_demo.py'),
    ('Figure 2  (NLP pipeline)',        'generate_fig2_demo.py'),
    ('Figure 3  (attention architect.)', 'generate_fig_attention_demo.py'),
]

# Nhãn đè nhau dưới ngưỡng này coi như không đáng kể (đơn vị: tỉ lệ diện
# tích chồng lấn so với nhãn nhỏ hơn).
OVERLAP_TOL = 0.08
EDGE_TOL = 0.04          # dung sai mép, đơn vị dữ liệu


def load_axes(script_path):
    """Exec script vẽ hình, trả về (fig, ax, tight) — bỏ phần savefig.

    `tight` = script có lưu bằng bbox_inches='tight' hay không. Nếu có thì
    chữ nằm ngoài xlim/ylim VẪN hiện đủ (khung được nới ra khi lưu), nên
    không được coi là lỗi tràn.
    """
    src = script_path.read_text(encoding='utf-8')
    tight = "bbox_inches='tight'" in src or 'bbox_inches="tight"' in src
    if 'plt.savefig' in src:
        src = src[:src.index('plt.savefig')]
    g = {'__name__': '__main__', '__file__': str(script_path)}
    exec(compile(src, script_path.name, 'exec'), g)
    fig, ax = g['fig'], g['ax']
    fig.canvas.draw()
    return fig, ax, tight


def bbox_of(artist, ax, renderer):
    return artist.get_window_extent(renderer=renderer).transformed(
        ax.transData.inverted())


def inter_area(a, b):
    dx = min(a.x1, b.x1) - max(a.x0, b.x0)
    dy = min(a.y1, b.y1) - max(a.y0, b.y0)
    return dx * dy if (dx > 0 and dy > 0) else 0.0


def check(script_name):
    path = HERE / script_name
    fig, ax, tight = load_axes(path)
    r = fig.canvas.get_renderer()

    texts = [t for t in ax.texts if t.get_text().strip()]
    tb = [bbox_of(t, ax, r) for t in texts]

    problems = []

    # ── 1. chữ đè chữ ────────────────────────────────────────────────────
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            ov = inter_area(tb[i], tb[j])
            if ov <= 0:
                continue
            small = min(tb[i].width * tb[i].height, tb[j].width * tb[j].height)
            frac = ov / small if small else 0
            if frac > OVERLAP_TOL:
                problems.append(
                    ('TEXT-OVERLAP',
                     f'{texts[i].get_text()[:28]!r} <-> '
                     f'{texts[j].get_text()[:28]!r}  (chồng {frac:.0%})'))

    # ── 2. chữ tràn khỏi ô chứa nó ───────────────────────────────────────
    boxes = []
    for p in ax.patches:
        if type(p).__name__ not in ('FancyBboxPatch', 'Rectangle'):
            continue
        bb = bbox_of(p, ax, r)
        if bb.width < 0.4 or bb.height < 0.2:      # bỏ ô heatmap li ti
            continue
        boxes.append(bb)

    x0f, x1f = ax.get_xlim()
    y0f, y1f = ax.get_ylim()
    fig_w = x1f - x0f

    for t, b in zip(texts, tb):
        cx, cy = (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2
        cont = [bb for bb in boxes
                if bb.x0 <= cx <= bb.x1 and bb.y0 <= cy <= bb.y1]
        if not cont:
            continue
        host = min(cont, key=lambda bb: bb.width * bb.height)
        if host.width > 0.85 * fig_w:              # panel bao ngoài -> bỏ
            continue
        over_r = b.x1 - host.x1
        over_l = host.x0 - b.x0
        if over_r > EDGE_TOL or over_l > EDGE_TOL:
            problems.append(
                ('BOX-OVERFLOW',
                 f'{t.get_text()[:34]!r}  lố phải {over_r:+.2f} / '
                 f'lố trái {over_l:+.2f} đơn vị'))

    # ── 3. chữ tràn khỏi vùng vẽ ─────────────────────────────────────────
    # Chỉ có ý nghĩa khi KHÔNG lưu bằng bbox_inches='tight'; với 'tight'
    # matplotlib nới khung lúc lưu nên chữ ngoài xlim/ylim vẫn hiện đủ.
    if not tight:
        for t, b in zip(texts, tb):
            if (b.x0 < x0f - EDGE_TOL or b.x1 > x1f + EDGE_TOL
                    or b.y0 < y0f - EDGE_TOL or b.y1 > y1f + EDGE_TOL):
                problems.append(
                    ('FIG-OVERFLOW',
                     f'{t.get_text()[:34]!r} ra ngoài khung vẽ'))

    matplotlib.pyplot.close(fig)
    return len(texts), problems


def main():
    total = 0
    for title, script in FIG_SCRIPTS:
        n_text, problems = check(script)
        status = 'OK' if not problems else f'{len(problems)} LOI'
        print(f'\n{title:<34} [{n_text} nhãn]  -> {status}')
        for kind, msg in problems:
            print(f'    {kind:<14} {msg}')
        total += len(problems)

    print(f'\n{"=" * 62}\nTONG: {total} lỗi layout')
    return 1 if total else 0


if __name__ == '__main__':
    sys.exit(main())
