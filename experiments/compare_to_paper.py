"""So số chạy lại với đúng số đang in trong bài báo (Bảng 4 — tab:ovo_all21).

Đọc trực tiếp bảng LaTeX trong paper/sections/results.tex nên không cần chép tay số nào.
In ra bảng Markdown các ô lệch, dùng để dán vào file sai khác cho người dùng quyết định.

    PYTHONIOENCODING=utf-8 python experiments/compare_to_paper.py [--tol 0.0005]
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
TOL = 0.0005

ROW = re.compile(r"^(\d+)\s*&\s*([^&]+?)\s*&\s*(\d+)\s*&(.+?)\\\\", re.M)
NUM = re.compile(r"([0-9]*\.[0-9]+)")


def paper_rows():
    tex = (ROOT / "paper" / "sections" / "results.tex").read_text(encoding="utf-8")
    block = tex.split(r"\label{tab:ovo_all21}")[1].split(r"\end{tabular}")[0]
    rows = {}
    for idx, label, k, rest in ROW.findall(block):
        vals = [float(v) for v in NUM.findall(rest)]
        if len(vals) != 3:
            continue
        clean = re.sub(r"\s*\(BM\d\)", "", label).strip()
        rows[clean] = {"idx": int(idx), "k": int(k), "label": label.strip(),
                       "dyam": vals[0], "ovo": vals[1], "ovo_nlp": vals[2]}
    return rows


def rerun():
    out = {}
    pc = RESULTS / "paper_combos.json"
    if pc.exists():
        for key, v in json.load(pc.open(encoding="utf-8"))["combos"].items():
            out.setdefault(key, {})
            for col in ("dyam", "ovo"):
                if col in v and "mean_auc" in v[col]:
                    out[key][col] = v[col]["mean_auc"]
    on = RESULTS / "ovo_nlp_combos.json"
    if on.exists():
        for key, v in json.load(on.open(encoding="utf-8"))["combos"].items():
            if "nlp" in v and "mean" in v["nlp"]:
                out.setdefault(key, {})["ovo_nlp"] = v["nlp"]["mean"]
    return out


def main():
    tol = TOL
    if "--tol" in sys.argv:
        tol = float(sys.argv[sys.argv.index("--tol") + 1])
    paper, new = paper_rows(), rerun()
    print(f"| # | Tổ hợp | Cột | Bài báo | Chạy lại | Lệch |")
    print(f"|---|---|---|---|---|---|")
    n_cmp = n_diff = 0
    missing = []
    for key, p in sorted(paper.items(), key=lambda kv: kv[1]["idx"]):
        for col, name in (("dyam", "DyAM"), ("ovo", "OvO"), ("ovo_nlp", "OvO+NLP")):
            if key not in new or col not in new[key]:
                missing.append(f"{p['idx']} {key}/{name}")
                continue
            n_cmp += 1
            a, b = p[col], new[key][col]
            if abs(a - round(b, 3)) > tol:
                n_diff += 1
                print(f"| {p['idx']} | {p['label']} | {name} | {a:.3f} | "
                      f"{b:.4f} | {b - a:+.4f} |")
    print(f"\nĐã so {n_cmp} ô, lệch {n_diff}. Chưa có kết quả: {len(missing)} ô"
          + (f" ({', '.join(missing[:6])}…)" if missing else ""))


if __name__ == "__main__":
    main()
