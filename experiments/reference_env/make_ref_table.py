"""Gộp experiments/results/ref_all21_part*.json thành bảng 21 tổ hợp x 4 biến thể (1 seed và 5 seed) + thống kê tóm tắt."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
RES = HERE.parents[1] / "results"
ORDER = ["TMB", "PDL1", "IHC-A", "Gen", "Rad", "Rad-LU", "TMB+PDL1", "PDL1+Gen", "Rad+IHC-A", "Rad+IHC-G", "Rad+Gen",
         "IHC-A+Gen", "IHC-G+Gen", "Rad+IHC-A+Gen", "Rad+IHC-G+Gen", "Rad+IHC-A+MutAmp", "Rad+IHC-A+Gen+PDL1",
         "Rad+IHC-G+Gen+PDL1", "Rad+IHC-A+Gen+TMB+PDL1", "Rad+IHC-A+Gen+PDL1+Labs", "Rad+IHC-G+Gen+PDL1+Labs"]
# số AUC DyAM lần chạy đơn trong output lưu của tác giả (figures-finalized.html); None = không có trong bảng của tác giả
AUTHOR = {"IHC-A": 0.605631, "Gen": 0.675850, "Rad": 0.701413, "PDL1+Gen": 0.693112, "Rad+IHC-A": 0.680753, "Rad+Gen": 0.759285,
          "IHC-A+Gen": 0.714996, "Rad+IHC-A+Gen": 0.754228, "Rad+IHC-G+Gen": 0.782563, "Rad+IHC-A+Gen+PDL1": 0.798169,
          "Rad+IHC-G+Gen+PDL1": 0.783348}
V = ["dyam", "ovo", "dyam_nlp", "ovo_nlp"]
NAMES = {"dyam": "DyAM", "ovo": "OvO", "dyam_nlp": "NLP+DyAM", "ovo_nlp": "NLP+OvO"}

R = {}
for i in range(4):
    R.update(json.loads((RES / f"ref_all21_part{i}.json").read_text(encoding="utf-8")))
miss = [c for c in ORDER if c not in R]
if miss:
    sys.exit(f"thiếu tổ hợp: {miss}")
envs = {R[c]["_env"] for c in ORDER}
print("ENV:", "; ".join(sorted(envs)))

lines = []
lines.append("| # | Tổ hợp | k | DyAM | OvO | NLP+DyAM | NLP+OvO | tác giả (DyAM) |")
lines.append("|---|---|---|---|---|---|---|---|")
for n, c in enumerate(ORDER, 1):
    e = R[c]
    k = len(c.replace("+Labs", "").split("+")) if False else None
    row = " | ".join(f"{e[v]['single_run']:.4f}" for v in V)
    a = AUTHOR.get(c)
    lines.append(f"| {n} | {c} | | {row} | {a:.4f} |" if a else f"| {n} | {c} | | {row} | - |")
print("\n### 1 seed (lần chạy đơn, seed 42, thứ tự tự nhiên)\n")
print("\n".join(lines))

lines = ["| # | Tổ hợp | DyAM | OvO | NLP+DyAM | NLP+OvO |", "|---|---|---|---|---|---|"]
for n, c in enumerate(ORDER, 1):
    e = R[c]
    lines.append(f"| {n} | {c} | " + " | ".join(f"{e[v]['mean']:.4f} ± {e[v]['sd']:.3f}" for v in V) + " |")
print("\n### 5 seed (trung bình ± sd, hoán vị nhãn)\n")
print("\n".join(lines))

big = [c for c in ORDER if c in ("Rad+IHC-A+Gen", "Rad+IHC-G+Gen", "Rad+IHC-A+MutAmp", "Rad+IHC-A+Gen+PDL1", "Rad+IHC-G+Gen+PDL1",
                                 "Rad+IHC-A+Gen+TMB+PDL1", "Rad+IHC-A+Gen+PDL1+Labs", "Rad+IHC-G+Gen+PDL1+Labs")]
print("\n### Tóm tắt\n")
for key, lab in (("single_run", "1 seed"), ("mean", "5 seed")):
    print(f"**{lab}**: trung bình 21 tổ hợp: " + ", ".join(f"{NAMES[v]} {np.mean([R[c][v][key] for c in ORDER]):.4f}" for v in V))
    print(f"  trung bình 8 tổ hợp lớn (>=5 nguồn): " + ", ".join(f"{NAMES[v]} {np.mean([R[c][v][key] for c in big]):.4f}" for v in V))
    print(f"  cao nhất: " + ", ".join(f"{NAMES[v]} {max(R[c][v][key] for c in ORDER):.4f} ({max(ORDER, key=lambda c: R[c][v][key])})" for v in V))
    for a, b in (("ovo", "dyam"), ("ovo_nlp", "dyam_nlp"), ("dyam_nlp", "dyam"), ("ovo_nlp", "ovo")):
        d = np.array([R[c][a][key] - R[c][b][key] for c in ORDER])
        db = np.array([R[c][a][key] - R[c][b][key] for c in big])
        print(f"  {NAMES[a]} - {NAMES[b]}: mean {d.mean():+.4f} (hơn {(d > 0.0005).sum()}/kém {(d < -0.0005).sum()}/ngang {(abs(d) <= 0.0005).sum()}); 8 tổ hợp lớn {db.mean():+.4f} (hơn {(db > 0.0005).sum()}/8)")
print("\n### So với tác giả (DyAM, 1 seed)\n")
for c, a in AUTHOR.items():
    d = R[c]["dyam"]["single_run"] - a
    print(f"  {c:<24} {R[c]['dyam']['single_run']:.6f}  tác giả {a:.6f}  lệch {d:+.6f}{'  (khớp tuyệt đối)' if abs(d) < 5e-6 else ('  (<= một cặp bệnh nhân)' if abs(d) < 1e-4 else '')}")
