"""Linux: chạy các ô 1-13,16,17 của notebook gốc (main); lung_helpers.py của main; so với bảng AUC trong HTML. cwd = thư mục chứa ./datasets/"""
import sys, os, json, re
sys.path.insert(0, "/mnt/d/repro_linux"); os.chdir("/mnt/d/code/master/doan")
import matplotlib; matplotlib.use("Agg")
nb = json.load(open("/mnt/d/repro_linux/Figures-Finalized-main.ipynb", encoding="utf-8"))
ns = {"__name__": "__main__"}
RUN = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1,2,3,4,5,6,7,8,9,10,11,12,13,16,17]
for i in RUN:
    print(f"=== CELL {i}", flush=True)
    exec(compile("".join(nb["cells"][i]["source"]), f"<cell {i}>", "exec"), ns)
import numpy as np, scipy, sklearn, pandas as pd, torch, platform
from sklearn.metrics import roc_auc_score
html = {}
for l in open("/mnt/d/repro_linux/fig_html.txt", encoding="utf-8").read().splitlines()[1260:1290]:
    m = re.match(r'^\s*\d+\s+(.+?)\s+([0-9]\.[0-9]{6})\s+[0-9]\.[0-9]{6}\s+[0-9]\.[0-9]{6}', l)
    if m: html[m.group(1).strip()] = float(m.group(2))
print(f"ENV {platform.system()} py{platform.python_version()} torch {torch.__version__} sklearn {sklearn.__version__} numpy {np.__version__} scipy {scipy.__version__} pandas {pd.__version__}")
ex = 0; tot = 0; res = {}
for k, df in ns["summary_dfs"].items():
    d = df[["label", "score"]].dropna()
    a = float(roc_auc_score(d["label"].astype(float), d["score"].astype(float))); res[k] = a
    if k in html:
        tot += 1; ok = abs(a - html[k]) < 5e-6; ex += ok
        print(f"AUC {k:<34}{a:.6f}  HTML {html[k]:.6f}  d {a-html[k]:+.6f}{'  OK' if ok else ''}", flush=True)
    else:
        print(f"AUC {k:<34}{a:.6f}  (không có trong HTML)", flush=True)
print(f"KHOP_TUYET_DOI {ex}/{tot}")
json.dump(res, open(sys.argv[2] if len(sys.argv) > 2 else "/mnt/d/repro_linux/result_linux.json", "w"), indent=1)
print("XONG")
