"""Xuất toàn bộ kết quả chạy trong môi trường tham chiếu (ref-linux-py38) ra một file Excel.

    python experiments/reference_env/export_ref_excel.py     ->  excel/ref_linux_results_2026-10-13.xlsx
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "results"
REF = RES / "ref"
OUT = ROOT / "excel" / "ref_linux_results_2026-10-13.xlsx"
OUT.parent.mkdir(exist_ok=True)

V = ["dyam", "ovo", "dyam_nlp", "ovo_nlp"]
NAME = {"dyam": "DyAM", "ovo": "OvO", "dyam_nlp": "NLP+DyAM", "ovo_nlp": "NLP+OvO"}
AUTHOR = {"IHC-A": 0.605631, "Gen": 0.675850, "Rad": 0.701413, "PDL1+Gen": 0.693112, "Rad+IHC-A": 0.680753, "Rad+Gen": 0.759285,
          "IHC-A+Gen": 0.714996, "Rad+IHC-A+Gen": 0.754228, "Rad+IHC-G+Gen": 0.782563, "Rad+IHC-A+Gen+PDL1": 0.798169,
          "Rad+IHC-G+Gen+PDL1": 0.783348}

src = src_txt = open(ROOT / "paper" / "make_results_tables.py", encoding="utf-8").read()
ns = {}
exec(src[src.index("K = {"):src.index("BM = {")], ns)
K = ns["K"]
R = {}
for f in sorted(glob.glob(str(RES / "ref_all21_part*.json"))):
    R.update(json.load(open(f, encoding="utf-8")))
order = list(K)

sheets = {}

# 1) mô tả
sheets["README"] = pd.DataFrame({"Nội dung": [
    "Kết quả chạy trong môi trường tham chiếu ref-linux-py38 (WSL2 Ubuntu 22.04, Python 3.8.15, torch 1.8.1, scikit-learn 0.24.2, numpy 1.20.3).",
    "Cấu hình: epochs 125, lr 0.01, alpha 0.001, beta 0, KFold 10 (random_state 0), n=247 (62 đáp ứng / 185 không).",
    "Bộ lọc radiomics: ctx.rad_filters gốc. 1 seed = seed 42, thứ tự bệnh nhân tự nhiên. 5 seed = [42,7,123,2024,31337] hoán vị nhãn. 20 seed = 5 seed + 15 seed.",
    "Các sheet: 21combo_1seed, 21combo_5seed, 21combo_5seed_perseed, nlp20_20seed, loso_BM1, miss_vector, holdout_T1, learning_T2, NLP_AUC_arms, survival, rad_feature_counts.",
    "tdAUC tính ở Windows (scikit-survival) từ điểm nguy cơ do môi trường Linux sinh ra. Embedding câu tạo một lần ở Windows rồi cache.",
    "Nguồn thô: experiments/results/ref_all21_part*.json, experiments/results/ref/*.",
]})

# 2) 21 tổ hợp, 1 seed
rows = []
for i, c in enumerate(order, 1):
    r = {"#": i, "Tổ hợp": c, "k (số nguồn)": K[c]}
    for v in V:
        r[NAME[v]] = R[c][v]["single_run"]
    r["OvO − DyAM"] = R[c]["ovo"]["single_run"] - R[c]["dyam"]["single_run"]
    r["NLP+OvO − NLP+DyAM"] = R[c]["ovo_nlp"]["single_run"] - R[c]["dyam_nlp"]["single_run"]
    r["Tác giả gốc (DyAM)"] = AUTHOR.get(c)
    rows.append(r)
sheets["21combo_1seed"] = pd.DataFrame(rows)

# 3) 21 tổ hợp, 5 seed
rows = []
for i, c in enumerate(order, 1):
    r = {"#": i, "Tổ hợp": c, "k (số nguồn)": K[c]}
    for v in V:
        r[f"{NAME[v]} mean"] = R[c][v]["mean"]
        r[f"{NAME[v]} sd"] = R[c][v]["sd"]
    r["OvO − DyAM"] = R[c]["ovo"]["mean"] - R[c]["dyam"]["mean"]
    r["NLP+OvO − NLP+DyAM"] = R[c]["ovo_nlp"]["mean"] - R[c]["dyam_nlp"]["mean"]
    rows.append(r)
sheets["21combo_5seed"] = pd.DataFrame(rows)

rows = []
for i, c in enumerate(order, 1):
    for v in V:
        r = {"#": i, "Tổ hợp": c, "Mô hình": NAME[v]}
        r.update({f"seed {s}": a for s, a in zip([42, 7, 123, 2024, 31337], R[c][v]["per_seed"])})
        rows.append(r)
sheets["21combo_5seed_perseed"] = pd.DataFrame(rows)

# 4) 20 seed DyAM+NLP vs OvO+NLP
P = REF / "ovo_proof"
NLP20 = [("Rad+IHC-A+Gen", "Rad_IHCA_Gen"), ("Rad+IHC-G+Gen", "Rad_IHCG_Gen"), ("Rad+IHC-A+MutAmp", "Rad_IHCA_MutAmp"),
         ("Rad+IHC-A+Gen+PDL1", "Rad_IHCA_Gen_PDL1"), ("Rad+IHC-G+Gen+PDL1 (BM1)", "Rad_IHCG_Gen_PDL1"),
         ("Rad+IHC-A+Gen+TMB+PDL1", "Rad_IHCA_Gen_TMB_PDL1")]


def per(df):
    return np.array([roc_auc_score(df.loc[df[c].notna(), "label"], df.loc[df[c].notna(), c]) for c in df.columns if c.startswith("seed_")])


def ens(df):
    c = [x for x in df.columns if x.startswith("seed_")]
    return ((df[c] - df[c].mean()) / df[c].std()).mean(axis=1)


rows = []
rng = np.random.default_rng(0)
for nm, sl in NLP20:
    d = pd.read_csv(P / f"nlp20_dyam__{sl}.csv", index_col=0)
    o = pd.read_csv(P / f"nlp20_ovo__{sl}.csv", index_col=0)
    m = pd.read_csv(P / f"nlp20miss_ovo__{sl}.csv", index_col=0)
    a, b, c3 = per(d), per(o), per(m)
    ed, eo = ens(d), ens(o)
    k = (ed.notna() & eo.notna()).values
    y = d["label"].values[k]
    ed, eo = ed.values[k], eo.values[k]
    bs = []
    for _ in range(5000):
        i = rng.integers(0, k.sum(), k.sum())
        if len(set(y[i])) == 2:
            bs.append(roc_auc_score(y[i], eo[i]) - roc_auc_score(y[i], ed[i]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    rows.append({"Tổ hợp": nm, "NLP+DyAM mean (20 seed)": a.mean(), "NLP+DyAM sd": a.std(ddof=1), "NLP+OvO mean": b.mean(), "NLP+OvO sd": b.std(ddof=1),
                 "Δ mỗi phân hoạch (OvO−DyAM)": (b - a).mean(), "Số phân hoạch OvO hơn (/20)": int((b > a).sum()),
                 "Ensemble NLP+DyAM": roc_auc_score(y, ed), "Ensemble NLP+OvO": roc_auc_score(y, eo),
                 "Δ ensemble": roc_auc_score(y, eo) - roc_auc_score(y, ed), "Δ ens CI 2.5%": lo, "Δ ens CI 97.5%": hi,
                 "NLP+OvO+vector sẵn-có mean": c3.mean(), "Δ vector sẵn-có (có − không)": (c3 - b).mean(), "Số phân hoạch có vector hơn (/20)": int((c3 > b).sum())})
sheets["nlp20_20seed"] = pd.DataFrame(rows)

# 5) bỏ từng nguồn
ref = per(pd.read_csv(P / "nlp20_ovo__Rad_IHCG_Gen_PDL1.csv", index_col=0))
rows = [{"Cấu hình (BM1, NLP+OvO, 20 seed)": "Đầy đủ", "AUC mean": ref.mean(), "Δ so với đầy đủ": 0.0, "Số phân hoạch giảm (/20)": None}]
for a, nm in (("path", "Bỏ pathology IHC-G"), ("rad", "Bỏ CT radiomics (3 vị trí)"), ("pdl1", "Bỏ PD-L1 TPS"), ("nlp", "Bỏ NLP lâm sàng"), ("gen", "Bỏ genomics")):
    p = per(pd.read_csv(P / f"loso_ovo_{a}.csv", index_col=0))
    rows.append({"Cấu hình (BM1, NLP+OvO, 20 seed)": nm, "AUC mean": p.mean(), "Δ so với đầy đủ": (p - ref).mean(), "Số phân hoạch giảm (/20)": int((p < ref).sum())})
sheets["loso_BM1"] = pd.DataFrame(rows)

# 6) hold-out theo số chụp và đường cong học
t1 = json.load(open(REF / "order_ovo_t1.json", encoding="utf-8"))
rows = []
for cfg, d in t1.items():
    for kind, r in d.items():
        rows.append({"Tổ hợp": cfg, "Mô hình": kind, "AUC gộp": r["auc_pooled"], "CI 2.5%": r["ci"][0], "CI 97.5%": r["ci"][1],
                     "AUC 3 khối": ", ".join(f"{x:.3f}" for x in r["blocks"]), "n kiểm tra": r["n"]})
sheets["holdout_T1"] = pd.DataFrame(rows)
t2 = json.load(open(REF / "order_ovo_t2.json", encoding="utf-8"))
rows = []
for kind, d in t2.items():
    for frac, r in d.items():
        rows.append({"Mô hình": kind, "Tỉ lệ dữ liệu huấn luyện": float(frac), "~n huấn luyện": r["n_train_median"], "AUC mean (5 seed)": r["mean"], "sd": r["sd"]})
sheets["learning_T2"] = pd.DataFrame(rows)

# 7) bảng NLP + sống còn (DyAM, 5 seed)
five = json.load(open(REF / "verify_nlp_5seed.json", encoding="utf-8"))["arms"]
td = json.load(open(REF / "tdauc_5seed.json", encoding="utf-8"))
stem = {"No clinical": "no_clinical", "+ Labs": "labs", "+ NLP raw": "nlp_raw", "+ NLP-PCA16": "nlppca16", "+ Labs + NLP": "labs__nlp",
        "+ Labs + NLP-PCA16": "labs__nlppca16", "+ BioClinBERT-PCA16": "bioclinbertpca16"}
rows = []
for arm, A in five.items():
    for v, x in A.items():
        t = td[f"{arm}__{stem[v]}"]
        rows.append({"Nhánh": arm, "Biến thể lâm sàng": v, "AUC mean": x["auc"]["mean"], "AUC sd": x["auc"]["sd"],
                     "Δ vs không lâm sàng": x["auc"]["mean"] - A["No clinical"]["auc"]["mean"], "Δ vs Labs": x["auc"]["mean"] - A["+ Labs"]["auc"]["mean"],
                     "C-index mean": x["cindex"]["mean"], "C-index sd": x["cindex"]["sd"], "Cox HR mean": x["cox_hr"]["mean"], "Cox HR sd": x["cox_hr"]["sd"],
                     "KM χ² mean": x["km_chi2"]["mean"], "KM χ² sd": x["km_chi2"]["sd"], "KM χ² nhỏ nhất": min(x["km_chi2"]["per_seed"]),
                     "tdAUC 6th": t["6"]["mean"], "tdAUC 6th sd": t["6"]["sd"], "tdAUC 12th": t["12"]["mean"], "tdAUC 12th sd": t["12"]["sd"],
                     "tdAUC 18th": t["18"]["mean"], "tdAUC 18th sd": t["18"]["sd"]})
sheets["NLP_AUC_arms_survival"] = pd.DataFrame(rows)

# 8) số đặc trưng radiomics
rf = json.load(open(REF / "rad_feature_counts.json", encoding="utf-8"))
sheets["rad_feature_counts"] = pd.DataFrame([{"Vị trí": k, "Cột thô": v["raw_columns_in_modality"], "n bệnh nhân": v["n_patients"],
                                              "Trung vị": v["median"], "Nhỏ nhất": v["min"], "Lớn nhất": v["max"],
                                              "Theo fold": ", ".join(map(str, v["per_fold"]))} for k, v in rf.items()])

with pd.ExcelWriter(OUT, engine="openpyxl") as w:
    for name, df in sheets.items():
        df.to_excel(w, sheet_name=name, index=False)
        ws = w.sheets[name]
        for col in ws.columns:
            width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
            ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 60)
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                if isinstance(cell.value, float):
                    cell.number_format = "0.0000"
        ws.freeze_panes = "A2"
print("saved", OUT, {k: v.shape for k, v in sheets.items()})
