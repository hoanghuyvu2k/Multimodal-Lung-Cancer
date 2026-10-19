"""Các chỉ số của notebook gốc (`generate_metric_plot_V2`: F1, Precision, Recall, Accuracy tại ngưỡng điểm z-score > 0, và AUC)
tính lại từ các file điểm đã lưu. Nhãn 1 = không đáp ứng (SD/POD) là lớp dương, giống notebook. Không train lại.

  python experiments/reference_env/notebook_metrics_from_scores.py
Nguồn: BM1 4 mô hình (ref/win_ovo_env/win_ovo_orig: bm1_surv_scores/*.csv 5 seed; ref/single/scores/bm1_*.csv phân hoạch tự nhiên)
       11 biến thể lâm sàng (ref/survival_scores, results/survival_scores = Windows cũ; ref/single/scores)
Ra: experiments/results/ref/notebook_metrics.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

R = Path(__file__).resolve().parents[1] / "results"
out = {}


def metrics(y, s):
    p = (np.asarray(s) > 0).astype(int)
    return {"AUC": roc_auc_score(y, s), "F1": f1_score(y, p), "Precision": precision_score(y, p), "Recall": recall_score(y, p),
            "Accuracy": accuracy_score(y, p), "frac_pred_pos": float(p.mean())}


def multi(df, y):
    rows = [metrics(y, df[c].values) for c in df.columns if c.startswith("seed_")]
    return {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows], ddof=1))] for k in rows[0]}


# ---- BM1
lab = pd.read_csv(R / "ref/single/scores/bm1_dyam.csv", index_col=0)["label"]
bm = {}
for env, d in (("ref", "ref"), ("win_doanenv", "win_ovo_env"), ("win_doanorig", "win_ovo_orig")):
    for m in ("dyam", "ovo", "dyam_nlp", "ovo_nlp"):
        f = R / d / "bm1_surv_scores" / f"{m}.csv"
        df = pd.read_csv(f, index_col=0)
        bm[f"{env}|{m}|5seed"] = multi(df, lab.loc[df.index].values)
for m in ("dyam", "ovo", "dyam_nlp", "ovo_nlp"):
    d = pd.read_csv(R / "ref/single/scores" / f"bm1_{m}.csv", index_col=0)
    bm[f"ref|{m}|single_natural"] = metrics(d["label"].values, d["score"].values)
out["BM1"] = bm

# ---- biến thể lâm sàng
cv = {}
for env, d in (("ref", R / "ref/survival_scores"), ("win_old", R / "survival_scores")):
    for f in sorted(d.glob("*.csv")):
        df = pd.read_csv(f, index_col=0)
        cv[f"{env}|{f.stem}|5seed"] = multi(df.drop(columns="label"), df["label"].values)
for f in sorted((R / "ref/single/scores").glob("IHC-*.csv")):
    df = pd.read_csv(f, index_col=0)
    cv[f"ref|{f.stem}|single_natural"] = metrics(df["label"].values, df["score"].values)
out["clinical_variants"] = cv
(R / "ref/notebook_metrics.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def fmt(r):
    g = lambda k: f"{r[k][0]:.3f}±{r[k][1]:.3f}" if isinstance(r[k], list) else f"{r[k]:.3f}"
    return f"AUC {g('AUC')}  F1 {g('F1')}  Prec {g('Precision')}  Rec {g('Recall')}  Acc {g('Accuracy')}  predPos {g('frac_pred_pos')}"


for sec in out:
    print("==", sec)
    for k, v in out[sec].items():
        print(f"{k:55s} {fmt(v)}")
