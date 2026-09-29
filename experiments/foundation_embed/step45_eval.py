"""Step 4+5 — Phikon-embed vs GLCM thủ công: discovery-CV VÀ external (path_valid).

Câu hỏi chốt: embedding foundation model có generalize hơn GLCM ngoài mẫu không (GLCM ext ~0.744-0.767)?
Dùng LR (sạch, so công bằng) cho cả hai. Discovery-CV: 5-seed 10-fold. External: train discovery -> test path_valid.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data_setup import load_all                          # noqa: E402
from common.evaluate import save_result                         # noqa: E402
from validation.run_validation import valid_labels, boot_ci     # noqa: E402

DS = Path(__file__).resolve().parents[2].parent / "datasets"
LR_KW = dict(penalty="l2", C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs")


def cv_auc(X, y, seeds=(42, 7, 123, 2024, 31337), folds=10):
    """5-seed stratified k-fold LR, embedding KHÔNG scale mạnh (đã chuẩn hoá) -> RobustScaler nhẹ vẫn ok."""
    X = np.nan_to_num(X.values if hasattr(X, "values") else X)
    y = np.asarray(y)
    aucs = []
    for s in seeds:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=s)
        sc, lab = [], []
        for tr, te in skf.split(X, y):
            scaler = RobustScaler().fit(X[tr])
            clf = LogisticRegression(**LR_KW).fit(scaler.transform(X[tr]), y[tr])
            sc.extend(clf.decision_function(scaler.transform(X[te])))
            lab.extend(y[te])
        aucs.append(roc_auc_score(lab, sc))
    return float(np.mean(aucs)), float(np.std(aucs, ddof=1))


def ext_auc(Xtr, ytr, Xte, yte):
    cols = Xtr.columns.intersection(Xte.columns)
    sc = RobustScaler().fit(np.nan_to_num(Xtr[cols].values))
    clf = LogisticRegression(**LR_KW).fit(sc.transform(np.nan_to_num(Xtr[cols].values)), ytr)
    s = clf.decision_function(sc.transform(np.nan_to_num(Xte[cols].values)))
    lo, hi = boot_ci(yte, s)
    return float(roc_auc_score(yte, s)), (lo, hi), int(len(yte)), int(np.sum(yte))


def main():
    ctx = load_all()
    y_disc = ctx.df_outcomes["label"]
    _, path_lab = valid_labels()

    # ---- embeddings Phikon ----
    ph_disc = pd.read_parquet(DS / "path_fm_embed_discovery.parquet")   # index P-xxxx
    ph_val = pd.read_parquet(DS / "path_fm_embed_path_valid.parquet")   # index slide_id
    ph_disc = ph_disc[ph_disc.index.isin(y_disc.index)]
    yph = y_disc.loc[ph_disc.index]
    common_v = ph_val.index.intersection(path_lab.index.astype(str))
    ph_val_c = ph_val.loc[common_v]
    yph_val = path_lab.loc[common_v].values

    # ---- GLCM thủ công ----
    gl_disc = ctx.modality_dict["path_ihc_glcm"]
    present = ctx.modality_MASK.loc[gl_disc.index, "path_ihc_glcm"].values
    gl_disc = gl_disc[present]
    ygl = y_disc.loc[gl_disc.index]
    gl_val = pd.read_parquet(DS / "lung_pathology_pdl1_glcm_v3_validation.parquet")
    gl_val.index = gl_val.index.astype(str)
    gl_val = gl_val.select_dtypes(include=[np.number])
    common_g = gl_val.index.intersection(path_lab.index.astype(str))
    gl_val_c, ygl_val = gl_val.loc[common_g], path_lab.loc[common_g].values

    out = {"discovery_cv": {}, "external": {}}
    # ---- Step 4: discovery-CV ----
    out["discovery_cv"]["phikon"] = dict(zip(("mean", "sd"), cv_auc(ph_disc, yph)))
    out["discovery_cv"]["glcm"] = dict(zip(("mean", "sd"), cv_auc(gl_disc, ygl)))

    # ---- Step 5: external ----
    a, ci, n, pos = ext_auc(ph_disc, yph.values, ph_val_c, yph_val)
    out["external"]["phikon"] = {"auc": a, "ci": ci, "n_test": n, "pos": pos}
    a, ci, n, pos = ext_auc(gl_disc, ygl.values, gl_val_c, ygl_val)
    out["external"]["glcm"] = {"auc": a, "ci": ci, "n_test": n, "pos": pos}

    # ---- Step 5b: head-to-head trên CÙNG tập giao bệnh nhân (công bằng) ----
    inter = ph_val_c.index.intersection(gl_val_c.index)
    if len(inter) >= 10:
        yi = path_lab.loc[inter].values
        ap, cip, _, _ = ext_auc(ph_disc, yph.values, ph_val_c.loc[inter], yi)
        ag, cig, _, _ = ext_auc(gl_disc, ygl.values, gl_val_c.loc[inter], yi)
        out["external_intersect"] = {
            "n": int(len(inter)), "pos": int(np.sum(yi)),
            "phikon": {"auc": ap, "ci": cip}, "glcm": {"auc": ag, "ci": cig}}

    save_result("fm_pathology", out)
    print(f"\n{'=' * 72}\nPHIKON embed vs GLCM thủ công (LR)\n{'=' * 72}")
    print(f"{'':<10}{'discovery-CV (5seed)':>24}{'external (train->path_valid)':>30}")
    for m in ("phikon", "glcm"):
        d = out["discovery_cv"][m]; e = out["external"][m]
        dcv = f"{d['mean']:.4f}±{d['sd']:.4f}"
        ext = f"{e['auc']:.4f} [{e['ci'][0]:.2f},{e['ci'][1]:.2f}] n={e['n_test']}"
        print(f"{m:<10}{dcv:>24}{ext:>30}")
    print("=" * 72)
    dp, dg = out["external"]["phikon"]["auc"], out["external"]["glcm"]["auc"]
    print(f"\nĐIỂM CHỐT external: Phikon {dp:.4f} vs GLCM {dg:.4f} -> "
          f"{'Phikon VƯỢT' if dp > dg else 'Phikon KHÔNG vượt'} GLCM (Δ={dp-dg:+.4f})")


if __name__ == "__main__":
    main()
