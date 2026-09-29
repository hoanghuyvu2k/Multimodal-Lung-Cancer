"""Step 4+5 CT — CT-embed (BiomedCLIP) vs radiomics thủ công: discovery-CV VÀ external (rad_valid).

Câu hỏi chốt: CT-embed có phá được rào external ~0.46-0.60 của radiomics không?
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data_setup import load_all                          # noqa: E402
from common.evaluate import save_result                         # noqa: E402
from validation.run_validation import (                         # noqa: E402
    valid_labels, boot_ci, largest_lesion, numeric_features, read_any,
)

DS = Path(__file__).resolve().parents[2].parent.parent / "datasets"
DS = Path(__file__).resolve().parents[2].parent / "datasets"
LR_KW = dict(penalty="l2", C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs")


def cv_auc(X, y, k=None, seeds=(42, 7, 123, 2024, 31337), folds=10):
    X = np.nan_to_num(X.values if hasattr(X, "values") else X)
    y = np.asarray(y)
    aucs = []
    for s in seeds:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=s)
        sc, lab = [], []
        for tr, te in skf.split(X, y):
            Xtr, Xte = X[tr], X[te]
            if k and k < X.shape[1]:
                sel = SelectKBest(f_classif, k=k).fit(Xtr, y[tr])
                Xtr, Xte = Xtr[:, sel.get_support()], Xte[:, sel.get_support()]
            scaler = RobustScaler().fit(Xtr)
            clf = LogisticRegression(**LR_KW).fit(scaler.transform(Xtr), y[tr])
            sc.extend(clf.decision_function(scaler.transform(Xte)))
            lab.extend(y[te])
        aucs.append(roc_auc_score(lab, sc))
    return float(np.mean(aucs)), float(np.std(aucs, ddof=1))


def ext_auc(Xtr, ytr, Xte, yte, k=None):
    cols = Xtr.columns.intersection(Xte.columns) if hasattr(Xtr, "columns") else None
    if cols is not None:
        Xtr, Xte = Xtr[cols].values, Xte[cols].values
    Xtr, Xte = np.nan_to_num(Xtr), np.nan_to_num(Xte)
    if k and k < Xtr.shape[1]:
        sel = SelectKBest(f_classif, k=k).fit(Xtr, ytr)
        Xtr, Xte = Xtr[:, sel.get_support()], Xte[:, sel.get_support()]
    sc = RobustScaler().fit(Xtr)
    clf = LogisticRegression(**LR_KW).fit(sc.transform(Xtr), ytr)
    s = clf.decision_function(sc.transform(Xte))
    return float(roc_auc_score(yte, s)), boot_ci(yte, s), int(len(yte))


def main():
    ctx = load_all()
    y_disc = ctx.df_outcomes["label"]
    rad_lab, _ = valid_labels()

    # ---- CT-embed ----
    ct_d = pd.read_parquet(DS / "ct_fm_embed_discovery.parquet")
    ct_v = pd.read_parquet(DS / "ct_fm_embed_rad_valid.parquet")
    ct_d = ct_d[ct_d.index.isin(y_disc.index)]
    yct = y_disc.loc[ct_d.index]
    cvc = ct_v.index.intersection(rad_lab.index.astype(str))
    ct_v_c, yct_v = ct_v.loc[cvc], rad_lab.loc[cvc].values

    # ---- radiomics thủ công (largest lesion) ----
    rad_d = numeric_features(largest_lesion(read_any(
        "lung_radiomics_spacing1.0_mirpon_window1350.250_allimagetypes_bw20.parquet")))
    yrd = y_disc.reindex(rad_d.index)
    keep = yrd.notna()
    rad_d, yrd = rad_d[keep.values], yrd[keep]
    rad_v = numeric_features(largest_lesion(read_any(
        "lung_radiomics_spacing1.0_mirpon_window1350.250_allimagetypes_bw20_validation.parquet")))
    crv = rad_v.index.intersection(rad_lab.index.astype(str))
    rad_v_c, yrv = rad_v.loc[crv], rad_lab.loc[crv].values

    out = {"discovery_cv": {}, "external": {}}
    out["discovery_cv"]["ct_embed"] = dict(zip(("mean", "sd"), cv_auc(ct_d, yct)))
    out["discovery_cv"]["radiomics"] = dict(zip(("mean", "sd"), cv_auc(rad_d, yrd, k=30)))

    a, ci, n = ext_auc(ct_d, yct.values, ct_v_c, yct_v)
    out["external"]["ct_embed"] = {"auc": a, "ci": ci, "n": n}
    a, ci, n = ext_auc(rad_d, yrd.values, rad_v_c, yrv, k=30)
    out["external"]["radiomics"] = {"auc": a, "ci": ci, "n": n}

    # head-to-head external cùng bệnh nhân
    inter = ct_v_c.index.intersection(rad_v_c.index)
    if len(inter) >= 10:
        yi = rad_lab.loc[inter].values
        ac, cic, _ = ext_auc(ct_d, yct.values, ct_v_c.loc[inter], yi)
        ar, cir, _ = ext_auc(rad_d, yrd.values, rad_v_c.loc[inter], yi, k=30)
        out["external_intersect"] = {"n": int(len(inter)), "pos": int(np.sum(yi)),
                                     "ct_embed": {"auc": ac, "ci": cic},
                                     "radiomics": {"auc": ar, "ci": cir}}

    save_result("fm_ct", out)
    print(f"\n{'=' * 72}\nCT-embed (BiomedCLIP) vs radiomics thủ công (LR)\n{'=' * 72}")
    print(f"{'':<12}{'discovery-CV':>18}{'external':>26}")
    for m in ("ct_embed", "radiomics"):
        d = out["discovery_cv"][m]; e = out["external"][m]
        dcv = f"{d['mean']:.4f}±{d['sd']:.4f}"
        ext = f"{e['auc']:.4f} [{e['ci'][0]:.2f},{e['ci'][1]:.2f}] n={e['n']}"
        print(f"{m:<12}{dcv:>18}{ext:>26}")
    if "external_intersect" in out:
        it = out["external_intersect"]
        c, r = it["ct_embed"]["auc"], it["radiomics"]["auc"]
        print(f"\nhead-to-head external (n={it['n']}): CT-embed {c:.4f} vs radiomics {r:.4f} (delta={c-r:+.4f})")
    print("=" * 72)


if __name__ == "__main__":
    main()
