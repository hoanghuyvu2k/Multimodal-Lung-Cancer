"""Step 5 — Kiểm chứng ngoài mẫu (Hướng A). Train trên discovery, test trên cohort ngoài.

Thực tế dữ liệu (probe): cohort ngoài đơn-modality.
  rad_valid  (ID = did_acc)        : chỉ radiomics (46/50 có feature)
  path_valid (ID = pdl1_image_id)  : chỉ pathology glcm (52/71)
Nhãn lấy từ `bor` trong omnibus qua crosswalk ID.

Đơn-modality → fusion vô nghĩa; so hai họ: head neural (backbone) vs LR. Train toàn bộ discovery
(bệnh nhân có modality đó), test 1 lần trên cohort ngoài. AUC + 95% CI bootstrap trên bệnh nhân test.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import benchmarks as B          # noqa: E402
from common.data_setup import load_all, BASE_DB_DIR  # noqa: E402
from common.evaluate import load_result, save_result  # noqa: E402

from lung_helpers import RAD_JOB_TAG        # noqa: E402
from baselines.model_uniform_avg import MultiModalDynamicModelUncertainty  # noqa: E402

BASE = Path(BASE_DB_DIR)
LR_KW = dict(penalty="l2", C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs")
VOL = "original_shape_MeshVolume"


def read_any(*names):
    for n in names:
        p = BASE / n
        if p.exists():
            return pd.read_parquet(p)
    return None


def valid_labels():
    """Nhãn cohort ngoài: bor 1/2 -> 0 (đáp ứng), còn lại 1. Crosswalk qua omnibus."""
    # did_acc/radiology_accession_number là số thuần -> pandas ép float ("190310.0"),
    # lệch với index parquet ("190310"). Đọc các cột ID bằng converter str để khớp chính xác.
    id_cols = {c: str for c in ["did_acc", "radiology_accession_number",
                                "pdl1_image_id", "slide_id"]}
    omni = pd.read_csv(BASE / "18193mskmindprojectm-omnibusinventory_data_2021-12-20_1540"
                              "-with-tb-and-scanner.csv", low_memory=False, converters=id_cols)
    omni["label"] = 1
    omni.loc[omni["bor"].isin([1, 2]), "label"] = 0
    rad = omni.dropna(subset=["did_acc"]).copy()
    rad.index = rad["did_acc"].astype(str)
    path = omni.dropna(subset=["pdl1_image_id"]).copy()
    path.index = path["pdl1_image_id"].astype(str)
    return rad["label"], path["label"]


def largest_lesion(df):
    """1 tổn thương/bệnh nhân: filtered-radiomics, chọn lesion lớn nhất theo MeshVolume."""
    d = df[df["job_tag"] == RAD_JOB_TAG].copy()
    d.index = d.index.astype(str)
    if VOL in d.columns:
        d = d.sort_values(VOL, ascending=False)
    return d[~d.index.duplicated(keep="first")]


def numeric_features(df):
    drop = {"job_tag", "site"}
    return df.drop(columns=[c for c in df.columns if c in drop], errors="ignore") \
             .select_dtypes(include=[np.number])


def select_k(Xtr, ytr, Xte, k):
    """SelectKBest (ANOVA F) fit CHỈ trên discovery train, áp cho cả hai. Cho radiomics chiều cao."""
    cols = Xtr.columns.intersection(Xte.columns)
    Xtr, Xte = Xtr[cols], Xte[cols]
    k = min(k, len(cols))
    sel = SelectKBest(f_classif, k=k).fit(np.nan_to_num(Xtr.values), ytr)
    keep = cols[sel.get_support()]
    return Xtr[keep], Xte[keep]


def boot_ci(y, s, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); s = np.asarray(s); m = len(y)
    vals = []
    for _ in range(n):
        pick = rng.integers(0, m, m)
        if len(np.unique(y[pick])) < 2:
            continue
        vals.append(roc_auc_score(y[pick], s[pick]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def eval_pair(Xtr, ytr, Xte, yte):
    """Train head neural + LR trên discovery, test ngoài. Trả dict AUC + CI cho cả hai."""
    cols = Xtr.columns.intersection(Xte.columns)
    Xtr, Xte = Xtr[cols], Xte[cols]
    sc = RobustScaler()
    Atr = np.nan_to_num(sc.fit_transform(Xtr.values))
    Ate = np.nan_to_num(sc.transform(Xte.values))

    # LR
    lr = LogisticRegression(**LR_KW).fit(Atr, ytr)
    s_lr = lr.decision_function(Ate)
    auc_lr = roc_auc_score(yte, s_lr)

    # Neural head (đơn modality, mask toàn 1)
    params = {k: v for k, v in B.MODEL_PARAMS.items() if k != "cross_modality_enabled"}
    clf = MultiModalDynamicModelUncertainty(**params)
    clf.fit([Xtr.values], np.ones((len(Xtr), 1), dtype=int), pd.Series(ytr, index=Xtr.index))
    s_nn = clf.predict_proba([Xte.values], np.ones((len(Xte), 1), dtype=int))
    auc_nn = roc_auc_score(yte, s_nn)

    return {
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)), "n_feat": int(len(cols)),
        "pos_test": int(np.sum(yte)),
        "neural": {"auc": float(auc_nn), "ci": boot_ci(yte, s_nn)},
        "lr": {"auc": float(auc_lr), "ci": boot_ci(yte, s_lr)},
    }


def main():
    ctx = load_all()
    rad_lab, path_lab = valid_labels()
    out = {}

    # ---------- PATHOLOGY: discovery path_ihc_glcm -> path_valid ----------
    print("\n=== PATHOLOGY (discovery -> path_valid) ===", flush=True)
    Xtr_p = ctx.modality_dict["path_ihc_glcm"]
    ytr_p = ctx.df_outcomes.loc[Xtr_p.index, "label"]
    present = ctx.modality_MASK.loc[Xtr_p.index, "path_ihc_glcm"].values
    Xtr_p, ytr_p = Xtr_p[present], ytr_p[present].values

    val_p = numeric_features(read_any("lung_pathology_pdl1_glcm_v3_validation.parquet"))
    val_p.index = val_p.index.astype(str)
    common = val_p.index.intersection(path_lab.index)
    Xte_p = val_p.loc[common]
    yte_p = path_lab.loc[common].values
    out["pathology"] = eval_pair(Xtr_p, ytr_p, Xte_p, yte_p)
    print(f"  {out['pathology']}", flush=True)
    save_result("validation", out)

    # ---------- RADIOMICS: discovery combined -> rad_valid ----------
    print("\n=== RADIOMICS (discovery -> rad_valid) ===", flush=True)
    disc_rad = largest_lesion(read_any(
        "lung_radiomics_spacing1.0_mirpon_window1350.250_allimagetypes_bw20.parquet"))
    val_rad = largest_lesion(read_any(
        "lung_radiomics_spacing1.0_mirpon_window1350.250_allimagetypes_bw20_validation.parquet"))
    Xtr_r = numeric_features(disc_rad)
    ytr_r = ctx.df_outcomes.reindex(Xtr_r.index)["label"]
    keep = ytr_r.notna()
    Xtr_r, ytr_r = Xtr_r[keep.values], ytr_r[keep].values

    Xte_r = numeric_features(val_rad)
    common_r = Xte_r.index.intersection(rad_lab.index)
    Xte_r = Xte_r.loc[common_r]
    yte_r = rad_lab.loc[common_r].values
    # Radiomics 1689 feature / 187 bệnh nhân -> overfit. Chọn k=30 (≈ số feature sau L1 của
    # discovery) bằng SelectKBest fit trên discovery train, để so công bằng với cách pipeline mô hình hoá.
    Xtr_r_sel, Xte_r_sel = select_k(Xtr_r, ytr_r, Xte_r, k=30)
    out["radiomics"] = eval_pair(Xtr_r_sel, ytr_r, Xte_r_sel, yte_r)
    out["radiomics_allfeat"] = eval_pair(Xtr_r, ytr_r, Xte_r, yte_r)  # tham chiếu (không chọn feature)
    print(f"  sel: {out['radiomics']}\n  all: {out['radiomics_allfeat']}", flush=True)
    save_result("validation", out)

    # ---------- Bảng discovery-CV vs external ----------
    audit = load_result("signal_audit")
    disc_ref = {
        "pathology": ("path_ihc_glcm", audit),
        "radiomics": ("rad_lesion_lu", audit),   # bản radiomics gộp gần nhất
    }
    print(f"\n\n{'=' * 82}\nDISCOVERY-CV vs EXTERNAL (AUC)\n{'=' * 82}")
    print(f"{'modality':<12}{'disc uniform':>14}{'disc LR':>10}"
          f"{'ext neural':>22}{'ext LR':>22}{'n_test':>8}")
    for mod in ("pathology", "radiomics"):
        key = disc_ref[mod][0]
        du = audit["uniform_avg"][key]["mean_auc"]
        dl = audit["lr"][key]["mean_auc"]
        e = out[mod]
        en = f"{e['neural']['auc']:.3f}[{e['neural']['ci'][0]:.2f},{e['neural']['ci'][1]:.2f}]"
        el = f"{e['lr']['auc']:.3f}[{e['lr']['ci'][0]:.2f},{e['lr']['ci'][1]:.2f}]"
        print(f"{mod:<12}{du:>14.4f}{dl:>10.4f}{en:>22}{el:>22}{e['n_test']:>8}")
    print("=" * 82)
    save_result("validation", out)


if __name__ == "__main__":
    main()
