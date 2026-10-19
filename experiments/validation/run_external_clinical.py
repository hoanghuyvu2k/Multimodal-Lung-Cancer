"""Kiểm chứng NGOÀI bằng rad_valid + path_valid (2026-10-06) — kế hoạch ở ledger.md (định trước khi chạy).

  A  mã hoá lâm sàng rút gọn 6 biến: raw6 vs NLP6, LR, test trên gộp rad_valid ∪ path_valid
  B  ba nguồn trên path_valid: IHC-G (GLCM) + PD-L1 lâm sàng + NLP6, uniform_avg vs DyAM vs các nguồn riêng

    PYTHONIOENCODING=utf-8 python experiments/validation/run_external_clinical.py
"""
import json
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
_CODE = _EXP.parent
for _p in (str(_CODE), str(_EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
import torch                                                      # noqa: E402
from sklearn.linear_model import LogisticRegression               # noqa: E402
from sklearn.metrics import roc_auc_score                         # noqa: E402
from sklearn.model_selection import KFold                         # noqa: E402
from sklearn.preprocessing import RobustScaler                    # noqa: E402

from archloop import harness as H                                 # noqa: E402,F401  (vá scipy)
from archloop import cands as C                                   # noqa: E402
import baselines.model_uniform_avg as mua                         # noqa: E402
from common import benchmarks as B                                # noqa: E402
from common.data_setup import load_all, BASE_DB_DIR               # noqa: E402
import lung_helpers as LH                                         # noqa: E402

BASE = Path(BASE_DB_DIR)
VARS = ["age", "pack_years", "ecog", "albumin", "dnlr", "hist_adeno"]
INIT_SEEDS = [42, 7, 123, 2024, 31337]
N_BOOT = 5000
OUT = _CODE / "experiments" / "results" / "external_clinical.json"


def prompts6(df):
    out = []
    for _, r in df.iterrows():
        hist = "adenocarcinoma" if r["hist_adeno"] == 1 else "non-adenocarcinoma"
        out.append(f"The patient is a {int(r['age'])}-year-old. Smoking history in pack-years is {r['pack_years']}. "
                   f"ECOG performance status is {r['ecog']}. Blood albumin concentration is {r['albumin']:.2f}. "
                   f"Derived neutrophil-to-lymphocyte ratio (dNLR) is {r['dnlr']:.2f}. "
                   f"The patient is diagnosed with {hist}.")
    return out


def external_rows():
    """Hàng omnibus của hai cohort ngoài (loại trùng). Trả DataFrame có cột cohort, label và các biến."""
    conv = {c: str for c in ["did_acc", "radiology_accession_number", "pdl1_image_id", "slide_id"]}
    om = pd.read_csv(BASE / "18193mskmindprojectm-omnibusinventory_data_2021-12-20_1540-with-tb-and-scanner.csv",
                     low_memory=False, converters=conv)
    coh = pd.read_csv(BASE / "final_cohort_listing.csv")
    coh["main_index"] = coh["main_index"].astype(str)
    pv = set(coh[coh.cohort == "path_valid"].main_index)
    rv = set(coh[coh.cohort == "rad_valid"].main_index)
    om["is_path"] = om["pdl1_image_id"].isin(pv)
    om["is_rad"] = om["did_acc"].isin(rv)
    ext = om[om.is_path | om.is_rad].copy()
    ext["label"] = np.where(ext["bor"].isin([1, 2]), 0, 1)
    ext["cohort"] = np.where(ext.is_path, "path_valid", "rad_valid")
    ext["pdl1"] = -pd.to_numeric(ext["clinical_pdl1_score"], errors="coerce") / 100.0
    ext["path_key"] = ext["pdl1_image_id"].where(ext.is_path)
    print(f"cohort ngoài: path_valid {int(ext.is_path.sum())}, rad_valid {int(ext.is_rad.sum())}, "
          f"hai hàng thuộc cả hai: {int((ext.is_path & ext.is_rad).sum())}, gộp {len(ext)}", flush=True)
    return ext


def boot_auc(y, s, rng):
    y, s = np.asarray(y), np.asarray(s)
    v = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            v.append(roc_auc_score(y[i], s[i]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def boot_delta(y, sa, sb, seed=0):
    rng = np.random.default_rng(seed)
    y, sa, sb = (np.asarray(x) for x in (y, sa, sb))
    d, base = [], roc_auc_score(y, sa) - roc_auc_score(y, sb)
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            d.append(roc_auc_score(y[i], sa[i]) - roc_auc_score(y[i], sb[i]))
    d = np.array(d)
    p = float(2 * min((d <= 0).mean(), (d >= 0).mean()))
    return float(base), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], p


def fit_lr(Xtr, ytr, Xte, grid=(0.01, 0.1, 1.0, 10.0), scale=True):
    """Chọn C bằng CV 10-fold TRONG discovery; huấn luyện lại trên toàn bộ discovery; trả (điểm test, C, AUC CV)."""
    sc = RobustScaler().fit(Xtr) if scale else None
    A = sc.transform(Xtr) if scale else Xtr
    T = sc.transform(Xte) if scale else Xte
    best = None
    for c in grid:
        oof = np.zeros(len(ytr))
        for tr, te in KFold(10, shuffle=True, random_state=0).split(A):
            m = LogisticRegression(C=c, class_weight="balanced", max_iter=3000).fit(A[tr], ytr[tr])
            oof[te] = m.decision_function(A[te])
        a = roc_auc_score(ytr, oof)
        if best is None or a > best[1]:
            best = (c, a)
    m = LogisticRegression(C=best[0], class_weight="balanced", max_iter=3000).fit(A, ytr)
    return m.decision_function(T), best[0], best[1]


def embed(prompts):
    from clinical_nlp_embedding import ClinicalTextEmbedder
    return np.asarray(ClinicalTextEmbedder().model.encode(prompts, show_progress_bar=False))


def fit_predict(kind, cols, tr, te, seed):
    """Huấn luyện mô hình dự án trên TOÀN BỘ discovery; dự đoán cohort ngoài. cols = chỉ số nguồn dùng."""
    Xtr = [tr["X"][i] for i in cols]; Xte = [te["X"][i] for i in cols]
    Mtr = tr["M"][:, cols]; Mte = te["M"][:, cols]
    no_scale = [k for k, i in enumerate(cols) if tr["names"][i] == "nlp6"]
    params = {k: v for k, v in B.MODEL_PARAMS.items() if k != "cross_modality_enabled"}
    y = pd.Series(tr["y"], index=range(len(tr["y"])))
    if kind == "uniform":
        clf = mua.MultiModalDynamicModelUncertainty(**{**params, "no_scale": no_scale, "init_seed": seed})
    else:
        LH.set_global_seed(seed)
        clf = LH.MultiModalDynamicModel(epochs=params["epochs"], lr=params["lr"], alpha=params["alpha"],
                                        beta=params["beta"], cross_modality_enabled=False, no_scale=no_scale)
    clf.fit(Xtr, Mtr.astype(int), y)
    return np.asarray(clf.predict_proba(Xte, Mte.astype(int)), dtype=float)


def main():
    torch.set_num_threads(2)
    ctx = load_all()
    ext = external_rows()
    rng = np.random.default_rng(0)
    res = {"n_external": int(len(ext))}

    lab = ctx.modality_dict["cnl_dem_labs"].reindex(ctx.df_outcomes.index)[VARS]
    ytr = ctx.df_outcomes["label"].astype(float).values
    med = lab.median()
    Dtr = lab.fillna(med)
    Dte = ext[VARS].apply(pd.to_numeric, errors="coerce").fillna(med)
    print("trung bình discovery vs ngoài (kiểm tra đơn vị/mã hoá):\n",
          pd.DataFrame({"discovery": Dtr.mean(), "ngoài": Dte.mean()}).round(2).to_string(), flush=True)

    # ---------------- A: mã hoá lâm sàng ----------------
    Etr, Ete = embed(prompts6(Dtr)), embed(prompts6(Dte))
    s_raw, c_raw, cv_raw = fit_lr(Dtr.values, ytr, Dte.values)
    s_nlp, c_nlp, cv_nlp = fit_lr(Etr, ytr, Ete, scale=False)
    y = ext["label"].values
    A = {"n": int(len(y)), "n_pos": int(y.sum())}
    for name, s, c, cv in (("raw6", s_raw, c_raw, cv_raw), ("nlp6", s_nlp, c_nlp, cv_nlp)):
        A[name] = {"auc": float(roc_auc_score(y, s)), "ci": boot_auc(y, s, rng), "C": c, "cv_discovery_auc": float(cv)}
        for co in ("path_valid", "rad_valid"):
            m = (ext.cohort == co).values
            A[name][co] = float(roc_auc_score(y[m], s[m]))
    d, ci, p = boot_delta(y, s_nlp, s_raw)
    A["nlp6_minus_raw6"] = {"delta": d, "ci": ci, "p": p}
    res["A"] = A
    print(f"\n[A] n={A['n']} (đáp ứng-không {int((1-y).sum())}/{int(y.sum())})")
    for k in ("raw6", "nlp6"):
        print(f"  {k}: ngoài AUC {A[k]['auc']:.3f} [{A[k]['ci'][0]:.3f}, {A[k]['ci'][1]:.3f}]  path_valid {A[k]['path_valid']:.3f}  "
              f"rad_valid {A[k]['rad_valid']:.3f}  | CV discovery {A[k]['cv_discovery_auc']:.3f}  C={A[k]['C']}")
    print(f"  NLP6 − raw6 = {d:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] p={p:.3f}", flush=True)

    # ---------------- B: ba nguồn trên path_valid ----------------
    glcm_tr = ctx.modality_dict["path_ihc_glcm"].reindex(ctx.df_outcomes.index)
    m_gl = ctx.modality_MASK["path_ihc_glcm"].reindex(ctx.df_outcomes.index).fillna(False).values
    pdl_tr = C._clinical_pdl1().reindex(ctx.df_outcomes.index)
    pv = ext[ext.is_path].copy()
    vp = pd.read_parquet(BASE / "lung_pathology_pdl1_glcm_v3_validation.parquet").select_dtypes(include=[np.number])
    vp.index = vp.index.astype(str)
    cols = glcm_tr.columns.intersection(vp.columns)
    glcm_te = vp.reindex(pv["pdl1_image_id"].values)[cols]
    glcm_te.index = pv.index
    m_gl_te = glcm_te.notna().all(axis=1).values
    print(f"\n[B] path_valid n={len(pv)}; có GLCM {int(m_gl_te.sum())}; có PD-L1 {int(pv.pdl1.notna().sum())}; "
          f"cột GLCM chung {len(cols)}", flush=True)
    nlp_tr = pd.DataFrame(Etr, index=ctx.df_outcomes.index)
    nlp_te = pd.DataFrame(embed(prompts6(pv[VARS].apply(pd.to_numeric, errors="coerce").fillna(med))), index=pv.index)
    tr = {"names": ["glcm", "pdl1", "nlp6"],
          "X": [glcm_tr[cols].values, pdl_tr.values.reshape(-1, 1), nlp_tr.values],
          "M": np.column_stack([m_gl, pdl_tr.notna().values, np.ones(len(ytr), bool)]), "y": ytr}
    te = {"X": [glcm_te.values, pv[["pdl1"]].values, nlp_te.values],
          "M": np.column_stack([m_gl_te, pv.pdl1.notna().values, np.ones(len(pv), bool)])}
    yv = pv["label"].values
    arms = {"PD-L1": ("uniform", [1]), "GLCM": ("uniform", [0]), "NLP6": ("uniform", [2]),
            "GLCM+PD-L1": ("uniform", [0, 1]), "GLCM+PD-L1+NLP6": ("uniform", [0, 1, 2]),
            "DyAM GLCM+PD-L1+NLP6": ("dyam", [0, 1, 2])}
    sc, Bres = {}, {}
    for name, (kind, ix) in arms.items():
        sc[name] = np.mean([fit_predict(kind, ix, tr, te, s) for s in INIT_SEEDS], axis=0)
        ok = np.ones(len(yv), bool) if 0 not in ix or len(ix) > 1 else m_gl_te      # GLCM một mình: chỉ 52 người có đặc trưng
        Bres[name] = {"n": int(ok.sum()), "auc": float(roc_auc_score(yv[ok], sc[name][ok])),
                      "ci": boot_auc(yv[ok], sc[name][ok], rng)}
        print(f"  {name:<24} n={Bres[name]['n']:>3}  AUC {Bres[name]['auc']:.3f} [{Bres[name]['ci'][0]:.3f}, {Bres[name]['ci'][1]:.3f}]", flush=True)
    for a, b in (("GLCM+PD-L1+NLP6", "PD-L1"), ("GLCM+PD-L1+NLP6", "GLCM+PD-L1"),
                 ("GLCM+PD-L1+NLP6", "DyAM GLCM+PD-L1+NLP6"), ("GLCM+PD-L1", "PD-L1")):
        d, ci, p = boot_delta(yv, sc[a], sc[b])
        Bres[f"{a} − {b}"] = {"delta": d, "ci": ci, "p": p}
        print(f"  {a} − {b}: {d:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] p={p:.3f}", flush=True)
    res["B"] = {"n": int(len(pv)), "n_pos": int(yv.sum()), "arms": Bres}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("XONG ->", OUT)


if __name__ == "__main__":
    main()
