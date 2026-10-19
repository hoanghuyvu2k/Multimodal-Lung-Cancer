"""Kiểm chứng "NLP có cải thiện sống còn không" bằng nhiều cách (ref-linux-py38). Đọc điểm nguy cơ đã lưu, không train lại DyAM (trừ phần D).

  A  từng seed (5): C-index, HR/1SD (z-score, điều chỉnh hiệp biến) -> hiệu NLP - không LS, NLP - Labs, số seed dương
  B  ensemble 5 seed + bootstrap bệnh nhân ghép cặp (2000): Δ C-index, Δ log HR/1SD
  C  giá trị tăng thêm: Cox(hiệp biến + điểm không LS) so với (+ điểm NLP) và ngược lại: Δ log-lik, LRT
  D  trực tiếp, không qua DyAM: Cox phạt OOF từ Labs số / NLP-PCA16 / NLP raw (PCA64), 5 x KFold(10) -> C-index OOF

    run_ref.sh experiments/reference_env/survival_nlp_checks_ref.py
Kết quả: experiments/results/ref/survival_nlp_checks.json
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1])); sys.path.insert(0, str(_HERE.parents[2]))
from nlp_verify import verify_nlp_5seed as V                       # noqa: E402

import os
RES = _HERE.parents[1] / "results" / os.environ.get("CHK_OUT", "ref")
RES.mkdir(parents=True, exist_ok=True)
SC = _HERE.parents[1] / "results" / os.environ.get("CHK_SC", "ref/survival_scores")
COVS = ["age", "ecog", "albumin", "dnlr", "liver_mets"]
clin = V.load_clinical()
SEEDS = ["seed_42", "seed_7", "seed_123", "seed_2024", "seed_31337"]
PAIRS = {"IHC-A": [("nlp_raw", "no_clinical"), ("nlppca16", "no_clinical"), ("nlp_raw", "labs"), ("nlppca16", "labs")],
         "IHC-G": [("nlp_raw", "no_clinical"), ("nlppca16", "no_clinical"), ("nlp_raw", "labs"), ("nlppca16", "labs")]}
out = {}


def z(s):
    return (s - s.mean()) / s.std()


def cox_hr(score, idx, covs=True):
    d = pd.DataFrame({"s": z(pd.Series(score, index=idx))}).join(clin[COVS + ["pfs", "pfs_censor"]] if covs else clin[["pfs", "pfs_censor"]]).dropna()
    c = CoxPHFitter(penalizer=0.0).fit(d, "pfs", "pfs_censor")
    return float(c.params_["s"])


def cidx(score, idx):
    d = pd.DataFrame({"s": score}, index=idx).join(clin[["pfs", "pfs_censor"]]).dropna()
    return float(concordance_index(d["pfs"], -d["s"], d["pfs_censor"]))


def load(arm, v):
    return pd.read_csv(SC / f"{arm}__{v}.csv", index_col=0)


# ---------------------------------------------------------------- A: từng seed
A = {}
for arm in ("IHC-A", "IHC-G"):
    vs = ["no_clinical", "labs", "nlp_raw", "nlppca16"]
    tab = {}
    for v in vs:
        d = load(arm, v)
        tab[v] = {"c": [cidx(d[s].values, d.index) for s in SEEDS], "logHR": [cox_hr(d[s].values, d.index) for s in SEEDS]}
    A[arm] = {"per_variant": {v: {k: [round(x, 4) for x in tab[v][k]] for k in tab[v]} for v in vs}, "diff": {}}
    for a, b in PAIRS[arm]:
        for m in ("c", "logHR"):
            dd = np.array(tab[a][m]) - np.array(tab[b][m])
            A[arm]["diff"][f"{a}-{b}:{m}"] = {"mean": round(float(dd.mean()), 4), "sd": round(float(dd.std(ddof=1)), 4), "n_pos": int((dd > 0).sum()), "of": len(dd)}
out["A_per_seed"] = A

# ---------------------------------------------------------------- B: ensemble + bootstrap ghép cặp
rng = np.random.default_rng(0)
B = {}
for arm in ("IHC-A", "IHC-G"):
    ens = {}
    for v in ("no_clinical", "labs", "nlp_raw", "nlppca16"):
        d = load(arm, v)
        ens[v] = z(d[SEEDS]).mean(axis=1) if False else d[SEEDS].apply(z).mean(axis=1)
    idx = ens["no_clinical"].index
    B[arm] = {}
    for a, b in PAIRS[arm]:
        obs_c = cidx(ens[a].values, idx) - cidx(ens[b].values, idx)
        obs_h = cox_hr(ens[a].values, idx) - cox_hr(ens[b].values, idx)
        dc, dh = [], []
        for _ in range(1000):
            i = rng.integers(0, len(idx), len(idx))
            ia = idx[i]
            # hàng trùng chỉ số làm lifelines lỗi -> dùng chỉ số vị trí qua reset
            sa, sb = ens[a].values[i], ens[b].values[i]
            t = clin.loc[ia, ["pfs", "pfs_censor"]].reset_index(drop=True)
            try:
                dc.append(concordance_index(t["pfs"], -sa, t["pfs_censor"]) - concordance_index(t["pfs"], -sb, t["pfs_censor"]))
                cv = clin.loc[ia, COVS].reset_index(drop=True)
                fa = pd.concat([pd.Series(sa).pipe(z).rename("s"), cv, t], axis=1).dropna()
                fb = pd.concat([pd.Series(sb).pipe(z).rename("s"), cv, t], axis=1).dropna()
                dh.append(CoxPHFitter().fit(fa, "pfs", "pfs_censor").params_["s"] - CoxPHFitter().fit(fb, "pfs", "pfs_censor").params_["s"])
            except Exception:
                continue
        B[arm][f"{a}-{b}"] = {"dC": round(obs_c, 4), "dC_ci": [round(float(x), 4) for x in np.percentile(dc, [2.5, 97.5])], "dC_pos_frac": round(float(np.mean(np.array(dc) > 0)), 3),
                              "dlogHR": round(obs_h, 4), "dlogHR_ci": [round(float(x), 4) for x in np.percentile(dh, [2.5, 97.5])], "dlogHR_pos_frac": round(float(np.mean(np.array(dh) > 0)), 3), "n_boot": len(dc)}
        print("B", arm, a, b, B[arm][f"{a}-{b}"], flush=True)
out["B_bootstrap"] = B

# ---------------------------------------------------------------- C: giá trị tăng thêm (Cox, ensemble)
C = {}
for arm in ("IHC-A", "IHC-G"):
    ens = {v: load(arm, v)[SEEDS].apply(z).mean(axis=1) for v in ("no_clinical", "labs", "nlp_raw", "nlppca16")}
    C[arm] = {}
    for nlp in ("nlp_raw", "nlppca16"):
        d = pd.DataFrame({"base": z(ens["no_clinical"]), "nlp": z(ens[nlp])}).join(clin[COVS + ["pfs", "pfs_censor"]]).dropna()
        m0 = CoxPHFitter().fit(d.drop(columns="nlp"), "pfs", "pfs_censor")
        m1 = CoxPHFitter().fit(d, "pfs", "pfs_censor")
        m2 = CoxPHFitter().fit(d.drop(columns="base"), "pfs", "pfs_censor")
        lr_add = 2 * (m1.log_likelihood_ - m0.log_likelihood_)
        lr_rev = 2 * (m1.log_likelihood_ - m2.log_likelihood_)
        C[arm][nlp] = {"add_nlp_to_noclin": {"LR": round(lr_add, 3), "p": round(float(stats.chi2.sf(lr_add, 1)), 4), "coef": round(float(m1.params_["nlp"]), 3)},
                       "add_noclin_to_nlp": {"LR": round(lr_rev, 3), "p": round(float(stats.chi2.sf(lr_rev, 1)), 4), "coef": round(float(m1.params_["base"]), 3)},
                       "corr_scores": round(float(np.corrcoef(d["base"], d["nlp"])[0, 1]), 3)}
        print("C", arm, nlp, C[arm][nlp], flush=True)
out["C_incremental"] = C

# ---------------------------------------------------------------- D: trực tiếp từ Labs số vs NLP (không DyAM)
from common.data_setup import load_all                               # noqa: E402
from nlp_verify.nlp_modality import add_nlp_modality                 # noqa: E402
from nlp_verify.verify_nlp_variants import add_pca16                 # noqa: E402

ctx = load_all(); add_nlp_modality(ctx); add_pca16(ctx, "cnl_nlp", "cnl_nlp_pca16")
md = ctx.modality_dict
idx = ctx.df_outcomes.index
feats = {"labs_numeric(13)": md["cnl_dem_labs"].reindex(idx), "nlp_pca16": md["cnl_nlp_pca16"].reindex(idx), "nlp_raw(384)": md["cnl_nlp"].reindex(idx)}
T = clin.reindex(idx)[["pfs", "pfs_censor"]]
D = {}
for name, X in feats.items():
    X = X.copy(); X.columns = [f"f{i}" for i in range(X.shape[1])]
    for pen in (0.1, 1.0, 10.0):
        res = []
        for seed in range(5):
            oof = pd.Series(index=idx, dtype=float)
            for tr, te in KFold(10, shuffle=True, random_state=seed).split(idx):
                sc = RobustScaler().fit(X.iloc[tr]) if name.startswith("labs") else None
                Xtr = X.iloc[tr].fillna(X.iloc[tr].median()); Xte = X.iloc[te].fillna(X.iloc[tr].median())
                if sc is not None:
                    Xtr = pd.DataFrame(sc.transform(Xtr), index=Xtr.index, columns=Xtr.columns); Xte = pd.DataFrame(sc.transform(Xte), index=Xte.index, columns=Xte.columns)
                if name.startswith("nlp_raw"):
                    p = PCA(32, random_state=0).fit(Xtr); Xtr = pd.DataFrame(p.transform(Xtr), index=Xtr.index); Xte = pd.DataFrame(p.transform(Xte), index=Xte.index)
                    Xtr.columns = Xte.columns = [f"f{i}" for i in range(Xtr.shape[1])]
                dtr = Xtr.join(T.iloc[tr])
                try:
                    cph = CoxPHFitter(penalizer=pen).fit(dtr, "pfs", "pfs_censor")
                    oof.iloc[te] = cph.predict_partial_hazard(Xte).values
                except Exception:
                    oof.iloc[te] = np.nan
            ok = oof.notna()
            res.append(float(concordance_index(T["pfs"][ok], -oof[ok], T["pfs_censor"][ok])))
        D[f"{name}|pen={pen}"] = {"mean": round(float(np.mean(res)), 4), "sd": round(float(np.std(res, ddof=1)), 4)}
        print("D", name, pen, D[f"{name}|pen={pen}"], flush=True)
out["D_direct_oof_cindex"] = D
(RES / "survival_nlp_checks.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("XONG")
