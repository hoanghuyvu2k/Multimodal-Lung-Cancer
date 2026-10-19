"""R5 — kiểm chứng NGOÀI một phần cho chuyên gia genomics bằng cohort MSK-IMPACT miễn dịch công khai (Samstein 2019, cBioPortal `tmb_mskcc_2018`).
Quy tắc định trước ở ledger.md: chỉ TMB + đột biến "bất kỳ không đồng nghĩa" ở 6 gen (EGFR, ERBB2, BRAF, MET, STK11, ARID1A); KHÔNG có CNA nên mất 4 đặc trưng khuếch đại;
KHÔNG đồng nhất với `driver_binarized`. Loại mọi bệnh nhân trùng discovery. Huấn luyện trên discovery (nhãn đáp ứng), áp lên Samstein NSCLC, đánh giá bằng SỐNG CÒN (OS).

    PYTHONIOENCODING=utf-8 python experiments/archloop/analyze_samstein.py <thư mục có tmb_mskcc_2018_data_clinical_*.txt và mut6.json>
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
_EXP = Path(__file__).resolve().parents[1]
_CODE = _EXP.parent
for _p in (str(_CODE), str(_EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from lifelines import CoxPHFitter                                 # noqa: E402
from lifelines.statistics import logrank_test                     # noqa: E402
from lifelines.utils import concordance_index                     # noqa: E402
from sklearn.linear_model import LogisticRegression               # noqa: E402
from sklearn.metrics import roc_auc_score                         # noqa: E402
from sklearn.model_selection import KFold                         # noqa: E402
from sklearn.preprocessing import RobustScaler                    # noqa: E402

from common import data_setup as ds                               # noqa: E402

GENES = {1956: "EGFR", 2064: "ERBB2", 673: "BRAF", 4233: "MET", 6794: "STK11", 8289: "ARID1A"}
N_BOOT = 3000


def main():
    D = Path(sys.argv[1])
    rd = lambda f: pd.read_csv(D / f, sep="\t", comment="#", dtype=str)   # noqa: E731
    ps, ss = rd("tmb_mskcc_2018_data_clinical_patient.txt"), rd("tmb_mskcc_2018_data_clinical_sample.txt")
    coh = pd.read_csv(f"{ds.BASE_DB_DIR}/final_cohort_listing.csv").set_index("main_index")
    mine = set(coh.index.astype(str))
    nsc = ss[ss["CANCER_TYPE"].str.contains("Non-Small", na=False)].copy()
    nsc["TMB"] = pd.to_numeric(nsc["TMB_NONSYNONYMOUS"], errors="coerce")
    all_overlap = set(ps.PATIENT_ID) & mine
    nsc = nsc[~nsc.PATIENT_ID.isin(all_overlap)]
    muts = json.load(open(D / "mut6.json"))
    hit = {(m["patientId"], GENES[m["entrezGeneId"]]) for m in muts}
    for g in GENES.values():
        nsc[g] = [1.0 if (p, g) in hit else 0.0 for p in nsc.PATIENT_ID]
    nsc = nsc.merge(ps, on="PATIENT_ID")
    nsc["T"] = pd.to_numeric(nsc["OS_MONTHS"])
    nsc["E"] = (nsc["OS_STATUS"] == "1:DECEASED").astype(int)
    print(f"Samstein NSCLC không trùng: {len(nsc)} bệnh nhân (đã loại {len(all_overlap)} ID trùng discovery/ngoài), "
          f"đơn trị anti-PD-(L)1 {int((nsc.DRUG_TYPE == 'PD-1/PDL-1').sum())}, sự kiện OS {int(nsc.E.sum())}")

    g = pd.read_parquet(f"{ds.BASE_DB_DIR}/genomic_data_v3.parquet")
    cols = ["TMB"] + [f"{x} driver_binarized" for x in GENES.values()]
    X = g[cols].astype(float)
    X.columns = ["TMB"] + list(GENES.values())
    coh_d = pd.read_csv(f"{ds.BASE_DB_DIR}/final_cohort_listing.csv").set_index("main_index")
    import lung_helpers as LH
    cl = LH.get_clinical_table_v2(
        path=f"{ds.BASE_DB_DIR}/18193MSKMINDProjectM-OmnibusInventory_DATA_2021-12-20_1540-WITH-TB-and-SCANNER.csv",
        main_index_col="dmp_pt_id", cohort=coh_d[coh_d["cohort"] == "discovery"])
    y = cl["label"].astype(float).reindex(X.index)
    ok = y.notna().values
    X, y = X[ok], y[ok].values
    print("\nso khớp thang đo (trung vị / tỉ lệ có đột biến):")
    print(pd.DataFrame({"discovery": X.median(), "Samstein": nsc[X.columns].median()}).round(2).T.to_string())
    print(pd.DataFrame({"discovery": X.mean(), "Samstein": nsc[X.columns].mean()}).round(2).T.to_string())

    sc = RobustScaler().fit(X.values)
    best = None
    for C in (0.01, 0.1, 1.0, 10.0):
        oof = np.zeros(len(y))
        for tr, te in KFold(10, shuffle=True, random_state=0).split(X):
            m = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(sc.transform(X.values[tr]), y[tr])
            oof[te] = m.decision_function(sc.transform(X.values[te]))
        a = roc_auc_score(y, oof)
        if best is None or a > best[1]:
            best = (C, a)
    lr = LogisticRegression(C=best[0], class_weight="balanced", max_iter=3000).fit(sc.transform(X.values), y)
    print(f"\ndiscovery: LR 7 đặc trưng, C = {best[0]}, AUC CV đáp ứng = {best[1]:.3f}; hệ số:",
          dict(zip(X.columns, np.round(lr.coef_[0], 2))))
    rng = np.random.default_rng(0)
    out = {}
    for name, sub in (("tất cả NSCLC không trùng", nsc), ("chỉ đơn trị anti-PD-(L)1", nsc[nsc.DRUG_TYPE == "PD-1/PDL-1"])):
        s_model = lr.decision_function(sc.transform(sub[X.columns].fillna(sub[X.columns].median()).values))
        s_tmb = -sub["TMB"].fillna(sub["TMB"].median()).values            # TMB cao -> nguy cơ thấp
        T, E = sub["T"].values, sub["E"].values
        n = len(sub)
        R = {"n": int(n), "events": int(E.sum())}
        for lab, s in (("mô hình genomics (7 đặc trưng)", s_model), ("chỉ TMB", s_tmb)):
            ci = [float(np.percentile([concordance_index(T[i], -s[i], E[i]) for i in
                                       (rng.integers(0, n, n) for _ in range(N_BOOT))], q)) for q in (2.5, 97.5)]
            hi = s >= np.median(s)
            d = pd.DataFrame({"z": (s - s.mean()) / s.std(), "T": T, "E": E})
            cph = CoxPHFitter().fit(d, "T", "E")
            R[lab] = {"cindex": float(concordance_index(T, -s, E)), "ci": ci, "hr_per_sd": float(np.exp(cph.params_["z"])),
                      "hr_ci": [float(np.exp(v)) for v in cph.confidence_intervals_.iloc[0]],
                      "logrank_p": float(logrank_test(T[hi], T[~hi], E[hi], E[~hi]).p_value)}
        out[name] = R
        print(f"\n[{name}] n = {R['n']}, sự kiện = {R['events']}")
        for lab in ("mô hình genomics (7 đặc trưng)", "chỉ TMB"):
            r = R[lab]
            print(f"  {lab:<32} C-index {r['cindex']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]  HR/SD {r['hr_per_sd']:.2f} "
                  f"[{r['hr_ci'][0]:.2f}, {r['hr_ci'][1]:.2f}]  log-rank p = {r['logrank_p']:.4f}")
    (_CODE / "experiments" / "results" / "samstein_genomics.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
