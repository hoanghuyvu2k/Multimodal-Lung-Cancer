"""R1 + R2 — khoảng tin cậy cấp bệnh nhân của con số chốt và hai điểm cuối phụ (PFS, DCB), trên điểm đã lưu.
Quy tắc định trước ở ledger.md: C-index = concordance_index(T, -điểm, E) với E = pfs_censor (1 = sự kiện); HR = Cox liên tục theo 1 SD của điểm
ensemble + log-rank theo trung vị; DCB: bor == 3 & pfs >= 6; SD bị kiểm duyệt trước 6 tháng bị LOẠI khỏi điểm cuối DCB. PFS KHÔNG độc lập với nhãn.

    PYTHONIOENCODING=utf-8 python experiments/archloop/analyze_endpoints.py
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
from sklearn.metrics import roc_auc_score                         # noqa: E402

from common import data_setup as ds                               # noqa: E402
import lung_helpers as LH                                         # noqa: E402

OUT = _CODE / "experiments" / "results"
COMBOS = ["Rad+IHC-G+Gen+PDL1", "Rad+IHC-A+Gen+TMB+PDL1", "Rad+IHC-A+Gen+PDL1", "Rad+IHC-A+MutAmp"]
MODELS = [("DyAM", "dyam"), ("uniform", "uniform"), ("mô hình chốt", "c18stack")]
N_BOOT = 5000


def slug(l):
    return l.replace("+", "_").replace("-", "").replace(" ", "")


def ensemble(df):
    cols = [c for c in df.columns if c.startswith("seed_")]
    z = (df[cols] - df[cols].mean()) / df[cols].std()
    return z.mean(axis=1), cols


def auc_na(y, s):
    m = ~np.isnan(s)
    return roc_auc_score(y[m], s[m])


def boot(fn, n, rng, k=N_BOOT):
    v = []
    for _ in range(k):
        i = rng.integers(0, n, n)
        try:
            v.append(fn(i))
        except ValueError:
            pass
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    coh = pd.read_csv(f"{ds.BASE_DB_DIR}/final_cohort_listing.csv").set_index("main_index")
    c = LH.get_clinical_table_v2(
        path=f"{ds.BASE_DB_DIR}/18193MSKMINDProjectM-OmnibusInventory_DATA_2021-12-20_1540-WITH-TB-and-SCANNER.csv",
        main_index_col="dmp_pt_id", cohort=coh[coh["cohort"] == "discovery"])
    rng = np.random.default_rng(0)
    res = {}
    # điểm cuối DCB: dương = KHÔNG lợi ích; loại SD bị kiểm duyệt trước 6 tháng
    sd = c["bor"] == 3
    unknown = sd & (c["pfs"] < 6) & (c["pfs_censor"] == 0)
    dcb_sd = sd & (c["pfs"] >= 6)
    y_dcb = ((c["label"] == 1) & ~dcb_sd).astype(float)
    keep_dcb = ~unknown
    print(f"DCB: SD {int(sd.sum())}, có lợi ích bền vững {int(dcb_sd.sum())}, SD < 6 tháng bị kiểm duyệt (loại) {int(unknown.sum())}; "
          f"điểm cuối DCB n = {int(keep_dcb.sum())}, dương (không lợi ích) {int(y_dcb[keep_dcb].sum())}")

    for label in COMBOS:
        res[label] = {}
        for name, key in MODELS:
            df = pd.read_csv(OUT / "ovo_proof" / f"{key}__{slug(label)}.csv", index_col=0)
            s, cols = ensemble(df)
            s = s.reindex(c.index)
            y = c["label"].astype(float).values
            sv = s.values
            per = [auc_na(y, ((df[x] - df[x].mean()) / df[x].std()).reindex(c.index).values) for x in cols]
            R = {"auc_seed_mean": float(np.mean(per)), "auc_seed_sd": float(np.std(per, ddof=1)),
                 "auc_ens": float(auc_na(y, sv)),
                 "auc_ens_ci": boot(lambda i: roc_auc_score(y[i], sv[i]), len(y), rng)}
            T, E = c["pfs"].values, c["pfs_censor"].values
            R["cindex_pfs"] = float(concordance_index(T, -sv, E))
            R["cindex_pfs_ci"] = boot(lambda i: concordance_index(T[i], -sv[i], E[i]), len(y), rng, 1000)
            d = pd.DataFrame({"z": (sv - sv.mean()) / sv.std(), "T": T, "E": E})
            cph = CoxPHFitter().fit(d, "T", "E")
            R["hr_per_sd"] = float(np.exp(cph.params_["z"]))
            R["hr_per_sd_ci"] = [float(np.exp(cph.confidence_intervals_.iloc[0, 0])), float(np.exp(cph.confidence_intervals_.iloc[0, 1]))]
            hi = sv >= np.median(sv)
            R["logrank_p"] = float(logrank_test(T[hi], T[~hi], E[hi], E[~hi]).p_value)
            R["median_pfs_high_vs_low"] = [float(np.median(T[hi])), float(np.median(T[~hi]))]
            k = keep_dcb.values
            R["auc_dcb"] = float(roc_auc_score(y_dcb.values[k], sv[k]))
            R["auc_dcb_ci"] = boot(lambda i: roc_auc_score(y_dcb.values[k][i], sv[k][i]), int(k.sum()), rng, 2000)
            res[label][name] = R
        print(f"\n=== {label} ===")
        print(f"{'mô hình':<14}{'AUC/seed':>14}{'AUC ens [CI 95% bệnh nhân]':>34}{'C-index PFS [CI]':>26}{'HR/SD [CI]':>20}{'log-rank p':>11}{'AUC DCB [CI]':>24}")
        for name, _ in MODELS:
            R = res[label][name]
            print(f"{name:<14}{R['auc_seed_mean']:>8.3f}±{R['auc_seed_sd']:.3f}"
                  f"{R['auc_ens']:>14.3f} [{R['auc_ens_ci'][0]:.3f}, {R['auc_ens_ci'][1]:.3f}]"
                  f"{R['cindex_pfs']:>12.3f} [{R['cindex_pfs_ci'][0]:.3f}, {R['cindex_pfs_ci'][1]:.3f}]"
                  f"{R['hr_per_sd']:>9.2f} [{R['hr_per_sd_ci'][0]:.2f}, {R['hr_per_sd_ci'][1]:.2f}]"
                  f"{R['logrank_p']:>11.4f}"
                  f"{R['auc_dcb']:>9.3f} [{R['auc_dcb_ci'][0]:.3f}, {R['auc_dcb_ci'][1]:.3f}]")
    (OUT / "endpoints.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
