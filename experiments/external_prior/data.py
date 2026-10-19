"""Dựng bảng đặc trưng thống nhất (PD-L1, TMB, 6 đột biến) + nhãn DCB cho MIND (discovery) và hai cohort ngoài công khai.
Dữ liệu thô tải trước từ cBioPortal API vào thư mục TMP (xem ledger.md): {study}_{PATIENT,SAMPLE}.json, {study}_mut6.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

GENES = {1956: "EGFR", 2064: "ERBB2", 673: "BRAF", 4233: "MET", 6794: "STK11", 8289: "ARID1A"}
FEATS = ["pdl1", "tmb"] + [f"mut_{g}" for g in GENES.values()]


def _clin(D, sid, typ):
    d = json.load(open(Path(D) / f"{sid}_{typ}.json"))
    return pd.DataFrame(d).pivot_table(index="patientId", columns="clinicalAttributeId", values="value", aggfunc="first")


def _muts(D, sid, ids):
    out = pd.DataFrame(0.0, index=ids, columns=[f"mut_{g}" for g in GENES.values()])
    for m in json.load(open(Path(D) / f"{sid}_mut6.json")):
        if m["patientId"] in out.index:
            out.loc[m["patientId"], f"mut_{GENES[m['entrezGeneId']]}"] = 1.0
    return out


def num(s):
    return pd.to_numeric(s, errors="coerce")


def rizvi(D):
    p, s = _clin(D, "nsclc_pd1_msk_2018", "PATIENT"), _clin(D, "nsclc_pd1_msk_2018", "SAMPLE")
    df = pd.DataFrame(index=p.index)
    df["dcb_none"] = p["DURABLE_CLINICAL_BENEFIT"].map({"YES": 0.0, "NO": 1.0})       # NE -> NaN (loại)
    df["pdl1"] = num(s["PDL1_SCORE"]).reindex(df.index) / 100.0
    df["tmb"] = num(s["TMB_NONSYNONYMOUS"]).reindex(df.index)
    return df.join(_muts(D, "nsclc_pd1_msk_2018", df.index))


def hellmann(D):
    p, s = _clin(D, "nsclc_mskcc_2018", "PATIENT"), _clin(D, "nsclc_mskcc_2018", "SAMPLE")
    df = pd.DataFrame(index=p.index)
    df["dcb_none"] = p["DURABLE_CLINICAL_BENEFIT"].map({"Durable Clinical Benefit": 0.0, "No Durable Benefit": 1.0})
    df["pdl1"] = num(p["PDL1_EXP"]).reindex(df.index) / 100.0
    df["tmb"] = num(s["TMB_NONSYNONYMOUS"]).reindex(df.index)
    return df.join(_muts(D, "nsclc_mskcc_2018", df.index))


def mind(D, bor, pfs, clinical_pdl1):
    """MIND: DCB = PR/CR hoặc SD có PFS >= 6 tháng. `bor`, `pfs` là Series theo patient; clinical_pdl1 = -TPS/100 (kiểu C18)."""
    p, s = _clin(D, "lung_msk_mind_2020", "PATIENT"), _clin(D, "lung_msk_mind_2020", "SAMPLE")
    idx = bor.index
    df = pd.DataFrame(index=idx)
    benefit = bor.isin([1, 2]) | ((bor == 3) & (pfs >= 6))
    df["dcb_none"] = (~benefit).astype(float)
    df["pdl1"] = (-clinical_pdl1.reindex(idx))
    df["tmb"] = num(s["TMB_NONSYNONYMOUS"]).reindex(idx)
    return df.join(_muts(D, "lung_msk_mind_2020", idx))


EXTRA = {3845: "KRAS", 7157: "TP53", 9817: "KEAP1", 6597: "SMARCA4"}


def add_extras(D, sid, df, smoke, sex, age):
    """Thêm 4 đột biến (KRAS, TP53, KEAP1, SMARCA4) + hút thuốc từng/chưa, giới nữ, tuổi. `smoke`, `sex`, `age`: Series theo patientId."""
    out = df.copy()
    for g in EXTRA.values():
        out[f"mut_{g}"] = 0.0
    fn = {"lung_msk_mind_2020": "mind4"}.get(sid, f"{sid}_mut4")
    for m in json.load(open(Path(D) / f"{fn}.json")):
        if m["patientId"] in out.index:
            out.loc[m["patientId"], f"mut_{EXTRA[m['entrezGeneId']]}"] = 1.0
    out["smoke_ever"] = smoke.reindex(out.index).astype(float)
    out["sex_female"] = sex.reindex(out.index).astype(float)
    out["age"] = num(age.reindex(out.index))
    return out


def extras_rizvi(D, df):
    p = _clin(D, "nsclc_pd1_msk_2018", "PATIENT")
    return add_extras(D, "nsclc_pd1_msk_2018", df, p["SMOKER"].map({"Ever": 1.0, "Never": 0.0}), p["SEX"].eq("Female"), p["AGE"])


def extras_hellmann(D, df):
    p = _clin(D, "nsclc_mskcc_2018", "PATIENT")
    return add_extras(D, "nsclc_mskcc_2018", df, p["SMOKING_STATUS"].map({"Former": 1.0, "Current": 1.0, "Never": 0.0}), p["SEX"].eq("Female"), p["AGE_YRS"])


def extras_mind(D, df):
    p = _clin(D, "lung_msk_mind_2020", "PATIENT")
    smoke = p["SMOKING_STATUS"].map(lambda v: 0.0 if str(v).startswith("Never") else (1.0 if isinstance(v, str) else np.nan))
    return add_extras(D, "lung_msk_mind_2020", df, smoke, p["SEX"].eq("Female"), p["AGE"])


def hellmann_bor(D):
    """Nhãn đáp ứng gốc cho Hellmann: CR/PR -> 0, SD/PD -> 1, NE -> NaN."""
    p = _clin(D, "nsclc_mskcc_2018", "PATIENT")
    return p["BEST_OVERALL_RESPONSE"].map({"CR": 0.0, "PR": 0.0, "SD": 1.0, "PD": 1.0})
