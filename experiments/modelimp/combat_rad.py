"""Hướng 1: hài hoà hoá radiomics theo máy chụp bằng ComBat.

Đặc trưng radiomics phụ thuộc mạnh vào máy chụp/tham số tái tạo. Cohort có hai cột máy trong
omnibus CSV: `scanner_lightspeed` và `scanner_revolution` → batch ∈ {lightspeed, revolution,
unknown}. ComBat là hiệu chỉnh KHÔNG GIÁM SÁT (không dùng nhãn) nên không rò rỉ nhãn.

Dùng: `ctx = load_harmonized()` thay cho `load_all()`.
"""
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
_CODE = _EXP.parent
for _p in (str(_CODE), str(_EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402

from common import data_setup as ds                               # noqa: E402

OMNIBUS = ("18193MSKMINDProjectM-OmnibusInventory_DATA_2021-12-20_1540"
           "-WITH-TB-and-SCANNER.csv")


def scanner_batch():
    """Trả Series: main_index -> nhãn máy chụp."""
    df = pd.read_csv(f"{ds.BASE_DB_DIR}/{OMNIBUS}")
    key = "dmp_pt_id" if "dmp_pt_id" in df.columns else df.columns[0]
    df = df.set_index(key)
    lab = pd.Series("unknown", index=df.index, dtype=object)
    if "scanner_lightspeed" in df.columns:
        lab[df["scanner_lightspeed"].fillna(False).astype(bool)] = "lightspeed"
    if "scanner_revolution" in df.columns:
        lab[df["scanner_revolution"].fillna(False).astype(bool)] = "revolution"
    return lab[~lab.index.duplicated()]


def combat_frame(df, batch):
    """ComBat trên các cột số của df (index = bệnh nhân, có thể lặp theo tổn thương)."""
    from neuroCombat import neuroCombat

    b = batch.reindex(df.index).fillna("unknown").values
    keep_rows = pd.Series(b).value_counts()
    valid = set(keep_rows[keep_rows >= 10].index)          # batch quá nhỏ thì gộp vào unknown
    b = np.array([x if x in valid else "unknown" for x in b])
    if len(set(b)) < 2:
        return df, 0

    META = {"job_tag", "lesion_index", "site", "main_index"}   # KHÔNG hài hoà cột siêu dữ liệu
    num = [c for c in df.select_dtypes(include=[np.number]).columns if c not in META]
    ok = []
    for c in num:                       # bỏ cột hằng số toàn cục hoặc hằng số trong một batch
        v = df[c].astype(float)
        if not np.isfinite(v).any() or v.std(skipna=True) in (0, np.nan):
            continue
        if any(v[b == g].std(skipna=True) in (0, np.nan) for g in set(b)):
            continue
        ok.append(c)
    if not ok:
        return df, 0

    dat = df[ok].astype(float).fillna(df[ok].astype(float).mean()).T.values   # features x samples
    covars = pd.DataFrame({"batch": b})
    out = neuroCombat(dat=dat, covars=covars, batch_col="batch")["data"]
    res = df.copy()
    res[ok] = out.T
    return res, len(ok)


def load_harmonized(verbose=True):
    """load_all() nhưng radiomics đã qua ComBat theo máy chụp."""
    batch = scanner_batch()
    orig = ds._read_parquet_any
    info = {}

    def patched(*names):
        df = orig(*names)
        if df.empty or not any("radiomic" in n.lower() for n in names):
            return df
        out, n_feat = combat_frame(df, batch)
        info["n_feat"] = n_feat
        info["n_rows"] = len(df)
        return out

    # QUAN TRỌNG: load_all(use_cache=False) GHI ĐÈ cache dùng chung. Phải đổi hướng cache
    # sang file riêng, nếu không dữ liệu đã hài hoà sẽ ngấm vào mọi thí nghiệm chạy sau.
    orig_cache = ds._CACHE_PATH
    ds._read_parquet_any = patched
    ds._CACHE_PATH = orig_cache.with_name("combat_" + orig_cache.name)
    try:
        ctx = ds.load_all(use_cache=ds._CACHE_PATH.exists())
    finally:
        ds._read_parquet_any = orig
        ds._CACHE_PATH = orig_cache
    if verbose:
        print(f"   [ComBat] đã hài hoà {info.get('n_feat', 0)} đặc trưng radiomics "
              f"trên {info.get('n_rows', 0)} dòng; batch: "
              f"{dict(pd.Series(batch.values).value_counts())}")
    return ctx
