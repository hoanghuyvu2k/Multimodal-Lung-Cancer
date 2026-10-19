"""Bộ nhớ đệm cho kiểm chứng NGOÀI trong ref env (2026-10-14). Chạy ở MÔI TRƯỜNG CHÍNH (có sentence-transformers):

    cd code && PYTHONIOENCODING=utf-8 python experiments/extval_ref/make_cache.py

Ref env (py3.8, torch 1.8.1) không có sentence-transformers nên embedding NLP6 tính ở đây rồi nạp từ pickle.
Kết quả: D:/repro_linux/extval_cache.pkl (discovery 247 + path_valid 71: GLCM, PD-L1 lâm sàng, 6 biến lâm sàng, NLP6, nhãn).
Chuẩn bị dữ liệu GIỐNG `validation/run_external_clinical.py` (khối B) để số liệu so được về mặt cấu trúc.
"""
import pickle
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
_CODE = _EXP.parent
for _p in (str(_CODE), str(_EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402

from archloop import cands as C                                   # noqa: E402
from common.data_setup import load_all, BASE_DB_DIR               # noqa: E402
from validation.run_external_clinical import VARS, embed, external_rows, prompts6  # noqa: E402

OUT = Path("D:/repro_linux/extval_cache.pkl")


def main():
    ctx = load_all()
    ext = external_rows()
    pv = ext[ext.is_path].copy()
    idx = ctx.df_outcomes.index

    lab = ctx.modality_dict["cnl_dem_labs"].reindex(idx)[VARS]
    med = lab.median()
    lab_tr = lab.fillna(med)
    lab_te = pv[VARS].apply(pd.to_numeric, errors="coerce").fillna(med)
    n_miss_te = int(pv[VARS].apply(pd.to_numeric, errors="coerce").isna().sum().sum())

    glcm_tr = ctx.modality_dict["path_ihc_glcm"].reindex(idx)
    m_gl = ctx.modality_MASK["path_ihc_glcm"].reindex(idx).fillna(False).values.astype(bool)
    pdl_tr = C._clinical_pdl1().reindex(idx)

    vp = pd.read_parquet(Path(BASE_DB_DIR) / "lung_pathology_pdl1_glcm_v3_validation.parquet").select_dtypes(include=[np.number])
    vp.index = vp.index.astype(str)
    cols = glcm_tr.columns.intersection(vp.columns)
    glcm_te = vp.reindex(pv["pdl1_image_id"].values)[cols]
    glcm_te.index = pv.index
    m_gl_te = glcm_te.notna().all(axis=1).values

    cache = {
        "vars": VARS,
        "disc": {"index": list(idx), "y": ctx.df_outcomes["label"].astype(float).values,
                 "glcm": glcm_tr[cols].values, "m_glcm": m_gl,
                 "pdl1": pdl_tr.values, "m_pdl1": pdl_tr.notna().values,
                 "labs6": lab_tr.values, "nlp6": embed(prompts6(lab_tr))},
        "ext": {"index": list(pv["pdl1_image_id"].astype(str)), "y": pv["label"].values.astype(float),
                "glcm": glcm_te.values, "m_glcm": m_gl_te,
                "pdl1": pv["pdl1"].values, "m_pdl1": pv["pdl1"].notna().values,
                "labs6": lab_te.values, "nlp6": embed(prompts6(lab_te))},
        "note": f"clinical6 NaN ngoài được điền trung vị discovery ({n_miss_te} ô); GLCM cột chung {len(cols)}",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(pickle.dumps(cache, protocol=4))      # protocol 4: đọc được bởi py3.8
    d, e = cache["disc"], cache["ext"]
    print(f"discovery n={len(d['y'])} (đáp ứng {int((d['y']==0).sum())}); GLCM {int(d['m_glcm'].sum())}; PD-L1 {int(d['m_pdl1'].sum())}; nlp {d['nlp6'].shape}")
    print(f"path_valid n={len(e['y'])} (đáp ứng {int((e['y']==0).sum())}); GLCM {int(e['m_glcm'].sum())}; PD-L1 {int(e['m_pdl1'].sum())}; nlp {e['nlp6'].shape}")
    print(cache["note"])
    print("->", OUT)


if __name__ == "__main__":
    main()
