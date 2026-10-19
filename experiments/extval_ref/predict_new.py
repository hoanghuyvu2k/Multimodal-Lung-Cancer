"""Dự đoán dữ liệu MỚI bằng các model đã train sẵn (không train lại). Chạy trong ref-linux-py38:

    run_ref.sh experiments/extval_ref/predict_new.py <cache_moi.pkl> [--arm=A3] [--kind=ovo]

<cache_moi.pkl> có cùng cấu trúc khóa `ext` của extval_cache.pkl (glcm, m_glcm, pdl1, m_pdl1, labs6, nlp6, y, index);
tạo bằng make_cache.py chỉnh cho cohort mới. Model: D:/repro_linux/extval_models/{A1,A2,A3,P}__{dyam,ovo}__seed*.pt
(huấn luyện trên toàn bộ discovery n=247). Ghi điểm từng seed + trung bình 5 seed ra <cache_moi>.scores.csv; in AUC nếu có nhãn.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1])); sys.path.insert(0, str(_HERE.parents[2]))
import lung_helpers  # noqa: E402,F401  (cần để unpickle lớp mô hình)
from extval_ref.run_extval_ref import MODEL_DIR, mod_arrays  # noqa: E402


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = dict(a[2:].split("=") for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    new = pickle.loads(Path(args[0]).read_bytes())["ext"]
    cols, ens = {}, {}
    for f in sorted(MODEL_DIR.glob("*.pt")):
        arm, kind, tag = f.stem.split("__")
        if opt.get("arm") and arm != opt["arm"] or opt.get("kind") and kind != opt["kind"]:
            continue
        ck = torch.load(str(f), map_location="cpu", weights_only=False) if "weights_only" in torch.load.__code__.co_varnames else torch.load(str(f), map_location="cpu")
        X, M = mod_arrays(new, ck["mods"])
        cols[f.stem] = np.asarray(ck["model"].predict_proba(X, M), dtype=float).ravel()
        if tag.endswith("_perm"):
            ens.setdefault(f"{arm}__{kind}__ens5", []).append(cols[f.stem])
    for k, v in ens.items():
        cols[k] = np.mean(v, axis=0)
    df = pd.DataFrame(cols, index=new["index"])
    out = Path(args[0]).with_suffix(".scores.csv")
    df.to_csv(out)
    print("điểm ->", out)
    if "y" in new and len(set(new["y"])) == 2:
        from sklearn.metrics import roc_auc_score
        for c in df.columns:
            if c.endswith("seed42_natural") or c.endswith("ens5"):
                print(f"{c:<40} AUC {roc_auc_score(new['y'], df[c]):.3f}")


if __name__ == "__main__":
    main()
