"""H1 (bản ref) — mức thổi phồng do chọn mô hình, từ các cặp mô hình đã chạy trong ref-linux-py38 (không huấn luyện).

Giống analyze_selection_bias.py nhưng SE của hiệu AUC ensemble (BM1, 20 seed, bootstrap bệnh nhân 3000) lấy từ các cặp GẦN NHAU trong ref:
bỏ từng nguồn (5 cặp loso_ovo_* so với OvO+NLP đầy đủ) và OvO+NLP có/không vector sẵn-có (nlp20miss_ovo vs nlp20_ovo) ở BM1.
K = 52 ứng viên (số ứng viên đã ghi ở §2.40; không đổi theo môi trường).

    python experiments/archloop/analyze_selection_bias_ref.py     (phân tích trên điểm đã lưu, chạy được ở Windows)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parents[1] / "results" / "ref"
SL = "Rad_IHCG_Gen_PDL1"
K = 52


def ens(path):
    d = pd.read_csv(path, index_col=0)
    cols = [c for c in d.columns if c.startswith("seed_")]
    return d["label"].values, ((d[cols] - d[cols].mean()) / d[cols].std()).mean(axis=1).values


def main():
    rng = np.random.default_rng(0)
    P = OUT / "ovo_proof"
    full = P / f"nlp20_ovo__{SL}.csv"
    pairs = [(f"loso_ovo_{a}", P / f"loso_ovo_{a}.csv", "nlp20_ovo", full) for a in ("rad", "path", "gen", "pdl1", "nlp")]
    pairs += [("nlp20miss_ovo", P / f"nlp20miss_ovo__{SL}.csv", "nlp20_ovo", full),
              ("nlp20_ovo", full, "nlp20_dyam", P / f"nlp20_dyam__{SL}.csv")]
    res = {"pairs": {}}
    for na, fa, nb, fb in pairs:
        y, sa = ens(fa)
        _, sb = ens(fb)
        ok = ~(np.isnan(sa) | np.isnan(sb))
        y, sa, sb = y[ok], sa[ok], sb[ok]
        d = []
        for _ in range(3000):
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(y[i])) == 2:
                d.append(roc_auc_score(y[i], sa[i]) - roc_auc_score(y[i], sb[i]))
        obs = roc_auc_score(y, sa) - roc_auc_score(y, sb)
        res["pairs"][f"{na} - {nb}"] = {"delta": float(obs), "se": float(np.std(d))}
        print(f"  {na:<16} - {nb:<12} delta {obs:+.4f}  SE {np.std(d):.4f}")
    close = [v["se"] for k, v in res["pairs"].items() if k.startswith(("loso", "nlp20miss"))]
    se_ref = float(np.median(close))
    print(f"\nSE dai dien (trung vi {len(close)} cap gan nhau): {se_ref:.4f}")
    z = rng.standard_normal((200000, K))
    print(f"E[max cua {K} chuan doc lap] = {z.max(axis=1).mean():.2f}")
    res["null_max"] = {}
    for rho in (0.0, 0.5, 0.8):
        zz = np.sqrt(rho) * rng.standard_normal((200000, 1)) + np.sqrt(1 - rho) * z
        m = (zz.max(axis=1) * se_ref).mean()
        res["null_max"][str(rho)] = float(m)
        print(f"rho {rho}: E[max] khi khong co hieu ung that = {m:.4f}")
    res.update({"K": K, "se_ref": se_ref})
    (OUT / "selection_bias.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
