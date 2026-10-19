"""H1 — mức thổi phồng do chọn mô hình (winner's curse). Kế hoạch ở ledger.md.

SE của hiệu cặp (ensemble 20 seed, bootstrap bệnh nhân) ở BM1 giữa các cặp mô hình đã lưu -> kỳ vọng của MAX trên K ứng viên KHÔNG có hiệu ứng thật,
với tương quan ρ giữa các ứng viên: E[max] = sqrt(1-ρ)·SE·E[max của K chuẩn độc lập]. So với mức tăng quan sát (mô hình chốt so với uniform_avg và DyAM).

    PYTHONIOENCODING=utf-8 python experiments/archloop/analyze_selection_bias.py
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
_EXP = Path(__file__).resolve().parents[1]
_CODE = _EXP.parent
OUT = _CODE / "experiments" / "results"
SL = "Rad_IHCG_Gen_PDL1"

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from sklearn.metrics import roc_auc_score                         # noqa: E402

K = 52


def ens(name):
    d = pd.read_csv(OUT / "ovo_proof" / f"{name}__{SL}.csv", index_col=0)
    cols = [c for c in d.columns if c.startswith("seed_")]
    return d["label"].values, ((d[cols] - d[cols].mean()) / d[cols].std()).mean(axis=1).values


def main():
    rng = np.random.default_rng(0)
    pairs = [("c18stack", "uniform"), ("uniform", "dyam"), ("c18stack", "dyam"), ("loso_nlp", "c18stack"),
             ("loso_miss", "c18stack"), ("f1availstack", "c18stack"), ("f1cvstack", "c18stack"),
             ("f1cvstack", "f1availstack"), ("loso_gen", "c18stack")]
    res = {"pairs": {}}
    print("SE của hiệu AUC ensemble (BM1, 20 seed, bootstrap bệnh nhân 3000):")
    for a, b in pairs:
        y, sa = ens(a)
        _, sb = ens(b)
        ok = ~(np.isnan(sa) | np.isnan(sb))
        y, sa, sb = y[ok], sa[ok], sb[ok]
        d = []
        for _ in range(3000):
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(y[i])) == 2:
                d.append(roc_auc_score(y[i], sa[i]) - roc_auc_score(y[i], sb[i]))
        obs = roc_auc_score(y, sa) - roc_auc_score(y, sb)
        res["pairs"][f"{a} - {b}"] = {"delta": float(obs), "se": float(np.std(d))}
        print(f"  {a:<14} − {b:<14} Δ = {obs:+.4f}  SE = {np.std(d):.4f}")
    ses = np.array([v["se"] for k, v in res["pairs"].items() if "loso" in k or "f1" in k])
    # SE đại diện: các cặp mô hình GẦN NHAU (thay đổi nhỏ), không phải cặp c18stack-vs-dyam khác hẳn
    se_ref = float(np.median(ses))
    print(f"\nSE đại diện cho một thay đổi nhỏ (trung vị các cặp loso/f1): {se_ref:.4f}")
    emax = {}
    z = rng.standard_normal((200000, K))
    base = z.max(axis=1)
    emax["iid"] = float(base.mean())
    print(f"E[max của {K} chuẩn độc lập] = {emax['iid']:.2f}")
    obs_u = res["pairs"]["c18stack - uniform"]["delta"]
    obs_d = res["pairs"]["c18stack - dyam"]["delta"]
    print(f"\nMức tăng quan sát: mô hình chốt − uniform = {obs_u:+.4f}; − DyAM = {obs_d:+.4f}")
    print(f"{'ρ giữa ứng viên':<18}{'E[max] khi KHÔNG có hiệu ứng thật':>40}{'P(max ≥ quan sát vs uniform)':>32}")
    res["null_max"] = {}
    for rho in (0.0, 0.5, 0.8):
        zz = np.sqrt(rho) * rng.standard_normal((200000, 1)) + np.sqrt(1 - rho) * z
        m = zz.max(axis=1) * se_ref
        res["null_max"][str(rho)] = {"expected_max": float(m.mean()), "p_ge_obs_uniform": float((m >= obs_u).mean())}
        print(f"{rho:<18}{m.mean():>40.4f}{(m >= obs_u).mean():>32.3f}")
    res.update({"K": K, "se_ref": se_ref, "obs_vs_uniform": obs_u, "obs_vs_dyam": obs_d})
    (OUT / "selection_bias.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
