"""OvO có thực sự tốt hơn ở các tổ hợp nhiều nguồn? — phần KIỂM ĐỊNH (quyết định trước khi xem số).

Năm phép kiểm cho mỗi cặp mô hình (A − B):
  1. Wilcoxon + t ghép cặp trên AUC của 20 seed   -> ổn định qua phân hoạch/khởi tạo
     (LƯU Ý: seed không phải mẫu bệnh nhân độc lập; phép này KHÔNG nói gì về bệnh nhân mới)
  2. Bootstrap ghép cặp trên bệnh nhân (5000)      -> bất định do mẫu bệnh nhân
  3. DeLong cho hai AUC tương quan                 -> phép kiểm chuẩn trong ngành
  4. Hoán vị tráo điểm giữa hai mô hình (5000)     -> phi tham số
  5. Bootstrap gộp: ΔAUC trung bình trên cả 6 tổ hợp, lấy mẫu lại bệnh nhân (5000)
     -> MỘT kết luận tổng, tính đúng việc 6 tổ hợp dùng chung bệnh nhân

Phép 2-5 dùng điểm ensemble (trung bình z-score qua các seed) của mỗi mô hình.

    PYTHONIOENCODING=utf-8 python experiments/ovo_proof/analyze.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

_CODE = Path(__file__).resolve().parents[2]
OUT = _CODE / "experiments" / "results" / "ovo_proof"
COMBOS = ["Rad+IHC-A+Gen", "Rad+IHC-G+Gen", "Rad+IHC-A+MutAmp", "Rad+IHC-A+Gen+PDL1",
          "Rad+IHC-G+Gen+PDL1", "Rad+IHC-A+Gen+TMB+PDL1"]
N_BOOT = 5000


def slug(label):
    return label.replace("+", "_").replace("-", "").replace(" ", "")


def load(model, label):
    f = OUT / f"{model}__{slug(label)}.csv"
    return pd.read_csv(f, index_col=0) if f.exists() else None


def seeds_of(df):
    return [c for c in df.columns if c.startswith("seed_")]


def ensemble(df, seeds):
    z = df[seeds].apply(lambda c: (c - c.mean()) / c.std())
    return z.mean(axis=1, skipna=True)


def _midrank(x):
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def delong_paired(y, s1, s2):
    """Kiểm định DeLong cho hai AUC tương quan (Sun & Xu 2014). Trả (auc1, auc2, z, p)."""
    y = np.asarray(y).astype(int)
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    aucs, v01, v10 = [], [], []
    for s in (np.asarray(s1, float), np.asarray(s2, float)):
        tx, ty, tz = _midrank(s[pos]), _midrank(s[neg]), _midrank(np.r_[s[pos], s[neg]])
        aucs.append(tz[:m].sum() / m / n - (m + 1.0) / 2.0 / n)
        v01.append((tz[:m] - tx) / n)
        v10.append(1.0 - (tz[m:] - ty) / m)
    cov = np.cov(np.array(v01)) / m + np.cov(np.array(v10)) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    zstat = (aucs[0] - aucs[1]) / np.sqrt(max(var, 1e-18))
    return aucs[0], aucs[1], float(zstat), float(2 * stats.norm.sf(abs(zstat)))


def two_sided(deltas):
    return float(min(1.0, 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())))


def compare(a_name, b_name, rng):
    rows, ens_pairs, seed_delta = [], {}, {}
    for label in COMBOS:
        A, B = load(a_name, label), load(b_name, label)
        if A is None or B is None:
            continue
        seeds = [s for s in seeds_of(A) if s in seeds_of(B)]
        if len(seeds) < 5:
            continue
        idx = A.index.intersection(B.index)
        y = A.loc[idx, "label"]

        # 1. theo seed
        da = []
        for s in seeds:
            ok = A.loc[idx, s].notna() & B.loc[idx, s].notna()
            da.append(roc_auc_score(y[ok], A.loc[idx, s][ok]) - roc_auc_score(y[ok], B.loc[idx, s][ok]))
        da = np.array(da)
        seed_delta[label] = pd.Series(da, index=seeds)
        w_p = stats.wilcoxon(da).pvalue if np.any(da != 0) else 1.0
        t_p = stats.ttest_1samp(da, 0.0).pvalue

        # 2-4. theo bệnh nhân, trên điểm ensemble
        ea, eb = ensemble(A.loc[idx], seeds), ensemble(B.loc[idx], seeds)
        ok = ea.notna() & eb.notna()
        yy, sa, sb = y[ok].values, ea[ok].values, eb[ok].values
        ens_pairs[label] = (pd.Series(yy, index=y[ok].index), pd.Series(sa, index=y[ok].index),
                            pd.Series(sb, index=y[ok].index))
        auc_a, auc_b = roc_auc_score(yy, sa), roc_auc_score(yy, sb)
        n = len(yy)
        boot = []
        for _ in range(N_BOOT):
            p = rng.integers(0, n, n)
            if len(np.unique(yy[p])) < 2:
                continue
            boot.append(roc_auc_score(yy[p], sa[p]) - roc_auc_score(yy[p], sb[p]))
        boot = np.array(boot)
        _, _, zstat, dl_p = delong_paired(yy, sa, sb)
        obs = auc_a - auc_b
        perm = []
        for _ in range(N_BOOT):
            sw = rng.random(n) < 0.5
            perm.append(roc_auc_score(yy, np.where(sw, sb, sa)) - roc_auc_score(yy, np.where(sw, sa, sb)))
        perm_p = float((np.abs(perm) >= abs(obs) - 1e-12).mean())

        rows.append({
            "combo": label, "n_seeds": len(seeds), "n_patients": int(n),
            "seed_mean_delta": float(da.mean()), "seed_sd_delta": float(da.std(ddof=1)),
            "seed_wins": int((da > 0).sum()), "wilcoxon_p": float(w_p), "ttest_p": float(t_p),
            "ens_auc_a": float(auc_a), "ens_auc_b": float(auc_b), "ens_delta": float(obs),
            "boot_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "boot_p": two_sided(boot), "delong_z": zstat, "delong_p": dl_p, "perm_p": perm_p,
        })

    # 5. gộp trên các tổ hợp: lấy mẫu lại BỆNH NHÂN một lần, áp cho mọi tổ hợp
    omni = None
    if ens_pairs:
        allp = sorted(set().union(*[set(v[0].index) for v in ens_pairs.values()]))
        allp = np.array(allp)

        def mean_delta(sample):
            cnt = pd.Series(sample).value_counts()
            ds = []
            for yv, sa, sb in ens_pairs.values():
                keep = cnt.index.intersection(yv.index)
                w = cnt.loc[keep].values
                yy = np.repeat(yv.loc[keep].values, w)
                if len(np.unique(yy)) < 2:
                    return None
                ds.append(roc_auc_score(yy, np.repeat(sa.loc[keep].values, w))
                          - roc_auc_score(yy, np.repeat(sb.loc[keep].values, w)))
            return float(np.mean(ds))

        obs = float(np.mean([r["ens_delta"] for r in rows]))
        bs = [mean_delta(rng.choice(allp, len(allp), replace=True)) for _ in range(N_BOOT)]
        bs = np.array([b for b in bs if b is not None])
        # ổn định qua seed, đã lấy trung bình trên các tổ hợp
        sd = pd.DataFrame(seed_delta).mean(axis=1)
        omni = {"mean_delta": obs,
                "boot_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "boot_p": two_sided(bs), "n_combos": len(rows),
                "combos_a_wins_ens": int(sum(r["ens_delta"] > 0 for r in rows)),
                "seed_avg_mean": float(sd.mean()), "seed_avg_wins": int((sd > 0).sum()),
                "seed_avg_n": int(len(sd)),
                "seed_avg_wilcoxon_p": float(stats.wilcoxon(sd).pvalue) if np.any(sd != 0) else 1.0}
    return rows, omni


def show(a, b, rows, omni):
    print(f"\n{'=' * 118}\n{a.upper()} − {b.upper()}\n{'=' * 118}")
    print(f"{'tổ hợp':<24}{'Δ/seed':>9}{'thắng':>7}{'Wilcoxon':>10}{'AUC ens A':>11}{'AUC ens B':>11}"
          f"{'Δ ens':>9}{'CI 95% bootstrap':>21}{'p boot':>8}{'p DeLong':>10}{'p hoán vị':>11}")
    for r in rows:
        print(f"{r['combo']:<24}{r['seed_mean_delta']:>+9.4f}{r['seed_wins']:>4}/{r['n_seeds']:<2}"
              f"{r['wilcoxon_p']:>10.4f}{r['ens_auc_a']:>11.4f}{r['ens_auc_b']:>11.4f}"
              f"{r['ens_delta']:>+9.4f}   [{r['boot_ci'][0]:+.4f},{r['boot_ci'][1]:+.4f}]"
              f"{r['boot_p']:>8.3f}{r['delong_p']:>10.3f}{r['perm_p']:>11.3f}")
    if omni:
        print(f"\nGỘP {omni['n_combos']} tổ hợp: Δ ensemble TB = {omni['mean_delta']:+.4f}, "
              f"CI 95% [{omni['boot_ci'][0]:+.4f}, {omni['boot_ci'][1]:+.4f}], "
              f"p bootstrap = {omni['boot_p']:.3f}; A thắng {omni['combos_a_wins_ens']}/{omni['n_combos']} tổ hợp")
        print(f"Theo seed (TB trên các tổ hợp): Δ = {omni['seed_avg_mean']:+.4f}, A thắng "
              f"{omni['seed_avg_wins']}/{omni['seed_avg_n']} seed, Wilcoxon p = {omni['seed_avg_wilcoxon_p']:.5f}")


def main():
    rng = np.random.default_rng(0)
    out = {}
    for a, b in (("ovo", "dyam"), ("ovo", "uniform"), ("uniform", "dyam")):
        rows, omni = compare(a, b, rng)
        if rows:
            show(a, b, rows, omni)
            out[f"{a}_vs_{b}"] = {"per_combo": rows, "omnibus": omni}
    (OUT / "tests.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
