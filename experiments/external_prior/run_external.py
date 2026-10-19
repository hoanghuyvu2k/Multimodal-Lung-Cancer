"""V1 (kiểm chứng ngoài) và tạo điểm tiên nghiệm cho V2 — kế hoạch ở ledger.md.

    PYTHONIOENCODING=utf-8 python experiments/external_prior/run_external.py <thư mục dữ liệu thô cBioPortal>
Kết quả: experiments/results/external_prior.json, experiments/results/external_prior_scores.csv (điểm tiên nghiệm cho 247 bệnh nhân MIND).
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
import torch                                                      # noqa: E402
from sklearn.metrics import roc_auc_score                         # noqa: E402

from archloop import harness as H                                 # noqa: E402,F401  (vá scipy)
from archloop import cands as C                                   # noqa: E402
import baselines.model_uniform_avg as mua                         # noqa: E402
from common import benchmarks as B                                # noqa: E402
from common import data_setup as ds                               # noqa: E402
import lung_helpers as LH                                         # noqa: E402
from external_prior import data as D                              # noqa: E402

INIT_SEEDS = [42, 7, 123, 2024, 31337]
N_BOOT = 5000
GEN = [c for c in D.FEATS if c != "pdl1"]            # tmb + 6 đột biến (7 đặc trưng)


def fit_predict(tr, te, use, seed):
    """`use` ⊂ {"pdl1","gen"}: huấn luyện mô hình dự án trên `tr`, trả điểm cho `te` (cao = KHÔNG lợi ích)."""
    srcs = {"pdl1": ["pdl1"], "gen": GEN}
    Xtr, Xte, Mtr, Mte = [], [], [], []
    for u in use:
        cols = srcs[u]
        Xtr.append(tr[cols].values.astype(float)); Xte.append(te[cols].values.astype(float))
        Mtr.append(tr[cols].notna().all(axis=1).values); Mte.append(te[cols].notna().all(axis=1).values)
    Mtr, Mte = np.column_stack(Mtr).astype(int), np.column_stack(Mte).astype(int)
    for i in range(len(Xtr)):                       # gen: TMB thiếu -> trung vị của tập huấn luyện
        med = np.nanmedian(Xtr[i], axis=0)
        Xtr[i] = np.where(np.isnan(Xtr[i]), med, Xtr[i]); Xte[i] = np.where(np.isnan(Xte[i]), med, Xte[i])
    params = {k: v for k, v in B.MODEL_PARAMS.items() if k != "cross_modality_enabled"}
    out = []
    for s in seed:
        clf = mua.MultiModalDynamicModelUncertainty(**{**params, "init_seed": s})
        clf.fit(Xtr, Mtr, pd.Series(tr["dcb_none"].values, index=range(len(tr))))
        out.append(np.asarray(clf.predict_proba(Xte, Mte), dtype=float))
    return np.mean(out, axis=0)


def boot_auc(y, s, rng):
    v = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            v.append(roc_auc_score(y[i], s[i]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def boot_delta(y, a, b):
    rng = np.random.default_rng(1)
    d = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    d = np.array(d)
    return float(roc_auc_score(y, a) - roc_auc_score(y, b)), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], float(2 * min((d <= 0).mean(), (d >= 0).mean()))


def main():
    torch.set_num_threads(2)
    raw = sys.argv[1]
    coh = pd.read_csv(f"{ds.BASE_DB_DIR}/final_cohort_listing.csv")
    mine = set(coh.main_index.astype(str))
    cl = LH.get_clinical_table_v2(
        path=f"{ds.BASE_DB_DIR}/18193MSKMINDProjectM-OmnibusInventory_DATA_2021-12-20_1540-WITH-TB-and-SCANNER.csv",
        main_index_col="dmp_pt_id", cohort=pd.read_csv(f"{ds.BASE_DB_DIR}/final_cohort_listing.csv").set_index("main_index").query("cohort == 'discovery'"))
    mind = D.mind(raw, cl["bor"], cl["pfs"], C._clinical_pdl1())
    riz, hel = D.rizvi(raw), D.hellmann(raw)
    print(f"MIND n={len(mind)} (không lợi ích {int(mind.dcb_none.sum())}, PD-L1 có {int(mind.pdl1.notna().sum())})")
    out = {"n_mind": len(mind)}
    ext = {}
    for name, df in (("Rizvi 2018", riz), ("Hellmann 2018", hel)):
        n0 = len(df)
        df = df[~df.index.isin(mine)]
        n1 = len(df)
        df = df[df.dcb_none.notna()]
        print(f"{name}: {n0} → loại trùng {n0 - n1} → loại DCB không đánh giá được {n1 - len(df)} → {len(df)} "
              f"(không lợi ích {int(df.dcb_none.sum())}; PD-L1 có {int(df.pdl1.notna().sum())}; TMB có {int(df.tmb.notna().sum())})")
        ext[name] = df
    pooled = pd.concat(ext.values())
    print(f"gộp ngoài n={len(pooled)} | trung vị TMB: MIND {mind.tmb.median():.1f}, Rizvi {ext['Rizvi 2018'].tmb.median():.1f}, Hellmann {ext['Hellmann 2018'].tmb.median():.1f}"
          f" | tỉ lệ PD-L1 ≥ 50%: MIND {(mind.pdl1 >= .5).sum() / mind.pdl1.notna().sum():.2f}, "
          f"Rizvi {(ext['Rizvi 2018'].pdl1 >= .5).sum() / max(1, ext['Rizvi 2018'].pdl1.notna().sum()):.2f}, "
          f"Hellmann {(ext['Hellmann 2018'].pdl1 >= .5).sum() / max(1, ext['Hellmann 2018'].pdl1.notna().sum()):.2f}")
    rng = np.random.default_rng(0)

    # ---------- V1: huấn luyện trên MIND, kiểm tra ngoài ----------
    arms = {"PD-L1": ["pdl1"], "genomics (TMB + 6 đột biến)": ["gen"], "PD-L1 + genomics": ["pdl1", "gen"]}
    V1 = {}
    for scope, te in (("gộp", pooled), ("Rizvi", ext["Rizvi 2018"]), ("Hellmann", ext["Hellmann 2018"])):
        y = te.dcb_none.values
        sc = {k: fit_predict(mind, te, u, INIT_SEEDS) for k, u in arms.items()}
        sc["TMB một mình"] = -te.tmb.fillna(mind.tmb.median()).values
        R = {"n": int(len(y)), "n_none": int(y.sum())}
        for k, s in sc.items():
            R[k] = {"auc": float(roc_auc_score(y, s)), "ci": boot_auc(y, s, rng)}
        for a, b in (("PD-L1 + genomics", "PD-L1"), ("PD-L1 + genomics", "genomics (TMB + 6 đột biến)")):
            R[f"{a} − {b}"] = dict(zip(("delta", "ci", "p"), boot_delta(y, sc[a], sc[b])))
        V1[scope] = R
        print(f"\n[V1 — {scope}] n={R['n']} (không lợi ích {R['n_none']})")
        for k in list(arms) + ["TMB một mình"]:
            print(f"  {k:<30} AUC {R[k]['auc']:.3f} [{R[k]['ci'][0]:.3f}, {R[k]['ci'][1]:.3f}]")
        for k in [x for x in R if " − " in x]:
            print(f"  {k}: {R[k]['delta']:+.3f} [{R[k]['ci'][0]:+.3f}, {R[k]['ci'][1]:+.3f}] p={R[k]['p']:.3f}")
        # ĐÍNH CHÍNH thiết kế đánh giá: cánh "PD-L1" cho điểm HẰNG ở bệnh nhân thiếu PD-L1 nên bị kéo về 0.5 khi tính trên cả tập;
        # báo thêm trên tập con CÓ PD-L1 (mọi cánh cùng bệnh nhân).
        m = te.pdl1.notna().values
        if m.sum() > 20:
            Rs = {"n": int(m.sum()), "n_none": int(y[m].sum())}
            for k, sv in sc.items():
                Rs[k] = {"auc": float(roc_auc_score(y[m], sv[m])), "ci": boot_auc(y[m], sv[m], rng)}
            for a, b in (("PD-L1 + genomics", "PD-L1"), ("PD-L1 + genomics", "genomics (TMB + 6 đột biến)")):
                Rs[f"{a} − {b}"] = dict(zip(("delta", "ci", "p"), boot_delta(y[m], sc[a][m], sc[b][m])))
            V1[scope + " (chỉ bệnh nhân CÓ PD-L1)"] = Rs
            print(f"[V1 — {scope}, chỉ bệnh nhân CÓ PD-L1] n={Rs['n']} (không lợi ích {Rs['n_none']})")
            for k in list(arms) + ["TMB một mình"]:
                print(f"  {k:<30} AUC {Rs[k]['auc']:.3f} [{Rs[k]['ci'][0]:.3f}, {Rs[k]['ci'][1]:.3f}]")
            for k in [x for x in Rs if " − " in x]:
                print(f"  {k}: {Rs[k]['delta']:+.3f} [{Rs[k]['ci'][0]:+.3f}, {Rs[k]['ci'][1]:+.3f}] p={Rs[k]['p']:.3f}")
    out["V1"] = V1

    # ---------- V2: điểm tiên nghiệm = mô hình huấn luyện CHỈ trên dữ liệu ngoài ----------
    prior = fit_predict(pooled, mind, ["pdl1", "gen"], INIT_SEEDS)
    ps = pd.Series(prior, index=mind.index, name="prior")
    ps.to_csv(_CODE / "experiments" / "results" / "external_prior_scores.csv")
    y_m = mind.dcb_none.values
    orig = cl["label"].astype(float).reindex(mind.index).values
    out["V2_prior_on_mind"] = {"auc_dcb": float(roc_auc_score(y_m, prior)), "auc_orig_label": float(roc_auc_score(orig, prior)),
                               "auc_pdl1_only_orig": float(roc_auc_score(orig[mind.pdl1.notna()], -mind.pdl1.dropna()))}
    print(f"\n[V2] điểm tiên nghiệm (huấn luyện trên {len(pooled)} bệnh nhân ngoài) áp lên 247 MIND: AUC nhãn gốc {out['V2_prior_on_mind']['auc_orig_label']:.3f}, "
          f"AUC DCB {out['V2_prior_on_mind']['auc_dcb']:.3f} (PD-L1 một mình, nhãn gốc: {out['V2_prior_on_mind']['auc_pdl1_only_orig']:.3f})")
    (_CODE / "experiments" / "results" / "external_prior.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
