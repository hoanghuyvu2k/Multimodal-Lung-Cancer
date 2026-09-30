"""SK-3/4/5: chạy lại Bảng 5, 6, 7, 8 bằng đánh giá 5-seed thay cho đơn-phân-hoạch.

Bảng 5–8 của bài báo dùng MỘT lần chia 10-fold. Kiểm chứng 2026-09-30 cho thấy chúng không
tái lập được và hiệu ứng NLP đảo chiều — đúng như §3.2 của chính bài báo đã cảnh báo về tính
không ổn định của số đơn-phân-hoạch. Script này đánh giá đúng 11 biến thể đó bằng harness
5-seed chuẩn (§0 của result/training-results.md), cho cả 4 nhóm chỉ số:

    AUC (Bảng 5, 6) · C-index + Cox HR (Bảng 7) · KM log-rank χ² (Bảng 8)

Mỗi chỉ số báo mean ± sd trên 5 seed, kèm paired bootstrap so với biến thể "No clinical".

Chạy từ code/:
    PYTHONIOENCODING=utf-8 python experiments/nlp_verify/verify_nlp_5seed.py
Kết quả:
    experiments/results/verify_nlp_5seed.json
"""
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
_ROOT = _EXP.parent
for _p in (str(_ROOT), str(_EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Vá scipy >= 1.14 (BracketError trên cột radiomics suy biến) — xem experiments/verify_all.py
import sklearn.preprocessing._data as _skd                        # noqa: E402

_orig_yj = _skd.PowerTransformer._yeo_johnson_optimize


def _safe_yj(self, x):
    try:
        return _orig_yj(self, x)
    except Exception:
        return 1.0


_skd.PowerTransformer._yeo_johnson_optimize = _safe_yj

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from lifelines.statistics import logrank_test                     # noqa: E402
from sklearn.metrics import roc_auc_score                         # noqa: E402

from common import benchmarks as B                                # noqa: E402
from common.data_setup import load_all, BASE_DB_DIR               # noqa: E402
from common.evaluate import paired_bootstrap, save_result         # noqa: E402
from nlp_verify.nlp_modality import add_nlp_modality              # noqa: E402
from nlp_verify.verify_nlp_variants import (                      # noqa: E402
    ARMS, BASE, FOLDS, MODEL_PARAMS, VARIANTS, add_bioclinbert, add_pca16, slug,
)
from lung_helpers import (                                        # noqa: E402
    compute_cindex, compute_multivariate_cox, get_clinical_table_v2,
    get_training_data, train,
)

# Số đơn-phân-hoạch đang in trong bài, để đối chiếu
PAPER_AUC = {
    ("IHC-A", "No clinical"): 0.764, ("IHC-A", "+ Labs"): 0.768,
    ("IHC-A", "+ NLP raw"): 0.781, ("IHC-A", "+ NLP-PCA16"): 0.784,
    ("IHC-A", "+ Labs + NLP"): 0.767, ("IHC-A", "+ Labs + NLP-PCA16"): 0.753,
    ("IHC-A", "+ BioClinBERT-PCA16"): 0.767,
    ("IHC-G", "No clinical"): 0.784, ("IHC-G", "+ Labs"): 0.788,
    ("IHC-G", "+ NLP raw"): 0.813, ("IHC-G", "+ NLP-PCA16"): 0.812,
}
PAPER_SURV = {   # (C-index, Cox HR) của Bảng 7
    ("IHC-A", "No clinical"): (0.623, 5.30), ("IHC-A", "+ Labs"): (0.626, 5.31),
    ("IHC-A", "+ NLP raw"): (0.625, 5.91), ("IHC-A", "+ NLP-PCA16"): (0.628, 6.07),
    ("IHC-G", "No clinical"): (0.626, 4.74), ("IHC-G", "+ Labs"): (0.632, 4.49),
    ("IHC-G", "+ NLP raw"): (0.632, 4.97), ("IHC-G", "+ NLP-PCA16"): (0.631, 4.54),
}
PAPER_KM = {     # χ² của Bảng 8
    ("IHC-A", "No clinical"): 28.17, ("IHC-A", "+ NLP-PCA16"): 23.74,
    ("IHC-G", "No clinical"): 20.07, ("IHC-G", "+ NLP-PCA16"): 28.92,
}


def load_clinical():
    coh = pd.read_csv(f"{BASE_DB_DIR}/final_cohort_listing.csv").set_index("main_index")
    return get_clinical_table_v2(
        path=f"{BASE_DB_DIR}/18193MSKMINDProjectM-OmnibusInventory_DATA_2021-12-20_1540"
             f"-WITH-TB-and-SCANNER.csv",
        main_index_col="dmp_pt_id", cohort=coh[coh["cohort"] == "discovery"])


def km_chi2(summ, clin):
    """Log-rank χ² khi chia hai nhóm tại median điểm nguy cơ."""
    c = clin.loc[summ.index]
    hi = (summ["score"] > summ["score"].median()).values
    r = logrank_test(c.loc[hi, "pfs"], c.loc[~hi, "pfs"],
                     event_observed_A=c.loc[hi, "pfs_censor"],
                     event_observed_B=c.loc[~hi, "pfs_censor"])
    return float(r.test_statistic)


def run_5seed(ctx, clin, arm_mod, extra, no_scale_names):
    """Huấn luyện 5 seed (hoán vị nhãn như harness chuẩn), trả mọi chỉ số per-seed."""
    mods = BASE + [arm_mod] + extra
    data, mask, labels = get_training_data(mods, ctx.modality_dict,
                                           ctx.modality_MASK, ctx.df_outcomes)
    params = dict(MODEL_PARAMS)
    idx = [list(mask.columns).index(m) for m in no_scale_names if m in mask.columns]
    if idx:
        params["no_scale"] = idx

    y_full = labels["label"].astype(float)
    out = {"auc": [], "cindex": [], "hr": [], "km": []}
    cols = {}
    for seed in B.SEEDS:
        rng = np.random.default_rng(seed)
        shuffled = labels.iloc[rng.permutation(len(labels))]
        summ, _ = train(data, mask, shuffled, ctx.rad_filters, params,
                        folds=FOLDS, seed=seed)
        s = summ["score"].astype(float)
        cols[f"seed_{seed}"] = pd.Series(s.values, index=summ.index)
        out["auc"].append(float(roc_auc_score(y_full.loc[summ.index], s)))
        out["cindex"].append(float(compute_cindex(summ, clin)))
        try:
            row = compute_multivariate_cox(summ, clin).summary.loc["score"]
            out["hr"].append(float(row["exp(coef)"]))
        except Exception:
            out["hr"].append(float("nan"))
        out["km"].append(km_chi2(summ, clin))

    scores = pd.DataFrame(cols)
    out["_scores"] = scores
    out["_labels"] = y_full.loc[scores.index]
    return out


def ms(v):
    a = np.asarray(v, dtype=float)
    return float(np.nanmean(a)), float(np.nanstd(a, ddof=1))


def main():
    ctx = load_all()
    add_nlp_modality(ctx)
    add_pca16(ctx, "cnl_nlp", "cnl_nlp_pca16")
    add_bioclinbert(ctx)
    add_pca16(ctx, "cnl_bioclin", "cnl_bioclin_pca16")
    clin = load_clinical()
    print(f"\n5-seed x {FOLDS}-fold, seeds={B.SEEDS}\n")

    payload = {"config": {**MODEL_PARAMS, "folds": FOLDS, "seeds": B.SEEDS,
                          "model": "train (DyAM)", "note": "thay cho so don-phan-hoach Bang 5-8"},
               "arms": {}}
    hdr = (f"{'arm':<7}{'variant':<22}{'AUC 5-seed':>18}{'bài':>7}"
           f"{'C-index':>16}{'bài':>7}{'Cox HR':>15}{'bài':>7}{'KM χ²':>15}")
    print(hdr); print("-" * len(hdr))

    for arm, arm_mod in ARMS.items():
        payload["arms"][arm] = {}
        ref_res = None
        for name, extra, no_scale in VARIANTS:
            if (arm, name) not in PAPER_AUC:
                continue                       # Bảng 6/7 nhánh IHC-G chỉ có 4 biến thể
            r = run_5seed(ctx, clin, arm_mod, extra, no_scale)
            if ref_res is None:
                ref_res = r                    # "No clinical" làm mốc cho bootstrap
            auc_m, auc_s = ms(r["auc"]); ci_m, ci_s = ms(r["cindex"])
            hr_m, hr_s = ms(r["hr"]);    km_m, km_s = ms(r["km"])
            p_auc = PAPER_AUC.get((arm, name))
            p_ci, p_hr = PAPER_SURV.get((arm, name), (None, None))
            p_km = PAPER_KM.get((arm, name))
            entry = {
                "auc": {"mean": auc_m, "sd": auc_s, "per_seed": r["auc"], "paper": p_auc},
                "cindex": {"mean": ci_m, "sd": ci_s, "per_seed": r["cindex"], "paper": p_ci},
                "cox_hr": {"mean": hr_m, "sd": hr_s, "per_seed": r["hr"], "paper": p_hr},
                "km_chi2": {"mean": km_m, "sd": km_s, "per_seed": r["km"], "paper": p_km},
            }
            if r is not ref_res:
                d, lo, hi, p = paired_bootstrap(
                    {"_scores": r["_scores"], "_labels": r["_labels"]},
                    {"_scores": ref_res["_scores"], "_labels": ref_res["_labels"]})
                entry["vs_no_clinical"] = {"delta_auc": d, "ci": [lo, hi], "p": p}
            payload["arms"][arm][name] = entry
            save_result("verify_nlp_5seed", payload)
            print(f"{arm:<7}{name:<22}{f'{auc_m:.4f}±{auc_s:.3f}':>18}"
                  f"{(f'{p_auc:.3f}' if p_auc else '-'):>7}"
                  f"{f'{ci_m:.3f}±{ci_s:.3f}':>16}{(f'{p_ci:.3f}' if p_ci else '-'):>7}"
                  f"{f'{hr_m:.2f}±{hr_s:.2f}':>15}{(f'{p_hr:.2f}' if p_hr else '-'):>7}"
                  f"{f'{km_m:.1f}±{km_s:.1f}':>15}"
                  f"{(f'  (bài {p_km:.2f})' if p_km else '')}", flush=True)

    print("\nXONG — experiments/results/verify_nlp_5seed.json")


if __name__ == "__main__":
    main()
