"""Tái lập Bảng 5 & 6 của bài báo: 7 biến thể encoding lâm sàng x 2 nhánh IHC.

Bảng 5/6 không có script sinh trong repo (notebook gốc không chứa cell NLP nào —
đã quét cả 66 cell), nguồn duy nhất là document/nlp-model/nlp-ket-qua.md.
Script này dựng lại chúng để kiểm chứng.

Config lấy từ cell 13 notebook (trùng survival_analysis_nlp.py):
    epochs=125, lr=0.01, alpha=0.001, beta=0.0, cross_modality_enabled=False
    rad filter l1_strength=0.1 cho PC/PL/LN, folds=10, DyAM (train)

Chạy từ code/:  PYTHONPATH=. PYTHONIOENCODING=utf-8 python experiments/nlp_verify/verify_nlp_variants.py
Kết quả:        experiments/results/verify_nlp_variants.json
                experiments/results/scores/verify_<arm>_<slug>.csv
"""
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
_ROOT = _EXP.parent
for _p in (str(_ROOT), str(_EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd                                              # noqa: E402
from sklearn.decomposition import PCA                            # noqa: E402

from common.data_setup import load_all                           # noqa: E402
from common.evaluate import save_result                          # noqa: E402
from nlp_verify.nlp_modality import add_nlp_modality             # noqa: E402
from lung_helpers import train, get_training_data, auc_roc_ci    # noqa: E402

MODEL_PARAMS = {"epochs": 125, "lr": 0.01, "alpha": 0.001, "beta": 0.0,
                "cross_modality_enabled": False}
FOLDS = 10
BASE = ["rad_lesion_pc", "rad_lesion_pl", "rad_lesion_ln",
        "gen_driver_mut_amp", "cnl_pdl1_score"]
ARMS = {"IHC-A": "path_ihc_pdl1", "IHC-G": "path_ihc_glcm"}

# (tên biến thể, modality thêm vào, modality cần no_scale)
VARIANTS = [
    ("No clinical",           [],                                  []),
    ("+ Labs",                ["cnl_dem_labs"],                    []),
    ("+ NLP raw",             ["cnl_nlp"],                         ["cnl_nlp"]),
    ("+ NLP-PCA16",           ["cnl_nlp_pca16"],                   ["cnl_nlp_pca16"]),
    ("+ Labs + NLP",          ["cnl_dem_labs", "cnl_nlp"],         ["cnl_nlp"]),
    ("+ Labs + NLP-PCA16",    ["cnl_dem_labs", "cnl_nlp_pca16"],   ["cnl_nlp_pca16"]),
    ("+ BioClinBERT-PCA16",   ["cnl_bioclin_pca16"],               ["cnl_bioclin_pca16"]),
]

# Số trong bài báo để đối chiếu tự động (Bảng 5 / Bảng 6)
PAPER = {
    ("IHC-A", "No clinical"):         (0.764, "[0.695-0.833]"),
    ("IHC-A", "+ Labs"):              (0.768, "[0.700-0.837]"),
    ("IHC-A", "+ NLP raw"):           (0.781, "[0.717-0.845]"),
    ("IHC-A", "+ NLP-PCA16"):         (0.784, "[0.719-0.848]"),
    ("IHC-A", "+ Labs + NLP"):        (0.767, "[0.700-0.835]"),
    ("IHC-A", "+ Labs + NLP-PCA16"):  (0.753, "[0.682-0.823]"),
    ("IHC-A", "+ BioClinBERT-PCA16"): (0.767, "[0.701-0.833]"),
    ("IHC-G", "No clinical"):         (0.784, "[0.717-0.850]"),
    ("IHC-G", "+ Labs"):              (0.788, "[0.723-0.853]"),
    ("IHC-G", "+ NLP raw"):           (0.813, "[0.753-0.874]"),
    ("IHC-G", "+ NLP-PCA16"):         (0.812, "[0.751-0.873]"),
}

SCORE_DIR = _ROOT / "experiments" / "results" / "scores"


def slug(s):
    return (s.replace("+", "").replace(" ", "_").replace("-", "")
             .strip("_").lower())


def add_pca16(ctx, src, name):
    """PCA 16 chiều fit trên TOÀN BỘ cohort — đúng như pipeline gốc.

    Lưu ý (rủi ro R1 trong kế hoạch kiểm chứng): fit trước khi chia fold là rò rỉ
    dữ liệu nhẹ; giữ nguyên ở đây vì mục tiêu là tái lập đúng số cũ.
    """
    emb = ctx.modality_dict[src]
    present = emb.dropna(how="all")
    vals = PCA(n_components=16, random_state=42).fit_transform(present.values)
    df = pd.DataFrame(vals, index=present.index,
                      columns=[f"{name}_{i}" for i in range(16)])
    ctx.modality_dict[name] = df.reindex(ctx.df_outcomes.index)
    ctx.modality_MASK[name] = ctx.df_outcomes.index.isin(df.index)
    return df.shape


def add_bioclinbert(ctx, name="cnl_bioclin"):
    """Bio_ClinicalBERT 768-dim, prompt v2 (đúng như bản gốc), mean pooling."""
    from clinical_nlp_embedding import prepare_clinical_bert_modality
    labs = ctx.modality_dict["cnl_dem_labs"]
    present = ctx.modality_MASK.loc[labs.index, "cnl_dem_labs"].values.astype(bool)
    prepare_clinical_bert_modality(ctx.modality_dict, labs[present], ctx.modality_MASK,
                                   name=name, prompt_version="v2")


def run_variant(ctx, arm_mod, extra, no_scale_names):
    mods = BASE + [arm_mod] + extra
    data, mask, labels = get_training_data(mods, ctx.modality_dict,
                                           ctx.modality_MASK, ctx.df_outcomes)
    params = dict(MODEL_PARAMS)
    idx = [list(mask.columns).index(m) for m in no_scale_names if m in mask.columns]
    if idx:
        params["no_scale"] = idx
    summ, _ = train(data, mask, labels, ctx.rad_filters, params, folds=FOLDS)
    y, s = summ["label"].values, summ["score"].values
    auc, ci = auc_roc_ci(y, s, 0.95)
    return float(auc), (float(ci[0]), float(ci[1])), summ


def main():
    ctx = load_all()
    n_nlp = add_nlp_modality(ctx)                      # cnl_nlp, 384-dim từ cache
    add_pca16(ctx, "cnl_nlp", "cnl_nlp_pca16")
    add_bioclinbert(ctx)                               # cnl_bioclin, 768-dim
    add_pca16(ctx, "cnl_bioclin", "cnl_bioclin_pca16")
    print(f"NLP modality: {n_nlp} bệnh nhân | đã tạo cnl_nlp_pca16, cnl_bioclin_pca16\n")

    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"config": {**MODEL_PARAMS, "folds": FOLDS, "model": "train (DyAM)",
                          "rad_filter": "l1_strength=0.1 PC/PL/LN"},
               "arms": {}}

    hdr = f"{'arm':<7}{'variant':<22}{'AUC':>8}{'95% CI':>20}{'bài báo':>10}{'lệch':>9}"
    print(hdr); print("-" * len(hdr))
    for arm, arm_mod in ARMS.items():
        payload["arms"][arm] = {}
        for name, extra, no_scale in VARIANTS:
            if (arm, name) not in PAPER and arm == "IHC-G":
                continue                                # Bảng 6 chỉ có 4 biến thể
            auc, ci, summ = run_variant(ctx, arm_mod, extra, no_scale)
            summ.to_csv(SCORE_DIR / f"verify_{slug(arm)}_{slug(name)}.csv")
            ref = PAPER.get((arm, name), (None, ""))[0]
            diff = "" if ref is None else f"{auc - ref:+.4f}"
            flag = "" if ref is None or abs(auc - ref) <= 0.005 else "  <-- LỆCH"
            payload["arms"][arm][name] = {"auc": auc, "ci": ci, "paper": ref,
                                          "diff": None if ref is None else auc - ref}
            print(f"{arm:<7}{name:<22}{auc:>8.4f}{f'[{ci[0]:.3f}-{ci[1]:.3f}]':>20}"
                  f"{'' if ref is None else f'{ref:.3f}':>10}{diff:>9}{flag}", flush=True)
            save_result("verify_nlp_variants", payload)
    print("\nXONG — kết quả ở experiments/results/verify_nlp_variants.json")


if __name__ == "__main__":
    main()
