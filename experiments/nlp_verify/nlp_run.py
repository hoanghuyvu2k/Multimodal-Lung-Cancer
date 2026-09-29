"""Chạy uniform_avg cho 1 tổ hợp modality: 5-seed (mean±sd) + single-run. Xử lý no_scale cho NLP + l1 rad."""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments"))

from lung_helpers import get_training_data                      # noqa: E402
from baselines.model_uniform_avg import train_uniform_avg       # noqa: E402
from common import benchmarks as B                              # noqa: E402

# scipy>=1.14 (bản duy nhất cho py3.13) raise BracketError trên cột radiomics suy biến;
# scipy cũ (conda <1.14, chạy được sáng nay) thì bỏ qua. Khôi phục hành vi cũ: cột nào brent
# không bracket được -> lambda=1 (gần identity). Áp từ đây, KHÔNG sửa lung_helpers.
import sklearn.preprocessing._data as _skd                      # noqa: E402
_orig_yj = _skd.PowerTransformer._yeo_johnson_optimize
def _safe_yj(self, x):
    try:
        return _orig_yj(self, x)
    except Exception:
        return 1.0
_skd.PowerTransformer._yeo_johnson_optimize = _safe_yj

RAD = {"rad_lesion_pc": 0, "rad_lesion_pl": 1, "rad_lesion_ln": 2}
_CLEANED = {"done": False}


def clean_rad_filters(ctx):
    """Loại các cột radiomics suy biến (PowerTransformer scipy>=1.14 bracket lỗi). Chạy 1 lần.
    Cột near-constant có coef elastic-net ~0 nên không ảnh hưởng feature được chọn / AUC."""
    if _CLEANED["done"]:
        return
    import pandas as pd
    from sklearn.preprocessing import PowerTransformer
    from lung_helpers import RAD_JOB_TAG
    for pos, filt in ctx.rad_filters.items():
        df = filt["l1_selection_df"]
        sub = df[df["job_tag"] == RAD_JOB_TAG]
        feats = [c for c in sub.columns if c not in ("job_tag", "site")
                 and pd.api.types.is_numeric_dtype(sub[c])]
        bad = []
        for c in feats:
            col = sub[c].dropna()
            if col.std() < 1e-3:
                bad.append(c); continue
            try:
                PowerTransformer().fit_transform(col.to_frame())
            except Exception:
                bad.append(c)
        if bad:
            filt["l1_selection_df"] = df.drop(columns=bad)
    _CLEANED["done"] = True


def build_l1(mods, ctx):
    clean_rad_filters(ctx)
    return {mods.index(n): ctx.rad_filters[RAD[n]] for n in mods if n in RAD}


def _auc(summ):
    return float(roc_auc_score(summ["label"].astype(float).values, summ["score"].astype(float).values))


def run_variant(mods, ctx, seeds=None, train_fn=train_uniform_avg):
    seeds = seeds if seeds is not None else B.SEEDS
    data, mask, labels = get_training_data(mods, ctx.modality_dict, ctx.modality_MASK, ctx.df_outcomes)
    l1 = build_l1(mods, ctx)
    params = dict(B.MODEL_PARAMS)
    params["no_scale"] = [mods.index("cnl_nlp")] if "cnl_nlp" in mods else []

    # single-run: thứ tự tự nhiên (như bài báo)
    summ, _ = train_fn(data, mask, labels, l1, params, folds=B.FOLDS, seed=42)
    single = _auc(summ)

    # 5-seed hoán vị
    per = []
    for s in seeds:
        rng = np.random.default_rng(s)
        sh = labels.iloc[rng.permutation(len(labels))]
        summ, _ = train_fn(data, mask, sh, l1, params, folds=B.FOLDS, seed=s)
        per.append(_auc(summ))
    return {"mean": float(np.mean(per)), "sd": float(np.std(per, ddof=1)),
            "per_seed": [float(x) for x in per], "single_run": single}
