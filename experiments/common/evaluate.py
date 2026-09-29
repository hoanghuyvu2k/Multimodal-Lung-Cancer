"""
Đánh giá thống kê dùng chung.

Điểm chính: mọi so sánh đều chạy nhiều seed và dùng paired bootstrap trên bệnh nhân.
KHÔNG dùng DeLong CI trên score gộp từ CV — các fold không độc lập nên CI đó hẹp giả tạo.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_CODE_DIR = Path(__file__).resolve().parents[2]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from lung_helpers import get_training_data  # noqa: E402

from . import benchmarks as B  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def run_repeated_cv(train_fn, bm_key, ctx, seeds=None, model_params=None,
                    folds=None, extra_params=None, verbose=True):
    """
    Chạy train_fn trên một benchmark với nhiều seed.

    train_fn phải có chữ ký (modality_list_in, modality_mask, outcomes,
    l1_dfs_filter, model_params, folds, seed) -> (summary_df, coef_df)

    Trả về dict: mean_auc, sd_auc, per_seed_auc, scores (DataFrame patient x seed), labels
    """
    seeds = seeds if seeds is not None else B.SEEDS
    folds = folds if folds is not None else B.FOLDS
    params = dict(model_params if model_params is not None else B.MODEL_PARAMS)
    if extra_params:
        params.update(extra_params)

    bm = B.BENCHMARKS[bm_key]
    data, mask, labels = get_training_data(
        bm["modalities"], ctx.modality_dict, ctx.modality_MASK, ctx.df_outcomes)
    l1_filter = B.get_filters(bm_key, ctx)

    per_seed_auc = []
    score_cols = {}
    label_series = None

    for seed in seeds:
        # Tham số `seed` của train()/train_ovo() là VÔ HIỆU: cả hai hardcode
        # KFold(random_state=0) và AttentionMatrix.__init__ hardcode torch.manual_seed(42),
        # nên mọi seed cho ra kết quả giống hệt nhau (đã kiểm chứng: 5 seed -> cùng AUC 0.7976).
        # Cách lấy được phân hoạch CV khác nhau mà không phải sửa lung_helpers.py: hoán vị thứ tự
        # hàng của outcomes. train() gọi kf.split(outcomes.index) rồi lấy outcomes.index[train],
        # còn dữ liệu truy cập bằng .loc nên vẫn khớp đúng bệnh nhân.
        # Lưu ý: khởi tạo model vẫn cố định ở 42, nên sd đo được là phương sai do phân hoạch CV
        # (thành phần trội), không bao gồm phương sai do khởi tạo.
        rng = np.random.default_rng(seed)
        shuffled = labels.iloc[rng.permutation(len(labels))]

        summary_df, _ = train_fn(data, mask, shuffled, l1_filter, params,
                                 folds=folds, seed=seed)
        # Với OvO, summary_df chứa cả mảng 2D (mixing_matrix) trong các cột attn_*, khiến
        # toàn bộ DataFrame bị ép về dtype object -> roc_auc_score báo "unknown format".
        # Ép lại về float ở đây thay vì sửa lung_helpers.
        y = summary_df["label"].astype(float).values
        s = summary_df["score"].astype(float).values
        per_seed_auc.append(float(roc_auc_score(y, s)))
        score_cols[f"seed_{seed}"] = pd.Series(s, index=summary_df.index)
        if label_series is None:
            label_series = pd.Series(y, index=summary_df.index)
        if verbose:
            print(f"      seed {seed:>6}: AUC = {per_seed_auc[-1]:.4f}")

    scores = pd.DataFrame(score_cols)
    return {
        "benchmark": bm_key,
        "benchmark_name": bm["name"],
        "n_samples": int(len(label_series)),
        "seeds": list(seeds),
        "per_seed_auc": per_seed_auc,
        "mean_auc": float(np.mean(per_seed_auc)),
        "sd_auc": float(np.std(per_seed_auc, ddof=1)) if len(per_seed_auc) > 1 else 0.0,
        "_scores": scores,
        "_labels": label_series,
    }


def paired_bootstrap(res_a, res_b, n_boot=2000, seed=0):
    """
    Paired bootstrap trên bệnh nhân: resample bệnh nhân, tính AUC_a - AUC_b
    dùng score trung bình qua các seed. Trả về (delta, ci_low, ci_high, p_value).

    p_value là two-sided, tỉ lệ lần bootstrap đổi dấu so với delta quan sát.
    """
    idx = res_a["_labels"].index.intersection(res_b["_labels"].index)
    y = res_a["_labels"].loc[idx].values
    sa = res_a["_scores"].loc[idx].mean(axis=1).values
    sb = res_b["_scores"].loc[idx].mean(axis=1).values

    delta_obs = roc_auc_score(y, sa) - roc_auc_score(y, sb)

    rng = np.random.default_rng(seed)
    n = len(idx)
    deltas = []
    for _ in range(n_boot):
        pick = rng.integers(0, n, n)
        yb = y[pick]
        if len(np.unique(yb)) < 2:
            continue
        deltas.append(roc_auc_score(yb, sa[pick]) - roc_auc_score(yb, sb[pick]))

    deltas = np.array(deltas)
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return float(delta_obs), float(ci_low), float(ci_high), float(min(p, 1.0))


def strip_internals(res):
    """Bỏ các key nội bộ (DataFrame) để serialize JSON."""
    return {k: v for k, v in res.items() if not k.startswith("_")}


def save_scores(method, bm_key, res):
    """Lưu score per-seed + label ra CSV để step sau ghép cặp mà không phải chạy lại."""
    d = RESULTS_DIR / "scores"
    d.mkdir(parents=True, exist_ok=True)
    df = res["_scores"].copy()
    df["label"] = res["_labels"]
    df.to_csv(d / f"{method}__{bm_key}.csv")


def load_scores(method, bm_key):
    """Đọc lại thành cấu trúc dùng được cho paired_bootstrap."""
    df = pd.read_csv(RESULTS_DIR / "scores" / f"{method}__{bm_key}.csv", index_col=0)
    labels = df.pop("label")
    return {"_scores": df, "_labels": labels}


def save_result(name, payload):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"   -> đã ghi {path}")
    return path


def load_result(name):
    with open(RESULTS_DIR / f"{name}.json", encoding="utf-8") as fh:
        return json.load(fh)


def summary_table(results_by_method):
    """results_by_method: {method_name: {bm_key: result_dict}} -> DataFrame mean±sd."""
    rows = []
    for method, by_bm in results_by_method.items():
        row = {"method": method}
        for bm_key, res in by_bm.items():
            row[bm_key] = f"{res['mean_auc']:.4f} ± {res['sd_auc']:.4f}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")
