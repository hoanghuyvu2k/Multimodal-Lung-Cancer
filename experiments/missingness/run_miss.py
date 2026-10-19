"""Khuyết modality có phải là tín hiệu? Mã hoá tường minh mẫu khuyết thành một modality.

Phát hiện dẫn tới thí nghiệm này (đo trên điểm số đã lưu của uniform_avg, BM1):
    - AUC chỉ dùng SỐ modality có sẵn (k)            = 0.665
    - AUC chỉ dùng "có tiêu bản pathology hay không"  = 0.668
    - Tỉ lệ không đáp ứng theo k: k=1 100% · k=2 85% · k=3 80% · k=4 70% · k=5 50%
    - AUC trong từng nhóm k                          = 0.69-0.76 (mô hình vẫn có giá trị riêng)

Tức là mẫu khuyết mang thông tin tiên lượng mạnh, và mô hình hiện tại chỉ hấp thụ nó một cách
gián tiếp (điểm trung bình trên các nhánh có sẵn). Thí nghiệm: đưa vector mask thành một modality
tường minh và đo xem có cải thiện không.

    PYTHONIOENCODING=utf-8 python experiments/missingness/run_miss.py
"""
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
_CODE = _EXP.parent
for _p in (str(_CODE), str(_EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sklearn.preprocessing._data as _skd                        # noqa: E402

_orig_yj = _skd.PowerTransformer._yeo_johnson_optimize


def _safe_yj(self, x):                                            # vá scipy >= 1.14
    try:
        return _orig_yj(self, x)
    except Exception:
        return 1.0


_skd.PowerTransformer._yeo_johnson_optimize = _safe_yj

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from sklearn.metrics import roc_auc_score                         # noqa: E402

from allcombo.run_paper_combos import PAPER_COMBOS, make_filter    # noqa: E402
from baselines.model_uniform_avg import train_uniform_avg          # noqa: E402
from common import benchmarks as B                                 # noqa: E402
from common.data_setup import load_all                             # noqa: E402
from common.evaluate import load_result, paired_bootstrap, save_result   # noqa: E402
from nlp_verify.nlp_modality import add_nlp_modality               # noqa: E402
from lung_helpers import get_training_data                         # noqa: E402

BMS = ["Rad+IHC-G+Gen+PDL1", "Rad+IHC-G+Gen+PDL1+Labs", "Rad+Gen", "PDL1+Gen"]
MISS = "miss_pattern"


def add_miss_modality(ctx, mods, name=MISS):
    """Modality = vector nhị phân cho biết bệnh nhân có từng modality nào + tổng số.

    Luôn 'có sẵn' với mọi bệnh nhân, nên không tự tạo thêm khuyết.
    """
    base = [m for m in mods if m != name]
    mk = ctx.modality_MASK[base].astype(float)
    df = mk.copy()
    df.columns = [f"has_{c}" for c in base]
    df["n_available"] = mk.sum(axis=1)
    ctx.modality_dict[name] = df.reindex(ctx.df_outcomes.index).fillna(0.0)
    ctx.modality_MASK[name] = True
    return df.shape


def run(ctx, mods, l1, seeds):
    data, mask, labels = get_training_data(mods, ctx.modality_dict,
                                           ctx.modality_MASK, ctx.df_outcomes)
    params = dict(B.MODEL_PARAMS)
    no_scale = [i for i, c in enumerate(mask.columns) if c in ("cnl_nlp", MISS)]
    if no_scale:
        params["no_scale"] = no_scale
    y_full = labels["label"].astype(float)
    cols, per_seed = {}, []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        shuffled = labels.iloc[rng.permutation(len(labels))]
        summ, _ = train_uniform_avg(data, mask, shuffled, l1, params,
                                    folds=B.FOLDS, seed=seed)
        s = pd.Series(summ["score"].astype(float).values, index=summ.index)
        cols[f"seed_{seed}"] = s
        per_seed.append(float(roc_auc_score(y_full.loc[s.index], s)))
    scores = pd.DataFrame(cols)
    return {"mean": float(np.mean(per_seed)), "sd": float(np.std(per_seed, ddof=1)),
            "per_seed": per_seed, "_scores": scores,
            "_labels": y_full.loc[scores.index]}


def main(argv=None):
    wanted = [a for a in (argv or [])[1:] if not a.startswith("-")]
    if "--all" in (argv or []):
        bms = [lb for lb, _m, _f in PAPER_COMBOS]
    else:
        bms = [b for b in BMS if not wanted or b in wanted]   # khớp chính xác
    only = [a[len("--v="):] for a in (argv or []) if a.startswith("--v=")]
    only = only[0].split(",") if only else None
    ctx = load_all()
    add_nlp_modality(ctx)
    try:                                   # gộp vào kết quả cũ, KHÔNG ghi đè
        payload = load_result("missingness")
        payload.setdefault("bm", {})
    except Exception:
        payload = {"config": {"model": "uniform_avg", "seeds": B.SEEDS, "folds": B.FOLDS,
                              **B.MODEL_PARAMS}, "bm": {}}

    hdr = (f"{'tổ hợp':<26}{'base':>16}{'+miss':>16}{'+NLP':>16}{'+NLP+miss':>16}")
    print(hdr); print("-" * len(hdr))
    for label in bms:
        mods, ftype = next((m, f) for lb, m, f in PAPER_COMBOS if lb == label)
        l1 = make_filter(ftype, ctx)
        base = [m for m in mods if m != "cnl_dem_labs"]
        add_miss_modality(ctx, base)
        variants = {"base": base,
                    "miss": base + [MISS],
                    "nlp": base + ["cnl_nlp"],
                    "nlp_miss": base + ["cnl_nlp", MISS]}
        if only:
            variants = {k: v for k, v in variants.items() if k in only or k == "base"}
        res = {k: run(ctx, v, l1, B.SEEDS) for k, v in variants.items()}
        row = {}
        for k, r in res.items():
            row[k] = {x: r[x] for x in ("mean", "sd", "per_seed")}
            if k != "base":
                d, lo, hi, p = paired_bootstrap(
                    {"_scores": r["_scores"], "_labels": r["_labels"]},
                    {"_scores": res["base"]["_scores"], "_labels": res["base"]["_labels"]})
                row[k]["vs_base"] = {"delta": d, "ci": [lo, hi], "p": p}
        payload["bm"].setdefault(label, {}).update(row)
        save_result("missingness", payload)
        print(f"{label:<26}" + "".join(
            f"{res[k]['mean']:.4f}±{res[k]['sd']:.3f}".rjust(16)
            for k in ("base", "miss", "nlp", "nlp_miss") if k in res), flush=True)

    print("\nXONG — experiments/results/missingness.json")


if __name__ == "__main__":
    import torch
    torch.set_num_threads(2)          # giảm bộ nhớ/CPU khi máy đang thiếu RAM
    main(sys.argv)
