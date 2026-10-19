"""Vì sao lần chạy đơn (single-run) dao động? Tách 2 nguồn cho DyAM vs OvO, cấu hình notebook (bộ lọc rad l1_strength 0.1).

  A) PHÂN HOẠCH CỐ ĐỊNH (thứ tự tự nhiên, như notebook), đổi seed khởi tạo mạng: 20 seed  -> sd do khởi tạo
  B) KHỞI TẠO CỐ ĐỊNH (seed 42), đổi phân hoạch (hoán vị thứ tự bệnh nhân -> KFold khác): 20 phân hoạch -> sd do chia fold
  D) tính xác định: seed 42 chạy lại 2 lần (kỳ vọng hệt nhau)
  C) chỉ BM1: A với bộ lọc radiomics 'clean_rad_filters' (đường chạy của Bảng 4/5)

    python experiments/nlp_verify/run_single_variance.py BM1
Kết quả: experiments/results/single_variance_<BM>.json (lưu tăng dần).
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lung_helpers import get_training_data, train as train_dyam, train_ovo   # noqa: E402  (HÀM GỐC của lung_helpers, như notebook)
from common import benchmarks as B                          # noqa: E402
from common.data_setup import load_all                      # noqa: E402
from nlp_verify import nlp_run                              # noqa: E402  (vá scipy)

OUT = Path(__file__).resolve().parents[1] / "results"
INIT_SEEDS = [42, 7, 123, 2024, 31337, 11, 22, 33, 44, 55, 66, 77, 88, 99, 101, 202, 303, 404, 505, 606]
PART_SEEDS = INIT_SEEDS            # dùng cùng danh sách làm seed hoán vị


def auc(s):
    return float(roc_auc_score(s["label"].astype(float).values, s["score"].astype(float).values))


def safe(fn, *a, **k):
    """Hàm gốc crash khi 1 fold mất hết đặc trưng của 1 nguồn (ValueError '0 feature(s)') -> ghi None, đếm lần crash."""
    try:
        return auc(fn(*a, **k)[0])
    except ValueError as e:
        return None


def main():
    bm = sys.argv[1]
    cfg = B.BENCHMARKS[bm]
    mods = cfg["modalities"]
    ctx = load_all()
    data, mask, labels = get_training_data(mods, ctx.modality_dict, ctx.modality_MASK, ctx.df_outcomes)
    l1 = B.get_filters(bm, ctx)
    models = {"dyam": train_dyam, "ovo": train_ovo}
    f = OUT / f"single_variance_{bm}.json"
    res = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}

    def save():
        f.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")

    for name, fn in models.items():
        p = dict(B.MODEL_PARAMS)
        r = res.setdefault(name, {})
        if "A_init" not in r:
            r["A_init"] = [safe(fn, data, mask, labels, l1, p, folds=B.FOLDS, seed=s) for s in INIT_SEEDS]; save()
            print(bm, name, "A init-sd", np.nanstd([x for x in r["A_init"] if x is not None], ddof=1), flush=True)
        if "D_repeat" not in r:
            r["D_repeat"] = [safe(fn, data, mask, labels, l1, p, folds=B.FOLDS, seed=42) for _ in range(2)]; save()
        if "B_part" not in r:
            vals = []
            for s in PART_SEEDS:
                sh = labels.iloc[np.random.default_rng(s).permutation(len(labels))]
                vals.append(safe(fn, data, mask, sh, l1, p, folds=B.FOLDS, seed=42))
            r["B_part"] = vals; save()
            print(bm, name, "B partition-sd", np.nanstd([x for x in vals if x is not None], ddof=1), flush=True)
        if bm == "BM1" and "C_clean" not in r:
            nlp_run.clean_rad_filters(ctx)
            l1c = B.get_filters(bm, ctx)
            r["C_clean"] = [safe(fn, data, mask, labels, l1c, p, folds=B.FOLDS, seed=s) for s in INIT_SEEDS]; save()
            print(bm, name, "C clean-filter mean", np.mean([x for x in r["C_clean"] if x is not None]), flush=True)
            # trả bộ lọc về nguyên bản cho model kế tiếp: nạp lại ctx
            ctx = load_all(); data, mask, labels = get_training_data(mods, ctx.modality_dict, ctx.modality_MASK, ctx.df_outcomes)
            l1 = B.get_filters(bm, ctx)
    print("XONG", bm)


if __name__ == "__main__":
    main()
