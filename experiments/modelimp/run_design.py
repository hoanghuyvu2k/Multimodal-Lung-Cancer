"""Đánh giá mô hình đề xuất SHARE-Rad (model_shared.py), tách từng thành phần.

    M1  share   chia sẻ trọng số radiomics PC/PL/LN + chọn đặc trưng trên tổn thương gộp
    M2  light   chế độ nhẹ (lr 0.005, alpha 0.0001) cho kênh không phải radiomics
    M3  cả hai

Mốc `uniform_avg` lấy từ các lần chạy trước (cùng harness, cùng seed) — không chạy lại.
Bài học từ hướng 4: KHÔNG kết luận từ một tổ hợp / 3 seed -> 7 tổ hợp, đủ 5 seed.

    PYTHONIOENCODING=utf-8 python experiments/modelimp/run_design.py [--smoke]
"""
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
_CODE = _EXP.parent
for _p in (str(_CODE), str(_EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from functools import partial                                     # noqa: E402

from modelimp import run_all as R                                 # noqa: E402  (áp luôn bản vá scipy)
from modelimp.model_shared import RAD_SITES, train_shared         # noqa: E402
from allcombo.run_paper_combos import PAPER_COMBOS                # noqa: E402
from common import benchmarks as B                                # noqa: E402
from common.data_setup import load_all                            # noqa: E402
from common.evaluate import load_result, save_result              # noqa: E402

# mốc uniform_avg: (AUC trung bình 5 seed, AUC ensemble) — từ modelimp.json hướng 2/5/6
BASE = {
    "Rad":                (0.6661, 0.6925),
    "Rad+Gen":            (0.7160, 0.7291),
    "Rad+IHC-G+Gen":      (0.7510, 0.7631),
    "Rad+IHC-A+Gen+PDL1": (0.7547, 0.7672),
    "Rad+IHC-G+Gen+PDL1": (0.7730, 0.7838),
    "PDL1+Gen":           (0.7191, 0.7241),
    "IHC-G+Gen":          (0.7184, 0.7296),
}
VARIANTS = {"share": dict(share=True, light=False),
            "light": dict(share=False, light=True),
            "share+light": dict(share=True, light=True)}


def has_rad(label):
    mods = next(m for lb, m, _f in PAPER_COMBOS if lb == label)
    return sum(m in RAD_SITES for m in mods) >= 2


def main(argv):
    smoke = "--smoke" in argv
    only = [a[len("--v="):] for a in argv if a.startswith("--v=")]
    variants = {k: v for k, v in VARIANTS.items() if not only or k in only[0].split(",")}
    ctx = load_all()
    combos = ["Rad"] if smoke else [c for c in BASE if c != "Rad" or "share" in variants]
    seeds = B.SEEDS[:1] if smoke else B.SEEDS
    try:
        payload = load_result("share_rad")
    except Exception:
        payload = {"combos": {}}

    print(f"{'tổ hợp':<22}{'biến thể':<14}{'AUC':>16}{'Δ':>9}{'ensemble':>11}{'Δens':>9}")
    for label in combos:
        b_mean, b_ens = BASE[label]
        print(f"{label:<22}{'uniform_avg':<14}{b_mean:>16.4f}{0:>+9.4f}{b_ens:>11.4f}{0:>+9.4f}")
        for name, kw in variants.items():
            if "share" in name and not has_rad(label):
                continue                                # không có radiomics -> share vô nghĩa
            r = R.run_seeds(partial(train_shared, **kw), ctx, label, seeds)
            payload["combos"].setdefault(label, {})[name] = r
            if not smoke:
                save_result("share_rad", payload)
            print(f"{'':<22}{name:<14}{r['mean']:.4f}±{r['sd']:.3f}{r['mean'] - b_mean:>+9.4f}"
                  f"{r['ensemble']:>11.4f}{r['ensemble'] - b_ens:>+9.4f}", flush=True)
    print("\nXONG")


if __name__ == "__main__":
    import torch
    torch.set_num_threads(2)
    main(sys.argv)
