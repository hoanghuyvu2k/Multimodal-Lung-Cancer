"""Đánh giá RAEF (model_raef.py) với bảng ablation từng thành phần.

    PYTHONIOENCODING=utf-8 python experiments/modelimp/run_raef.py [--smoke]
"""
import sys
from functools import partial
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
_CODE = _EXP.parent
for _p in (str(_CODE), str(_EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from modelimp import run_all as R                                 # noqa: E402  (vá scipy)
from modelimp.model_raef import train_raef                        # noqa: E402
from modelimp.run_design import BASE                              # noqa: E402
from common import benchmarks as B                                # noqa: E402
from common.data_setup import load_all                            # noqa: E402
from common.evaluate import load_result, save_result              # noqa: E402

VARIANTS_CV = {"R-cv (độ tin cậy ước lượng CV)": dict(use_rel=True, use_avail=False,
                                                     rel_mode="cv")}
VARIANTS = {"R (độ tin cậy)": dict(use_rel=True, use_avail=False),
            "A (sẵn-có)": dict(use_rel=False, use_avail=True),
            "RAEF (R+A)": dict(use_rel=True, use_avail=True)}


def main(argv):
    smoke = "--smoke" in argv
    ctx = load_all()
    combos = ["Rad+IHC-G+Gen+PDL1"] if smoke else [c for c in BASE if c != "Rad"]
    seeds = B.SEEDS[:1] if smoke else B.SEEDS
    variants = ({"tắt cả hai (phải = uniform_avg)": dict(use_rel=False, use_avail=False),
                 **VARIANTS} if smoke else VARIANTS)
    if "--cv" in argv:
        variants = VARIANTS_CV
    try:
        payload = load_result("raef")
    except Exception:
        payload = {"combos": {}}
    print(f"{'tổ hợp':<22}{'biến thể':<34}{'AUC':>16}{'Δ':>9}{'ensemble':>11}{'Δens':>9}")
    for label in combos:
        b_mean, b_ens = BASE[label]
        print(f"{label:<22}{'uniform_avg':<34}{b_mean:>16.4f}{0:>+9.4f}{b_ens:>11.4f}{0:>+9.4f}")
        for name, kw in variants.items():
            r = R.run_seeds(partial(train_raef, **kw), ctx, label, seeds)
            if not smoke:
                payload["combos"].setdefault(label, {})[name] = r
                save_result("raef", payload)
            print(f"{'':<22}{name:<34}{r['mean']:.4f}±{r['sd']:.3f}{r['mean'] - b_mean:>+9.4f}"
                  f"{r['ensemble']:>11.4f}{r['ensemble'] - b_ens:>+9.4f}", flush=True)
    print("XONG")


if __name__ == "__main__":
    import torch
    torch.set_num_threads(2)
    main(sys.argv)
