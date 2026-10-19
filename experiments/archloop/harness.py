"""Harness chung của vòng lặp: Bước A (sàng lọc) và Bước B (xác nhận) theo protocol.md.

    python experiments/archloop/harness.py control                      (fusion mean phải = 0.7927)
    python experiments/archloop/harness.py screen  <tên_ứng_viên> ...
    python experiments/archloop/harness.py confirm <tên_ứng_viên> ...   (tiếp tục được chỗ dở)

Ứng viên đăng ký trong CANDIDATES: tên -> hàm train(data, mask, labels, l1, params, folds, seed).
Điểm từng bệnh nhân lưu ở experiments/results/ovo_proof/{tên}__{slug}.csv để analyze.py so với uniform.
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
import torch                                                      # noqa: E402
from sklearn.metrics import roc_auc_score                         # noqa: E402

from allcombo.run_paper_combos import PAPER_COMBOS, make_filter   # noqa: E402
from archloop.cands import CANDIDATES, OFF                      # noqa: E402
from common import benchmarks as B                                # noqa: E402
from common.data_setup import load_all                            # noqa: E402
from lung_helpers import get_training_data                        # noqa: E402

OUT = _CODE / "experiments" / "results" / "ovo_proof"
COMBOS = ["Rad+IHC-A+Gen", "Rad+IHC-G+Gen", "Rad+IHC-A+MutAmp", "Rad+IHC-A+Gen+PDL1",
          "Rad+IHC-G+Gen+PDL1", "Rad+IHC-A+Gen+TMB+PDL1"]
SCREEN = ["Rad+IHC-G+Gen+PDL1", "Rad+IHC-A+Gen+TMB+PDL1", "Rad+IHC-G+Gen"]
SEEDS20 = B.SEEDS + [11, 22, 33, 44, 55, 66, 77, 88, 99, 101, 202, 303, 404, 505, 606]

def slug(label):
    return label.replace("+", "_").replace("-", "").replace(" ", "")


def run_into(name, fn, combos, seeds, ctx):
    """Chạy (tiếp tục chỗ dở) và lưu điểm; trả dict combo -> DataFrame."""
    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    for label in combos:
        mods, ftype = next((m, f) for lb, m, f in PAPER_COMBOS if lb == label)
        data, mask, labels = get_training_data(mods, ctx.modality_dict, ctx.modality_MASK,
                                               ctx.df_outcomes)
        l1 = make_filter(ftype, ctx)
        f = OUT / f"{name}__{slug(label)}.csv"
        df = (pd.read_csv(f, index_col=0) if f.exists()
              else pd.DataFrame({"label": labels["label"].astype(float)}))
        for seed in seeds:
            col = f"seed_{seed}"
            if col in df.columns:
                continue
            rng = np.random.default_rng(seed)
            summ, _ = fn(data, mask, labels.iloc[rng.permutation(len(labels))], l1,
                         dict(B.MODEL_PARAMS), folds=B.FOLDS, seed=seed)
            df[col] = pd.Series(summ["score"].astype(float).values, index=summ.index)
            df.to_csv(f)
        out[label] = df
        n_seed = sum(c.startswith("seed_") for c in df.columns)
        print(f"[{name}] {label}: {n_seed} seed", flush=True)
    return out


def auc(df, col):
    ok = df[col].notna()
    return roc_auc_score(df.loc[ok, "label"], df.loc[ok, col])


def screen(name, ctx):
    res = run_into(name, CANDIDATES[name], SCREEN, B.SEEDS, ctx)
    deltas = []
    print(f"\n{'tổ hợp':<26}{'uniform':>9}{name:>12}{'Δ':>9}{'thắng':>7}")
    for label, df in res.items():
        ref = pd.read_csv(OUT / f"uniform__{slug(label)}.csv", index_col=0)
        d = [auc(df, f"seed_{s}") - auc(ref, f"seed_{s}") for s in B.SEEDS]
        u = np.mean([auc(ref, f"seed_{s}") for s in B.SEEDS])
        deltas.append(np.mean(d))
        print(f"{label:<26}{u:>9.4f}{u + np.mean(d):>12.4f}{np.mean(d):>+9.4f}"
              f"{sum(x > 0 for x in d):>5}/5")
    m = float(np.mean(deltas))
    verdict = "QUA (>= +0.005)" if m >= 0.005 else "KHONG QUA"
    print(f"\nSANG LOC {name}: delta trung binh = {m:+.4f} -> {verdict}", flush=True)
    return m


def main(argv):
    torch.set_num_threads(2)
    ctx = load_all()
    mode = argv[1]
    if mode == "verify":
        for name in (argv[2:] or list(OFF)):
            res = run_into(name, OFF[name], ["Rad+IHC-G+Gen+PDL1"], [42], ctx)
            a = auc(res["Rad+IHC-G+Gen+PDL1"], "seed_42")
            print(f"VERIFY {name}: seed 42 BM1 = {a:.4f} (phai = 0.7927)", flush=True)
            (OUT / f"{name}__{slug('Rad+IHC-G+Gen+PDL1')}.csv").unlink()
    elif mode == "screen":
        for name in argv[2:]:
            screen(name, ctx)
    elif mode == "confirm":
        only = [a[len("--only="):] for a in argv if a.startswith("--only=")]
        combos = [COMBOS[int(i)] for i in only[0].split(",")] if only else COMBOS
        for name in [a for a in argv[2:] if not a.startswith("--")]:
            run_into(name, CANDIDATES[name], combos, SEEDS20, ctx)
        print("XONG")


if __name__ == "__main__":
    main(sys.argv)
