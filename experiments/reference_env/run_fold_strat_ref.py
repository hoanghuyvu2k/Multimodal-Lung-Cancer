"""Cách chia fold có làm dao động AUC giữa các seed không? BM1 (Rad+IHC-G+Gen+PDL1), DyAM và OvO, 20 seed, ref-linux-py38.

Ba cách chia (cùng mã huấn luyện, chỉ đổi bộ chia fold; phần khác của giao thức không đổi):
  random : KFold(10, shuffle, random_state=0) trên thứ tự bệnh nhân đã hoán vị theo seed (giao thức chuẩn)
  label  : StratifiedKFold theo nhãn
  pattern: StratifiedKFold theo nhãn x mẫu nguồn có mặt (mẫu hiếm < MIN_PAT bệnh nhân gộp thành 'rare')

    run_ref.sh experiments/reference_env/run_fold_strat_ref.py random|label|pattern [--stats]
Kết quả: experiments/results/ref/foldstrat/{scheme}__{dyam|ovo}.csv (điểm theo seed, chạy tiếp được), fold_stats.json (--stats)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1])); sys.path.insert(0, str(_HERE.parents[2]))

from sklearn.metrics import roc_auc_score                                     # noqa: E402
from sklearn.model_selection import KFold, StratifiedKFold                    # noqa: E402

from common import benchmarks as B                                            # noqa: E402
from common.data_setup import load_all                                        # noqa: E402
from allcombo.run_paper_combos import PAPER_COMBOS, make_filter               # noqa: E402
from allcombo import run_combo as RC                                          # noqa: E402
from allcombo import run_ovo_singlerun as RO                                  # noqa: E402
from archloop.harness import SEEDS20                                          # noqa: E402
from lung_helpers import get_training_data                                    # noqa: E402

OUT = _HERE.parents[1] / "results" / "ref" / "foldstrat"
OUT.mkdir(parents=True, exist_ok=True)
MIN_PAT = 20
KEY = {}


class StratSplit:
    """Thay KFold trong các hàm train: cùng giao diện, phân tầng theo KEY[mã bệnh nhân]."""

    def __init__(self, n_splits=10, random_state=0, shuffle=True):
        self.n = n_splits

    def split(self, X, y=None, groups=None):
        keys = np.array([KEY[i] for i in X])
        return StratifiedKFold(self.n, shuffle=True, random_state=0).split(np.zeros(len(keys)), keys)


def build_keys(mask, labels, scheme):
    lab = labels["label"].astype(int)
    if scheme == "label":
        return {i: str(lab[i]) for i in lab.index}
    pat = mask.astype(int).astype(str).agg("".join, axis=1)
    cnt = pat.value_counts()
    pat = pat.where(pat.map(cnt) >= MIN_PAT, "rare")
    key = pd.Series({i: f"{lab[i]}|{pat[i]}" for i in lab.index})
    small = key.map(key.value_counts()) < 10        # tầng < số fold: gộp vào 'rare' của cùng nhãn
    key[small] = [f"{lab[i]}|rare" for i in key.index[small]]
    return key.to_dict()


def main():
    scheme = sys.argv[1]
    ctx = load_all()
    mods, ftype = next((m, f) for lb, m, f in PAPER_COMBOS if lb == "Rad+IHC-G+Gen+PDL1")
    data, mask, labels = get_training_data(mods, ctx.modality_dict, ctx.modality_MASK, ctx.df_outcomes)
    l1 = make_filter(ftype, ctx)
    y = labels["label"].astype(float)

    if "--stats" in sys.argv:
        stats = {}
        for sch in ("random", "label", "pattern"):
            if sch != "random":
                KEY.update(build_keys(mask, labels, sch))
            rows = []
            for seed in SEEDS20:
                perm = labels.iloc[np.random.default_rng(seed).permutation(len(labels))]
                idx = perm.index
                spl = (KFold(10, random_state=0, shuffle=True) if sch == "random" else StratSplit()).split(idx)
                comp = []
                for tr, te in spl:
                    ids = idx[te]
                    m = mask.loc[ids]
                    comp.append([y[ids].mean(), m.iloc[:, 3].mean(), m.astype(int).sum(axis=1).mean()])
                comp = np.array(comp)
                rows.append(comp.std(axis=0, ddof=1))
            r = np.mean(rows, axis=0)
            stats[sch] = {"sd_across_folds_pos_rate": float(r[0]), "sd_across_folds_frac_pathology": float(r[1]), "sd_across_folds_mean_n_sources": float(r[2])}
            print(sch, stats[sch], flush=True)
        (OUT / "fold_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
        pat = mask.astype(int).astype(str).agg("".join, axis=1)
        print("mẫu nguồn có mặt (cột:", list(mask.columns), "):", pat.value_counts().to_dict())
        return

    if scheme != "random":
        KEY.update(build_keys(mask, labels, scheme))
        RC.KFold = StratSplit
        RO.KFold = StratSplit
        print("strata:", len(set(KEY.values())), flush=True)

    for name, fn in (("dyam", RC.train), ("ovo", RO.train_ovo_safe)):
        f = OUT / f"{scheme}__{name}.csv"
        df = pd.read_csv(f, index_col=0) if f.exists() else pd.DataFrame({"label": y})
        for seed in SEEDS20:
            col = f"seed_{seed}"
            if col in df.columns:
                continue
            perm = labels.iloc[np.random.default_rng(seed).permutation(len(labels))]
            summ, _ = fn(data, mask, perm, l1, dict(B.MODEL_PARAMS), folds=B.FOLDS, seed=seed)
            df[col] = pd.Series(summ["score"].astype(float).values, index=summ.index)
            df.to_csv(f)
            ok = df[col].notna()  # fold bị bỏ (modality hết đặc trưng sau L1) -> thiếu điểm; AUC trên bệnh nhân có điểm
            auc = roc_auc_score(df.loc[ok, "label"], df.loc[ok, col])
            print(f"   n_scored {int(ok.sum())}/{len(df)}", flush=True)
            print(f"[{scheme}] {name} seed {seed}: AUC {auc:.4f}", flush=True)
    print("XONG", scheme)


if __name__ == "__main__":
    main()
