"""Step 3 — TabPFN late-fusion vs uniform_avg (BM1-4). Per-modality OOF TabPFN -> z-score -> trung bình có-mask."""
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from lung_helpers import get_training_data                      # noqa: E402
from common import benchmarks as B                              # noqa: E402
from common.data_setup import load_all                          # noqa: E402
from common.evaluate import save_result                         # noqa: E402
from tabpfn_exp.tabpfn_eval import get_tabpfn                    # noqa: E402

LOCKED = {"BM1": 0.7746, "BM2": 0.7665, "BM3": 0.7110, "BM4": 0.7191}
SEEDS = (42, 7, 123)
K = 40


def modality_oof(Xm, avail, y, idx, seed, folds=10):
    """OOF TabPFN z-score cho 1 modality (chỉ bệnh nhân có mặt)."""
    mk = get_tabpfn()
    out = np.full(len(idx), np.nan)
    pos = np.where(avail)[0]
    if len(pos) < 20:
        return out
    Xa, ya = np.nan_to_num(Xm[pos]), y[pos]
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(Xa, ya):
        if len(np.unique(ya[tr])) < 2:
            continue
        Xtr, Xte = Xa[tr], Xa[te]
        if K < Xtr.shape[1]:
            sel = SelectKBest(f_classif, k=K).fit(Xtr, ya[tr])
            Xtr, Xte = Xtr[:, sel.get_support()], Xte[:, sel.get_support()]
        sc = RobustScaler().fit(Xtr)
        clf = mk().fit(sc.transform(Xtr), ya[tr])
        p = clf.predict_proba(sc.transform(Xte))[:, 1]
        z = (p - p.mean()) / (p.std() + 1e-9)
        out[pos[te]] = z
    return out


def main():
    ctx = load_all()
    out = {}
    print(f"\n{'=' * 60}\nTabPFN late-fusion vs uniform_avg (BM1-4)\n{'=' * 60}")
    print(f"{'BM':<5}{'TabPFN-fusion':>18}{'uniform_avg (khoá)':>22}")
    for bm in B.BENCHMARKS:
        data, mask, labels = get_training_data(
            B.BENCHMARKS[bm]["modalities"], ctx.modality_dict, ctx.modality_MASK, ctx.df_outcomes)
        names = list(mask.columns)
        y = labels["label"].astype(float).values
        aucs = []
        for s in SEEDS:
            oof = np.column_stack([
                modality_oof(data[m].values, mask[names[m]].values.astype(bool), y, labels.index, s)
                for m in range(len(names))])
            score = np.nanmean(oof, axis=1)
            ok = ~np.isnan(score)
            aucs.append(roc_auc_score(y[ok], score[ok]))
        m_auc, sd = float(np.mean(aucs)), float(np.std(aucs, ddof=1))
        out[bm] = {"tabpfn_fusion": [m_auc, sd], "uniform_avg_locked": LOCKED[bm]}
        print(f"{bm:<5}{f'{m_auc:.4f}±{sd:.4f}':>18}{LOCKED[bm]:>22.4f}", flush=True)
    save_result("tabpfn_fusion", out)
    print("=" * 60)
    wins = sum(1 for bm in out if out[bm]["tabpfn_fusion"][0] > LOCKED[bm] + 0.005)
    print(f"TabPFN-fusion > uniform_avg (Δ>0.005) ở {wins}/4 BM")


if __name__ == "__main__":
    main()
