"""Sàng lọc họ mô hình ĐỒ THỊ BỆNH NHÂN (patient similarity network) — chưa từng thử trong dự án.

Mọi mô hình trước đây đều là hàm tham số áp lên TỪNG bệnh nhân. Họ này khác bản chất: dự đoán một
bệnh nhân từ những bệnh nhân GIỐNG họ (SNF, MOGONET, MMGCN, MAGNET 2025). Trước khi dựng GNN đầy đủ,
kiểm tra rẻ xem thiên kiến quy nạp này có tín hiệu trên dữ liệu của ta không:

  1. Với mỗi modality: độ tương đồng RBF giữa hai bệnh nhân CÙNG CÓ modality đó (đặc trưng đã chuẩn
     hoá trên fold train; radiomics qua đúng bộ lọc L1 theo vị trí của pipeline).
  2. Hợp nhất: trung bình các độ tương đồng trên những modality mà hai bệnh nhân cùng có
     (khuyết modality được xử lý tự nhiên, không cần nội suy).
  3. Dự đoán: trung bình nhãn của bệnh nhân train, có trọng số là độ tương đồng (Nadaraya-Watson).

Báo cáo: (a) AUC của riêng mô hình đồ thị; (b) AUC khi cộng điểm đồ thị vào điểm `uniform_avg`
đã lưu (z-score, λ cố định trước 0.5 và 1.0 — không dò) để xem thông tin có BỔ SUNG nhau không.

    PYTHONIOENCODING=utf-8 python experiments/graphfuse/run_psn.py
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
from scipy.spatial.distance import cdist, pdist                   # noqa: E402
from sklearn.metrics import roc_auc_score                         # noqa: E402
from sklearn.model_selection import KFold                         # noqa: E402
from sklearn.preprocessing import RobustScaler                    # noqa: E402

from allcombo.run_paper_combos import PAPER_COMBOS, make_filter   # noqa: E402
from common import benchmarks as B                                # noqa: E402
from common.data_setup import load_all                            # noqa: E402
from common.evaluate import save_result                           # noqa: E402
from lung_helpers import get_training_data, l1_filter_features_list  # noqa: E402

COMBOS = ["Rad+IHC-G+Gen+PDL1", "Rad+IHC-G+Gen", "Rad+Gen", "IHC-G+Gen", "PDL1+Gen"]
SCORES = _CODE / "experiments" / "results" / "scores"


def fold_nw(data, mask, labels, l1, tr, te):
    mods = [df.copy(deep=True) for df in data]
    tr_px, te_px = labels.index[tr], labels.index[te]
    for pos, filt in l1.items():
        l1_filter_features_list(mods, filt["l1_selection_df"], labels, te_px, pos,
                                **filt["kwargs"])
    S = np.zeros((len(te), len(tr))); C = np.zeros_like(S)
    for i, df in enumerate(mods):
        if df.shape[1] == 0:
            continue
        mtr = mask.iloc[:, i].loc[tr_px].values.astype(bool)
        mte = mask.iloc[:, i].loc[te_px].values.astype(bool)
        if mtr.sum() < 5 or mte.sum() == 0:
            continue
        num = df.select_dtypes(include=[np.number])
        sc = RobustScaler().fit(num.loc[tr_px].values[mtr])
        f = lambda X: np.clip(np.nan_to_num(sc.transform(X)), -5, 5)      # noqa: E731
        Xtr, Xte = f(num.loc[tr_px].values[mtr]), f(num.loc[te_px].values[mte])
        d = Xtr.shape[1]
        med = np.median(pdist(Xtr, "sqeuclidean")) / d
        K = np.exp(-(cdist(Xte, Xtr, "sqeuclidean") / d) / max(med, 1e-9))
        ix = np.ix_(mte, mtr)
        S[ix] += K; C[ix] += 1
    S = np.where(C > 0, S / np.maximum(C, 1), 0.0)
    ytr = labels.loc[tr_px, "label"].values.astype(float)
    return (S @ ytr) / np.maximum(S.sum(1), 1e-9)


def z(s):
    return (s - s.mean()) / s.std()


def main():
    ctx = load_all()
    out = {}
    print(f"{'tổ hợp':<22}{'uniform':>10}{'đồ thị':>14}{'uni+0.5·đt':>13}{'uni+1.0·đt':>13}")
    for label in COMBOS:
        mods, ftype = next((m, f) for lb, m, f in PAPER_COMBOS if lb == label)
        data, mask, labels = get_training_data(mods, ctx.modality_dict, ctx.modality_MASK,
                                               ctx.df_outcomes)
        l1 = make_filter(ftype, ctx)
        y = labels["label"].astype(float)
        slug = label.replace("+", "_").replace("-", "").replace(" ", "")
        uni = pd.read_csv(SCORES / f"uni_{slug}__PAPER.csv", index_col=0)
        res = {"uni": [], "graph": [], "c05": [], "c10": []}
        n = len(labels)
        for seed in B.SEEDS:
            order = np.random.default_rng(seed).permutation(n)
            oof = pd.Series(np.nan, index=labels.index)
            for tr_i, te_i in KFold(10, shuffle=True, random_state=0).split(order):
                tr, te = order[tr_i], order[te_i]
                oof.iloc[te] = fold_nw(data, mask, labels, l1, tr, te)
            u = uni[f"seed_{seed}"].reindex(labels.index)
            ok = u.notna() & oof.notna()
            yy = y[ok]
            res["uni"].append(roc_auc_score(yy, u[ok]))
            res["graph"].append(roc_auc_score(yy, oof[ok]))
            res["c05"].append(roc_auc_score(yy, z(u[ok]) + 0.5 * z(oof[ok])))
            res["c10"].append(roc_auc_score(yy, z(u[ok]) + 1.0 * z(oof[ok])))
        out[label] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                          "per_seed": [float(x) for x in v]} for k, v in res.items()}
        save_result("psn_screen", out)
        m = lambda k: f"{np.mean(res[k]):.4f}"                              # noqa: E731
        print(f"{label:<22}{m('uni'):>10}{m('graph'):>14}{m('c05'):>13}{m('c10'):>13}",
              flush=True)
    print("XONG")


if __name__ == "__main__":
    main()
