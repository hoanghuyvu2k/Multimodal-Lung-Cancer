"""GNN trên đồ thị bệnh nhân (kiểu MAGNET 2025 / MOGONET / MMGCN), thích ứng cho dữ liệu dự án.

Giữ nguyên các phần đã chứng minh được của pipeline: chọn đặc trưng radiomics L1 RIÊNG theo vị trí
trong từng fold, chuẩn hoá theo modality, và xử lý khuyết bằng mask.

Đồ thị: kNN (k=10) trên độ tương đồng hợp nhất = trung bình RBF trên các modality hai bệnh nhân
cùng có. Dựng trên ĐẶC TRƯNG của mọi bệnh nhân trong fold (train + test) — không dùng nhãn test.
Huấn luyện transductive: mất mát chỉ tính trên nút train, dự đoán nút test.

Bốn biến thể chạy trên CÙNG fold, CÙNG đặc trưng đã lọc:
  V0  expert-mean        chuyên gia vô hướng + trung bình có mask, KHÔNG đồ thị  (đối chứng trong vòng lặp)
  V1  expert+graph       V0 + nhánh GraphSAGE cộng vào, lớp cuối khởi tạo 0  -> bắt đầu đúng bằng V0
  V2  gnn-mean           encoder h=8 + trung bình có mask + 2 lớp GraphSAGE
  V3  gnn-attn (MAGNET)  encoder h=8 + attention bệnh nhân-modality có mask + 2 lớp GraphSAGE

    PYTHONIOENCODING=utf-8 python experiments/graphfuse/run_gnn.py
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
import torch.nn as nn                                             # noqa: E402
import torch.nn.functional as F                                   # noqa: E402
from scipy.spatial.distance import pdist, squareform              # noqa: E402
from sklearn.metrics import roc_auc_score                         # noqa: E402
from sklearn.model_selection import KFold                         # noqa: E402
from sklearn.preprocessing import RobustScaler                    # noqa: E402

from allcombo.run_paper_combos import PAPER_COMBOS, make_filter   # noqa: E402
from common import benchmarks as B                                # noqa: E402
from common.data_setup import load_all                            # noqa: E402
from common.evaluate import save_result                           # noqa: E402
from lung_helpers import get_training_data, l1_filter_features_list  # noqa: E402

COMBOS = ["Rad+IHC-G+Gen+PDL1", "Rad+IHC-G+Gen", "Rad+Gen", "IHC-G+Gen", "PDL1+Gen"]
BASE = {"Rad+IHC-G+Gen+PDL1": 0.7730, "Rad+IHC-G+Gen": 0.7510, "Rad+Gen": 0.7160,
        "IHC-G+Gen": 0.7184, "PDL1+Gen": 0.7191}
K_NN, HID, EPOCHS, LR, WD = 10, 8, 125, 0.01, 1e-3
VARIANTS = ["V0 expert-mean", "V1 expert+graph", "V2 gnn-mean", "V3 gnn-attn"]


def prep_fold(data, mask, labels, l1, tr, te):
    """Đặc trưng đã lọc + chuẩn hoá cho MỌI nút, mask, và ma trận kề kNN đã chuẩn hoá hàng."""
    mods = [df.copy(deep=True) for df in data]
    tr_px, te_px = labels.index[tr], labels.index[te]
    for pos, filt in l1.items():
        l1_filter_features_list(mods, filt["l1_selection_df"], labels, te_px, pos,
                                **filt["kwargs"])
    idx = tr_px.append(te_px)
    n, n_tr = len(idx), len(tr_px)
    Xs, keep = [], []
    S = np.zeros((n, n)); C = np.zeros((n, n))
    for i, df in enumerate(mods):
        if df.shape[1] == 0:
            continue
        m_all = mask.iloc[:, i].loc[idx].values.astype(bool)
        m_tr = m_all[:n_tr]
        if m_tr.sum() < 5:
            continue
        num = df.select_dtypes(include=[np.number]).loc[idx].values.astype(float)
        sc = RobustScaler().fit(num[:n_tr][m_tr])
        X = np.clip(np.nan_to_num(sc.transform(num)), -5, 5)
        X[~m_all] = 0.0
        Xs.append(X); keep.append(i)
        d = X.shape[1]
        D = squareform(pdist(X[m_all], "sqeuclidean")) / d
        med = np.median(D[np.triu_indices_from(D, 1)]) if m_all.sum() > 2 else 1.0
        ix = np.ix_(m_all, m_all)
        S[ix] += np.exp(-D / max(med, 1e-9)); C[ix] += 1
    S = np.where(C > 0, S / np.maximum(C, 1), 0.0)
    np.fill_diagonal(S, 0.0)
    A = np.zeros_like(S)
    nn_idx = np.argsort(-S, axis=1)[:, :K_NN]
    A[np.repeat(np.arange(n), K_NN), nn_idx.ravel()] = 1.0
    A = np.maximum(A, A.T)
    A = A / np.maximum(A.sum(1, keepdims=True), 1.0)
    M = mask.iloc[:, keep].loc[idx].values.astype(float)
    y = labels.loc[idx, "label"].values.astype(float)
    t = lambda a: torch.tensor(a, dtype=torch.float32)                # noqa: E731
    return [t(x) for x in Xs], t(M), t(A), t(y), n_tr, te_px


class Sage(nn.Module):
    """GraphSAGE gộp-trung-bình: h' = W_self h + W_nb (Â h)."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.s, self.n = nn.Linear(d_in, d_out), nn.Linear(d_in, d_out, bias=False)

    def forward(self, h, A):
        return self.s(h) + self.n(A @ h)


class GraphNet(nn.Module):
    def __init__(self, dims, variant):
        super().__init__()
        torch.manual_seed(42)
        self.v = variant
        N = len(dims)
        scalar = variant in ("V0 expert-mean", "V1 expert+graph")
        self.enc = nn.ModuleList([nn.Linear(d, 1 if scalar else HID) for d in dims])
        if variant == "V1 expert+graph":
            self.g1, self.g2 = Sage(2 * N, HID), Sage(HID, 1)
            for p in self.g2.parameters():                      # lớp cuối = 0 -> khởi đầu = V0
                nn.init.zeros_(p)
        if variant in ("V2 gnn-mean", "V3 gnn-attn"):
            self.g1, self.g2 = Sage(HID, HID), Sage(HID, 1)
        if variant == "V3 gnn-attn":
            self.att = nn.ModuleList([nn.Linear(HID, 1) for _ in dims])
        self.drop = nn.Dropout(0.3)

    def forward(self, Xs, M, A):
        if self.v in ("V0 expert-mean", "V1 expert+graph"):
            R = torch.cat([torch.tanh(e(x)) for e, x in zip(self.enc, Xs)], 1) * M
            out = R.sum(1) / M.sum(1).clamp(min=1.0)
            if self.v == "V1 expert+graph":
                h = self.drop(F.relu(self.g1(torch.cat([R, M], 1), A)))
                out = out + self.g2(h, A).squeeze(1)
            return out
        E = torch.stack([torch.tanh(e(x)) for e, x in zip(self.enc, Xs)], 1)     # n, N, HID
        if self.v == "V3 gnn-attn":
            sc = torch.cat([a(E[:, i]) for i, a in enumerate(self.att)], 1)
            w = torch.softmax(sc.masked_fill(M == 0, -1e9), 1) * M
        else:
            w = M / M.sum(1, keepdim=True).clamp(min=1.0)
        h = (E * w.unsqueeze(2)).sum(1)
        h = self.drop(F.relu(self.g1(h, A)))
        return self.g2(h, A).squeeze(1)


def fit_predict(Xs, M, A, y, n_tr, variant):
    net = GraphNet([x.shape[1] for x in Xs], variant)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    ytr = y[:n_tr]
    crit = nn.BCEWithLogitsLoss(pos_weight=(ytr == 0).sum() / (ytr == 1).sum().clamp(min=1))
    net.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        crit(net(Xs, M, A)[:n_tr], ytr).backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        return net(Xs, M, A)[n_tr:].numpy()


def main():
    torch.set_num_threads(2)
    ctx = load_all()
    out = {}
    print(f"{'tổ hợp':<22}" + "".join(f"{v:>22}" for v in VARIANTS) + f"{'uniform_avg':>14}")
    for label in COMBOS:
        mods, ftype = next((m, f) for lb, m, f in PAPER_COMBOS if lb == label)
        data, mask, labels = get_training_data(mods, ctx.modality_dict, ctx.modality_MASK,
                                               ctx.df_outcomes)
        l1 = make_filter(ftype, ctx)
        y = labels["label"].astype(float)
        n = len(labels)
        oofs = {v: {} for v in VARIANTS}
        for seed in B.SEEDS:
            order = np.random.default_rng(seed).permutation(n)
            col = {v: pd.Series(np.nan, index=labels.index) for v in VARIANTS}
            for tr_i, te_i in KFold(10, shuffle=True, random_state=0).split(order):
                Xs, M, A, yy, n_tr, te_px = prep_fold(data, mask, labels, l1,
                                                      order[tr_i], order[te_i])
                for v in VARIANTS:
                    col[v].loc[te_px] = fit_predict(Xs, M, A, yy, n_tr, v)
            for v in VARIANTS:
                oofs[v][seed] = col[v]
        row = {}
        for v in VARIANTS:
            aucs = [roc_auc_score(y, oofs[v][s]) for s in B.SEEDS]
            z = pd.DataFrame(oofs[v]).apply(lambda c: (c - c.mean()) / c.std()).mean(1)
            row[v] = {"mean": float(np.mean(aucs)), "sd": float(np.std(aucs, ddof=1)),
                      "per_seed": [float(a) for a in aucs],
                      "ensemble": float(roc_auc_score(y, z))}
        out[label] = row
        save_result("gnn_patient_graph", out)
        print(f"{label:<22}" + "".join(
            f"{row[v]['mean']:.4f}±{row[v]['sd']:.3f}".rjust(22) for v in VARIANTS)
            + f"{BASE[label]:>14.4f}", flush=True)
    print("XONG")


if __name__ == "__main__":
    main()
