"""Tăng cường dữ liệu radiomics bằng các bản NHIỄU LOẠN phân vùng có sẵn trong bộ dữ liệu.

Thông tin phía ĐẦU VÀO chưa được khai thác: mỗi tổn thương có 10 bản đặc trưng radiomics tính lại
dưới nhiễu loạn phân vùng (`job_tag = 'pertubation-radiomics'`). Pipeline hiện chỉ dùng chúng để LỌC
đặc trưng kém bền, rồi bỏ đi. Ở đây dùng chúng làm tăng cường dữ liệu cho đúng nút thắt đã xác định
(bộ mã hoá radiomics):

  Huấn luyện: mỗi epoch, đặc trưng radiomics của mỗi bệnh nhân train được rút ngẫu nhiên từ
              {bản gốc + 10 bản nhiễu loạn}. Số bước tối ưu KHÔNG đổi (vẫn full-batch, 125 epoch)
              nên không biến thành "huấn luyện nhiều hơn" — thứ đã biết là gây overfit.
  Suy luận:   (a) chỉ bản gốc;  (b) TTA — trung bình đầu ra trên 11 bản.

Đã kiểm: mỗi hàng modality khớp chính xác một tổn thương; mỗi tổn thương có đúng 10 bản nhiễu loạn;
độ lệch nhiễu loạn ≈ 0.08 (PC) và 0.22 (LN) lần độ lệch chuẩn giữa các bệnh nhân.

    PYTHONIOENCODING=utf-8 python experiments/modelimp/run_pertaug.py [--smoke]
"""
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
_CODE = _EXP.parent
for _p in (str(_CODE), str(_EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import copy                                                       # noqa: E402

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
import torch                                                      # noqa: E402
import torch.nn as nn                                             # noqa: E402
from sklearn.metrics import roc_auc_score                         # noqa: E402
from sklearn.model_selection import KFold                         # noqa: E402

from modelimp import run_all as R                                 # noqa: E402  (vá scipy)
import baselines.model_uniform_avg as mua                         # noqa: E402
from common import benchmarks as B                                # noqa: E402
from common.data_setup import load_all                            # noqa: E402
from common.evaluate import load_result, save_result              # noqa: E402
from lung_helpers import l1_filter_features_list, set_global_seed  # noqa: E402

SITES = {"rad_lesion_pc": 0, "rad_lesion_pl": 1, "rad_lesion_ln": 2}
N_PERT = 10
BASE = {"Rad+IHC-G+Gen+PDL1": (0.7730, 0.7838), "Rad+IHC-G+Gen": (0.7510, 0.7631),
        "Rad+Gen": (0.7160, 0.7291), "Rad": (0.6661, 0.6925)}


def build_pert(ctx):
    """site -> (mảng [n_bệnh_nhân, 10, n_đặc_trưng] khớp index/cột của modality, NaN nếu thiếu)."""
    out = {}
    for name, pos in SITES.items():
        mod = ctx.modality_dict[name]
        sel = ctx.rad_filters[pos]["l1_selection_df"]
        feats = list(mod.columns)
        arr = np.full((len(mod), N_PERT, len(feats)), np.nan, dtype=np.float32)
        filt = sel[sel.job_tag == "filtered-radiomics"]
        pert = sel[sel.job_tag == "pertubation-radiomics"]
        have = ctx.modality_MASK[name].reindex(mod.index).fillna(False).values.astype(bool)
        for k, p in enumerate(mod.index):
            if not have[k] or p not in filt.index:
                continue
            f = filt.loc[[p]]
            d = np.abs(f[feats].values.astype(float) - mod.loc[p].values.astype(float))
            j = int(np.nanargmin(np.nanmean(d, axis=1)))
            if np.nanmax(d[j]) > 1e-6:
                continue                                  # không khớp chính xác -> bỏ tăng cường
            rows = pert.loc[[p]] if p in pert.index else pert.iloc[0:0]
            rows = rows[rows.lesion_index == f.iloc[j]["lesion_index"]]
            if len(rows) >= N_PERT:
                arr[k] = rows[feats].values.astype(np.float32)[:N_PERT]
        out[name] = (arr, {c: i for i, c in enumerate(feats)},
                     {p: i for i, p in enumerate(mod.index)})
    return out


class ModelPert(mua.MultiModalDynamicModelUncertainty):
    """uniform_avg + rút ngẫu nhiên bản nhiễu loạn radiomics mỗi epoch."""

    def __init__(self, *a, aug=True, **kw):
        super().__init__(*a, **kw)
        self.aug = aug
        self.pert_train = {}                      # kênh -> ndarray [n_train, 10, d] (đặc trưng thô)

    def _pert_tensor(self, ch, P):
        n, r, d = P.shape
        bad = np.isnan(P).any(axis=(1, 2))        # bệnh nhân không có bản nhiễu loạn
        X = np.nan_to_num(self.d_scalers[ch].transform(np.nan_to_num(P.reshape(-1, d))))
        return (torch.tensor(X.reshape(n, r, d)).float().to(self.device),
                torch.tensor(~bad).to(self.device))

    def fit(self, l_X_INPUTS, arr_MASK, vector_Y):
        self.train()
        l_X_INPUTS = copy.deepcopy(l_X_INPUTS)
        self.dyam = mua.AttentionMatrixUncertainty(
            attention_gate_enabled=self.attention_gate_enabled, init_seed=self.init_seed,
            logvar_clamp=self.logvar_clamp)
        self.d_scalers = {}
        prepared = self._prepare(l_X_INPUTS, fitting=True)
        inputs = []
        for tensor, raw in prepared:
            inputs.append(tensor)
            self.dyam.add_channel(raw)
        self.dyam.setup_matrix()
        self.to(self.device)

        mask = torch.tensor(copy.deepcopy(arr_MASK)).float().to(self.device)
        targets = torch.tensor(np.asarray(vector_Y)).float().to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=sum(targets == 0) / sum(targets == 1))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        pert = ({ch: self._pert_tensor(ch, P) for ch, P in self.pert_train.items()}
                if self.aug else {})
        gen = torch.Generator().manual_seed(0)
        n = len(targets)

        for _ in range(self.epochs + 1):                         # full-batch như bản gốc
            ins = list(inputs)
            for ch, (P, ok) in pert.items():
                r = torch.randint(0, N_PERT + 1, (n,), generator=gen)       # 0 = bản gốc
                use = (r > 0) & ok
                x = inputs[ch].clone()
                x[use] = P[use, r[use] - 1]
                ins[ch] = x
            self.optimizer.zero_grad()
            output, _, _, attn_norm, _ = self.dyam(ins, mask)
            loss = self.criterion(output, targets)
            (loss + self.alpha * self.dyam.get_l2_weight_sum()
             + self.beta * attn_norm).backward()
            self.optimizer.step()
        output, _, _, _, _ = self.dyam(inputs, mask)
        return self.response_zscore(output, targets).detach().cpu().numpy()

    def predict_tta(self, l_X_INPUTS, arr_MASK, pert_test):
        """Trung bình đầu ra trên bản gốc + 10 bản nhiễu loạn."""
        self.eval()
        prepared = self._prepare(copy.deepcopy(l_X_INPUTS), fitting=False)
        inputs = [t for t, _ in prepared]
        mask = torch.tensor(copy.deepcopy(arr_MASK)).float().to(self.device)
        pert = {ch: self._pert_tensor(ch, P) for ch, P in pert_test.items()}
        with torch.no_grad():
            outs = [self.dyam(inputs, mask)[0]]
            for r in range(N_PERT):
                ins = list(inputs)
                for ch, (P, ok) in pert.items():
                    x = inputs[ch].clone()
                    x[ok] = P[ok, r]
                    ins[ch] = x
                outs.append(self.dyam(ins, mask)[0])
        return torch.stack(outs).mean(0).cpu().numpy()


def cv_once(data, mask, labels, l1, pert, seed, aug=True, extra=None):
    """Một lượt 10-fold đúng như train_uniform_avg; trả (điểm thường, điểm TTA)."""
    set_global_seed(seed)
    params = {k: v for k, v in B.MODEL_PARAMS.items() if k != "cross_modality_enabled"}
    params.update(extra or {})
    cols = list(mask.columns)
    plain = pd.Series(np.nan, index=labels.index)
    tta = pd.Series(np.nan, index=labels.index)
    kf = KFold(n_splits=B.FOLDS, random_state=0, shuffle=True)
    for tr, te in kf.split(labels.index):
        mods = [df.copy(deep=True) for df in data]
        tr_px, te_px = labels.index[tr], labels.index[te]
        for pos, filt in l1.items():
            l1_filter_features_list(mods, filt["l1_selection_df"], labels, te_px, pos,
                                    **filt["kwargs"])
        if any(len(df.columns) == 0 for df in mods):
            continue
        clf = ModelPert(aug=aug, **params)
        p_tr, p_te = {}, {}
        for ch, name in enumerate(cols):
            if name in pert:
                arr, cidx, pidx = pert[name]
                ci = [cidx[c] for c in mods[ch].columns]
                p_tr[ch] = arr[[pidx[p] for p in tr_px]][:, :, ci]
                p_te[ch] = arr[[pidx[p] for p in te_px]][:, :, ci]
        clf.pert_train = p_tr
        tr_in = [df.loc[tr_px].values for df in mods]
        te_in = [df.loc[te_px].values for df in mods]
        tr_mask = mask.loc[tr_px].astype(int).values
        te_mask = mask.loc[te_px].astype(int).values
        clf.fit(tr_in, tr_mask, labels.loc[tr_px, "label"])
        plain.loc[te_px] = clf.predict_proba(te_in, te_mask)
        tta.loc[te_px] = clf.predict_tta(te_in, te_mask, p_te)
    return plain.dropna(), tta.dropna()


def summarize(cols, y):
    aucs = [roc_auc_score(y.loc[s.index], s) for s in cols.values()]
    z = pd.DataFrame(cols).apply(lambda c: (c - c.mean()) / c.std()).mean(1, skipna=True).dropna()
    return {"mean": float(np.mean(aucs)), "sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
            "ensemble": float(roc_auc_score(y.loc[z.index], z)),
            "per_seed": [float(a) for a in aucs]}


def main(argv):
    smoke = "--smoke" in argv
    ctx = load_all()
    pert = build_pert(ctx)
    n_ok = {k: int((~np.isnan(v[0]).any(axis=(1, 2))).sum()) for k, v in pert.items()}
    print(f"bệnh nhân có đủ 10 bản nhiễu loạn: {n_ok}")
    combos = ["Rad+IHC-G+Gen+PDL1"] if smoke else list(BASE)
    seeds = B.SEEDS[:1] if smoke else B.SEEDS
    try:
        payload = load_result("pertaug")
    except Exception:
        payload = {"combos": {}}
    print(f"{'tổ hợp':<22}{'biến thể':<22}{'AUC':>15}{'Δ':>9}{'ensemble':>10}{'Δ':>9}")
    for label in combos:
        data, mask, labels, l1 = R.combo(ctx, label)
        y = labels["label"].astype(float)
        b_mean, b_ens = BASE[label]
        print(f"{label:<22}{'uniform_avg':<22}{b_mean:>15.4f}{0:>+9.4f}{b_ens:>10.4f}{0:>+9.4f}")
        runs = {"aug": {}, "aug+TTA": {}}
        if smoke:
            runs["tắt aug (=nền)"] = {}
        for seed in seeds:
            lab = labels.iloc[np.random.default_rng(seed).permutation(len(labels))]
            runs["aug"][seed], runs["aug+TTA"][seed] = cv_once(data, mask, lab, l1, pert, seed)
            if smoke:
                runs["tắt aug (=nền)"][seed], _ = cv_once(data, mask, lab, l1, pert, seed,
                                                          aug=False)
        for name, cols in runs.items():
            r = summarize(cols, y)
            if not smoke:
                payload["combos"].setdefault(label, {})[name] = r
                save_result("pertaug", payload)
            print(f"{'':<22}{name:<22}{r['mean']:.4f}±{r['sd']:.3f}{r['mean'] - b_mean:>+9.4f}"
                  f"{r['ensemble']:>10.4f}{r['ensemble'] - b_ens:>+9.4f}", flush=True)
    print("XONG")


if __name__ == "__main__":
    torch.set_num_threads(2)
    main(sys.argv)
