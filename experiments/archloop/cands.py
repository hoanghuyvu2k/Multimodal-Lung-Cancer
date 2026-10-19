"""Vòng lặp tìm kiến trúc — đợt 1, nhóm 2: C8, C7, C15, C4.

Mỗi ứng viên có chế độ TẮT tái lập đúng baseline uniform_avg (kiểm bằng `harness.py verify`).

  (C8 res_mlp, C7 mixup, C15 pca đã thất bại ở sàng lọc và bị xoá — xem ledger.md)
  (C16 extpath đã thất bại ở sàng lọc và bị xoá)
  C18 pdl1fill  lấp PD-L1 thiếu (45 bệnh nhân) bằng điểm PD-L1 trong báo cáo lâm sàng
  C4  stab      chọn đặc trưng radiomics ỔN ĐỊNH: elastic-net trên 8 mẫu bootstrap bệnh nhân,
                giữ đặc trưng được chọn ở >= 5/8 lần
"""
import sys
from collections import Counter
from functools import partial
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
from sklearn.decomposition import PCA                             # noqa: E402
from sklearn.preprocessing import RobustScaler                    # noqa: E402
from torch.utils.data import DataLoader                           # noqa: E402

import baselines.model_uniform_avg as mua                         # noqa: E402
import lung_helpers as LH                                         # noqa: E402
from lung_helpers import MaskedMultiModalLoader                   # noqa: E402

_BaseMatrix = mua.AttentionMatrixUncertainty
_BaseModel = mua.MultiModalDynamicModelUncertainty


def _with(attr, value, fn, *a, **kw):
    """Chạy fn với mua.<attr> tạm thay bằng value."""
    saved = getattr(mua, attr)
    setattr(mua, attr, value)
    try:
        return fn(*a, **kw)
    finally:
        setattr(mua, attr, saved)


# ── C15 / C4: thay bước xử lý radiomics trong từng fold ────────────────────────
def _robust_features(train_df, kw):
    _, f_out = LH.select_outlier_stable_features(train_df, cutoff=kw.get("outlier_cutoff", 6))
    _, f_rob = LH.select_by_interlesion_variance(train_df, cutoff=kw.get("robustness_cutoff", 0.15))
    return sorted(set(f_out).intersection(set(f_rob)))


def train_with_filter(data, mask, labels, l1, params, folds=10, seed=42, filt=None):
    if filt is None:
        return mua.train_uniform_avg(data, mask, labels, l1, params, folds=folds, seed=seed)
    return _with("l1_filter_features_list", filt, mua.train_uniform_avg,
                 data, mask, labels, l1, params, folds=folds, seed=seed)


# ── C18: lấp PD-L1 thiếu bằng điểm PD-L1 trong báo cáo lâm sàng ────────────────
_CLIN = {}


def _clinical_pdl1():
    """Series index = bệnh nhân discovery, giá trị = −TPS/100 (cùng thang với cnl_pdl1_score)."""
    if "s" not in _CLIN:
        from common import data_setup as ds
        coh = pd.read_csv(f"{ds.BASE_DB_DIR}/final_cohort_listing.csv").set_index("main_index")
        c = LH.get_clinical_table_v2(
            path=f"{ds.BASE_DB_DIR}/18193MSKMINDProjectM-OmnibusInventory_DATA_2021-12-20_1540"
                 f"-WITH-TB-and-SCANNER.csv",
            main_index_col="dmp_pt_id", cohort=coh[coh["cohort"] == "discovery"])
        _CLIN["s"] = -pd.to_numeric(c["clinical_pdl1_score"], errors="coerce") / 100.0
    return _CLIN["s"]


def train_pdl1fill(data, mask, labels, l1, params, folds=10, seed=42, fill=True):
    """Kênh PD-L1: dùng điểm chấm lại (201/247) khi có, nếu thiếu thì lấy điểm trong báo cáo lâm sàng
    (có ở 246/247) -> độ phủ của chỉ dấu mạnh nhất tăng từ 81% lên ~100%."""
    cols = list(mask.columns)
    if fill and "cnl_pdl1_score" in cols:
        i = cols.index("cnl_pdl1_score")
        cl = _clinical_pdl1().reindex(data[i].index)
        have = mask["cnl_pdl1_score"].astype(bool)
        df = data[i].copy()
        use = (~have.values) & cl.notna().values
        df.iloc[use, 0] = cl.values[use]
        data = list(data); data[i] = df
        mask = mask.copy(); mask.loc[mask.index[use], "cnl_pdl1_score"] = True
    return mua.train_uniform_avg(data, mask, labels, l1, params, folds=folds, seed=seed)


# ── ĐỢT 2 (D1–D5, D9, D10 đã thất bại và bị xoá — xem ledger.md) ───────────────
PATH_CH = ("path_ihc_glcm", "path_ihc_pdl1")


class AuxMatrix(mua.AttentionMatrixUncertainty):
    """D6 — chuyên gia pathology có lớp ẩn 4 nút, lớp ẩn đó đồng thời phải DỰ ĐOÁN điểm PD-L1 lâm sàng
    (nhiệm vụ phụ dự đoán chéo modality — "clinical prediction", arXiv 2304.00379)."""

    def __init__(self, *a, hid_idx=None, hidden=4, **kw):
        super().__init__(*a, **kw)
        self.hid_idx, self.hidden = hid_idx, hidden

    def setup_matrix(self):
        super().setup_matrix()
        if self.hid_idx is not None:
            d = self.input_channel_shapes[self.hid_idx]
            self.enc = nn.Linear(d, self.hidden)
            self.head = nn.Linear(self.hidden, 1)
            self.aux = nn.Linear(self.hidden, 1)
            nn.init.zeros_(self.head.bias)

    def hidden_rep(self, x):
        return torch.tanh(self.enc(x))

    def forward(self, inputs, mask):
        zs = []
        for i, x in enumerate(inputs):
            if i == self.hid_idx:
                zs.append(self.head(self.hidden_rep(x)))
            else:
                zs.append(self.l_risk_linears[i](x))
        z = torch.cat(zs, axis=1)
        r = mask * self.tanh(z)
        attn = torch.nn.functional.normalize(mask, p=1, dim=1)
        total = torch.sum(r * attn, dim=1)
        return total, r, attn, attn.norm(p=2, dim=1).mean(), z.norm(p=2, dim=1).mean()

    def get_l2_weight_sum(self):
        ws = [m.weight.norm(p=2) for i, m in enumerate(self.l_risk_linears) if i != self.hid_idx]
        if self.hid_idx is not None:
            ws += [self.enc.weight.norm(p=2), self.head.weight.norm(p=2)]
        return torch.stack(ws).sum()


class ModelAux(_BaseModel):
    def __init__(self, *a, hid_idx=None, aux_lam=0.5, aux_shuffle=False, **kw):
        super().__init__(*a, **kw)
        self.hid_idx, self.aux_lam, self.aux_shuffle = hid_idx, float(aux_lam), bool(aux_shuffle)

    def get_coefs(self, modality_list, modality_mask):
        return pd.DataFrame()

    def fit(self, l_X_INPUTS, arr_MASK, vector_Y):
        self.train()
        px = vector_Y.index
        l_X_INPUTS = copy.deepcopy(l_X_INPUTS)
        self.dyam = AuxMatrix(attention_gate_enabled=self.attention_gate_enabled,
                              init_seed=self.init_seed, logvar_clamp=self.logvar_clamp,
                              hid_idx=self.hid_idx)
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
        tgt = _clinical_pdl1().reindex(px)
        if self.aux_shuffle:                                   # đối chứng: đích phụ bị xáo -> vô nghĩa sinh học
            v = tgt.values.copy()
            ok = ~np.isnan(v)
            v[ok] = np.random.default_rng(0).permutation(v[ok])
            tgt = pd.Series(v, index=tgt.index)
        t_aux = torch.tensor(np.nan_to_num(tgt.values), dtype=torch.float32, device=self.device)
        rows = torch.tensor(tgt.notna().values, device=self.device) & (mask[:, self.hid_idx] > 0)
        for _ in range(self.epochs + 1):                       # full-batch (n_train < 256)
            self.optimizer.zero_grad()
            output, _, _, attn_norm, _ = self.dyam(inputs, mask)
            loss = self.criterion(output, targets)
            if self.aux_lam > 0 and rows.sum() > 5:
                pred = self.dyam.aux(self.dyam.hidden_rep(inputs[self.hid_idx])).squeeze(1)
                loss = loss + self.aux_lam * torch.mean((pred[rows] - t_aux[rows]) ** 2)
            (loss + self.alpha * self.dyam.get_l2_weight_sum() + self.beta * attn_norm).backward()
            self.optimizer.step()
        output, _, _, _, _ = self.dyam(inputs, mask)
        return self.response_zscore(output, targets).detach().cpu().numpy()


def train_aux(data, mask, labels, l1, params, folds=10, seed=42, aux_lam=0.5, max_dim=None,
              aux_shuffle=False):
    """aux_lam=0.5: có nhiệm vụ phụ; aux_lam=0: chỉ lớp ẩn (đối chứng).
    max_dim: chỉ áp khi chuyên gia pathology có <= max_dim đặc trưng (D6b: 32 -> chỉ IHC-A, 18 chiều)."""
    cols = list(mask.columns)
    idx = next((i for i, c in enumerate(cols) if c in PATH_CH), None)
    if idx is not None and max_dim is not None and data[idx].shape[1] > max_dim:
        idx = None
    if idx is None:
        return mua.train_uniform_avg(data, mask, labels, l1, params, folds=folds, seed=seed)
    p = {**params, "hid_idx": idx, "aux_lam": aux_lam, "aux_shuffle": aux_shuffle}
    return _with("MultiModalDynamicModelUncertainty", ModelAux, mua.train_uniform_avg,
                 data, mask, labels, l1, p, folds=folds, seed=seed)


# (D7 tự chưng cất đã thất bại ở sàng lọc và bị xoá — xem ledger.md)


CANDIDATES = {
    "c18_pdl1fill": partial(train_pdl1fill, fill=True),
    "d6b_auxlow": partial(train_aux, aux_lam=0.5, max_dim=32),
    "d6b_shuf": partial(train_aux, aux_lam=0.5, max_dim=32, aux_shuffle=True),
    "d6b_hid": partial(train_aux, aux_lam=0.0, max_dim=32),
}
OFF = {                                    # chế độ tắt: phải cho đúng 0.7927 ở seed 42 BM1
    "c18_off": partial(train_pdl1fill, fill=False),
}
