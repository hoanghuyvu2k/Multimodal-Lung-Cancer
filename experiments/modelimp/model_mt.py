"""Học đa nhiệm đáp ứng + sống còn (multi-task) trên backbone chuyên gia theo modality.

Quy luật rút ra từ 21 lần thử trước: thứ gì thêm THAM SỐ đều thua; hai thứ duy nhất thắng (mã hoá
NLP, mẫu khuyết) đều thêm THÔNG TIN. Mô hình này không thêm tham số nào ở tầng hợp nhất — nó thêm
TÍN HIỆU GIÁM SÁT đang bị bỏ phí:

  Nhãn nhị phân gộp 185 bệnh nhân "không đáp ứng" làm một, nhưng nhóm đó rất không đồng nhất:
      SD (48 người)   PFS trung vị  7.05 tháng
      PD (92 người)   PFS trung vị  1.75 tháng
      khác (45 người) PFS trung vị ~1-2.5 tháng
  và PFS là biến liên tục với 209/247 biến cố quan sát được.

Mất mát:
    L = BCE(s, y)                                   nhiệm vụ chính: đáp ứng
      + λ_cox · L_Cox(a·s, PFS, biến cố)            hợp lý riêng phần Cox (Breslow); a > 0 học được
      + λ_ord · L_ord(s, mức BOR)                   xếp hạng theo cặp trên 3 mức: CR/PR < SD < PD/khác

Chỉ dùng PFS/BOR của bệnh nhân TRAIN; suy luận chỉ cần đặc trưng. Với λ = 0 mô hình trùng khít
`uniform_avg` (huấn luyện full-batch — tập train luôn < 256 nên bản gốc cũng chỉ có một batch).
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
import torch                                                      # noqa: E402
import torch.nn as nn                                             # noqa: E402
import torch.nn.functional as F                                   # noqa: E402

import baselines.model_uniform_avg as mua                         # noqa: E402

_Base = mua.MultiModalDynamicModelUncertainty
AUX = {"df": None}        # runner gán: DataFrame index = bệnh nhân, cột pfs, pfs_censor, bor


def bor_level(bor):
    """0 = CR/PR (đáp ứng) · 1 = SD · 2 = PD và các mã khác (PFS trung vị ~1-2.5 tháng)."""
    b = np.asarray(bor, dtype=float)
    return np.where(b <= 2, 0, np.where(b == 3, 1, 2))


def cox_loss(risk, time, event):
    """Âm log hợp lý riêng phần Cox (Breslow). risk cao = nguy cơ cao = PFS ngắn."""
    order = torch.argsort(time, descending=True)
    r, e = risk[order], event[order]
    log_cum = torch.logcumsumexp(r, dim=0)             # log Σ_{t_j ≥ t_i} exp(r_j)
    return -((r - log_cum) * e).sum() / e.sum().clamp(min=1.0)


def ordinal_pair_loss(score, level):
    """mean softplus(-(s_i - s_j)) trên mọi cặp có mức_i > mức_j (i phải có điểm cao hơn)."""
    diff = score.unsqueeze(1) - score.unsqueeze(0)
    worse = (level.unsqueeze(1) > level.unsqueeze(0)).float()
    return (F.softplus(-diff) * worse).sum() / worse.sum().clamp(min=1.0)


class ModelMT(_Base):
    def __init__(self, *a, lam_cox=0.0, lam_ord=0.0, **kw):
        super().__init__(*a, **kw)
        self.lam_cox, self.lam_ord = float(lam_cox), float(lam_ord)

    def fit(self, l_X_INPUTS, arr_MASK, vector_Y):
        self.train()
        px = vector_Y.index if hasattr(vector_Y, "index") else None
        l_X_INPUTS = copy.deepcopy(l_X_INPUTS)
        arr_MASK = copy.deepcopy(arr_MASK)
        vector_Y = copy.deepcopy(vector_Y)

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
        self.cox_scale = nn.Parameter(torch.tensor(0.5413))      # softplus(0.5413) = 1
        self.to(self.device)

        mask = torch.tensor(arr_MASK).float().to(self.device)
        targets = torch.tensor(np.asarray(vector_Y)).float().to(self.device)
        use_aux = (self.lam_cox > 0 or self.lam_ord > 0) and px is not None \
            and AUX["df"] is not None
        if use_aux:
            aux = AUX["df"].loc[px]
            t_pfs = torch.tensor(aux["pfs"].values, dtype=torch.float32, device=self.device)
            t_evt = torch.tensor(aux["pfs_censor"].values, dtype=torch.float32,
                                 device=self.device)
            t_lvl = torch.tensor(bor_level(aux["bor"].values), dtype=torch.float32,
                                 device=self.device)

        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=sum(targets == 0) / sum(targets == 1))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        for _ in range(self.epochs + 1):                 # full-batch: n_train < 256
            self.optimizer.zero_grad()
            output, _, _, attn_norm, _ = self.dyam(inputs, mask)
            loss = self.criterion(output, targets)
            if use_aux and self.lam_cox > 0:
                loss = loss + self.lam_cox * cox_loss(
                    F.softplus(self.cox_scale) * output, t_pfs, t_evt)
            if use_aux and self.lam_ord > 0:
                loss = loss + self.lam_ord * ordinal_pair_loss(output, t_lvl)
            l2 = self.dyam.get_l2_weight_sum()
            (loss + self.alpha * l2 + self.beta * attn_norm).backward()
            self.optimizer.step()

        output, _, _, _, _ = self.dyam(inputs, mask)
        return self.response_zscore(output, targets).detach().cpu().numpy()


def train_mt(data, mask, labels, l1_filter, model_params, folds=10, seed=42,
             lam_cox=0.0, lam_ord=0.0):
    params = {**model_params, "lam_cox": lam_cox, "lam_ord": lam_ord}
    saved = mua.MultiModalDynamicModelUncertainty
    mua.MultiModalDynamicModelUncertainty = ModelMT
    try:
        return mua.train_uniform_avg(data, mask, labels, l1_filter, params,
                                     folds=folds, seed=seed)
    finally:
        mua.MultiModalDynamicModelUncertainty = saved
