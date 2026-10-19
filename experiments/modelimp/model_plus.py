"""Hướng 3: loss tối ưu trực tiếp AUC (pairwise) + modality dropout.

Hai thay đổi ở TẦNG HUẤN LUYỆN của backbone `uniform_avg`, không đụng đặc trưng đầu vào:

  loss='pairwise'      thay BCE bằng mất mát trên từng cặp (dương, âm):
                       mean softplus(-(s⁺ − s⁻)). AUC chính là xác suất xếp đúng một cặp,
                       nên đây là hàm thay thế trực tiếp cho chỉ số đánh giá. Với 62 ca đáp ứng
                       và 185 ca không đáp ứng, 247 bệnh nhân thành 11.470 cặp.

  modality_dropout=p   mỗi batch, che ngẫu nhiên các modality ĐANG CÓ với xác suất p
                       (luôn giữ ít nhất 1 nhánh). Cohort có 23 mẫu khuyết khác nhau nhưng
                       mô hình chưa từng được huấn luyện có chủ đích cho việc thiếu nhánh.
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
from torch.utils.data import DataLoader                           # noqa: E402

import baselines.model_uniform_avg as mua                         # noqa: E402
from lung_helpers import MaskedMultiModalLoader                   # noqa: E402

_Base = mua.MultiModalDynamicModelUncertainty


class ModelPlus(_Base):
    def __init__(self, *a, loss="bce", modality_dropout=0.0, **kw):
        super().__init__(*a, **kw)
        self.loss_kind = loss
        self.modality_dropout = float(modality_dropout)

    @staticmethod
    def _pairwise_loss(scores, labels):
        """mean softplus(-(s⁺ − s⁻)) trên mọi cặp (label=1, label=0) trong batch."""
        pos, neg = scores[labels == 1], scores[labels == 0]
        if pos.numel() == 0 or neg.numel() == 0:
            return scores.sum() * 0.0
        return F.softplus(-(pos.unsqueeze(1) - neg.unsqueeze(0))).mean()

    def _drop_mask(self, mask):
        """Che ngẫu nhiên các nhánh đang có, luôn chừa lại ít nhất một."""
        if self.modality_dropout <= 0:
            return mask
        keep = (torch.rand_like(mask) > self.modality_dropout).float()
        new = mask * keep
        empty = new.sum(dim=1) == 0
        if empty.any():                      # hàng nào bị che sạch thì trả lại mask gốc
            new[empty] = mask[empty]
        return new

    def fit(self, l_X_INPUTS, arr_MASK, vector_Y):
        self.train()
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
        self.to(self.device)

        mask = torch.tensor(arr_MASK).float().to(self.device)
        targets = torch.tensor(np.asarray(vector_Y)).float().to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=sum(targets == 0) / sum(targets == 1))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        loader = DataLoader(MaskedMultiModalLoader(inputs, mask, targets),
                            batch_size=256, shuffle=True, drop_last=False)
        n_batch = len(loader)

        for _ in range(self.epochs + 1):
            for b_inputs, b_mask, b_labels in loader:
                self.optimizer.zero_grad()
                b_mask = self._drop_mask(b_mask)
                output, _, _, attn_norm, _ = self.dyam(b_inputs, b_mask)
                loss = (self._pairwise_loss(output, b_labels)
                        if self.loss_kind == "pairwise"
                        else self.criterion(output, b_labels))
                l2 = self.dyam.get_l2_weight_sum()
                ((loss + self.alpha * l2 + self.beta * attn_norm) / n_batch).backward()
                self.optimizer.step()

        output, _, _, _, _ = self.dyam(inputs, mask)
        return self.response_zscore(output, targets).detach().cpu().numpy()


def train_plus(*args, **kwargs):
    """Như train_uniform_avg nhưng dùng ModelPlus (nhận thêm loss / modality_dropout)."""
    saved = mua.MultiModalDynamicModelUncertainty
    mua.MultiModalDynamicModelUncertainty = ModelPlus
    try:
        return mua.train_uniform_avg(*args, **kwargs)
    finally:
        mua.MultiModalDynamicModelUncertainty = saved
