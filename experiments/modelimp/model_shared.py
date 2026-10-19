"""Mô hình đề xuất: SHARE-Rad — chia sẻ bộ mã hoá radiomics + chế độ huấn luyện theo modality.

Thiết kế rút thẳng từ những gì các thí nghiệm trước chỉ ra:

  (1) Cơ chế hợp nhất không quan trọng (7 cơ chế cùng nằm trong 0.75-0.77; trung bình cộng tốt
      nhất) -> GIỮ trung bình có mask, không attention.
  (2) Nút thắt là bộ mã hoá radiomics (TabPFN / cooperative learning / XGBoost đều chỉ sụp ở tổ
      hợp có radiomics). Hiện mỗi vị trí tổn thương có MỘT ĐẦU RIÊNG học trên rất ít dữ liệu:
      PC 163, LN 67, PL chỉ 21 bệnh nhân — dù cả ba dùng CÙNG 1688 đặc trưng radiomics.
      -> CHIA SẺ TRỌNG SỐ: một vector trọng số chung cho PC/PL/LN, mỗi vị trí giữ bias riêng
         và chuẩn hoá riêng. Đầu radiomics được học trên toàn bộ ~250 tổn thương thay vì
         163/21/67. Chọn đặc trưng L1 cũng làm một lần trên tổn thương gộp (333 thay vì 204/33/96).
  (3) Modality ít chiều (PD-L1, genomics, pathology) hợp với huấn luyện nhẹ tay hơn, radiomics
      thì không (cấu hình nhẹ: PDL1+Gen +0.033, IHC-G+Gen +0.023, nhưng BM1 -0.004).
      -> CHẾ ĐỘ THEO MODALITY: lr và alpha riêng cho từng nhóm kênh trong cùng một lần huấn luyện.
  (4) Ensemble theo seed (+0.009, 19/21 tổ hợp) -> áp ở tầng đánh giá.
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
from torch.utils.data import DataLoader                           # noqa: E402

import baselines.model_uniform_avg as mua                         # noqa: E402
from lung_helpers import MaskedMultiModalLoader                   # noqa: E402

RAD_SITES = ("rad_lesion_pc", "rad_lesion_pl", "rad_lesion_ln")
_Base = mua.MultiModalDynamicModelUncertainty


class SharedMatrix(mua.AttentionMatrixUncertainty):
    """Như bản gốc, nhưng các kênh trong `shared` dùng CHUNG một vector trọng số (bias riêng)."""

    def __init__(self, *a, shared=(), **kw):
        super().__init__(*a, **kw)
        self.shared = list(shared)

    def setup_matrix(self):
        super().setup_matrix()
        if len(self.shared) >= 2:
            dims = {self.input_channel_shapes[i] for i in self.shared}
            if len(dims) == 1:                          # chỉ buộc được khi cùng số đặc trưng
                head = self.l_risk_linears[self.shared[0]]
                for i in self.shared[1:]:
                    self.l_risk_linears[i].weight = head.weight
                self.tied = True
                return
        self.tied = False

    def unique_risk_weights(self):
        """[(tham số weight, chỉ số kênh đại diện)] — weight đã buộc chỉ tính một lần."""
        seen, out = set(), []
        for i, lin in enumerate(self.l_risk_linears):
            if id(lin.weight) not in seen:
                seen.add(id(lin.weight))
                out.append((lin.weight, i))
        return out


class ModelShared(_Base):
    def __init__(self, *a, shared_channels=(), light_channels=(),
                 light_lr=None, light_alpha=None, **kw):
        super().__init__(*a, **kw)
        self.shared_channels = list(shared_channels)
        self.light_channels = set(light_channels)
        self.light_lr = light_lr
        self.light_alpha = light_alpha

    def _alpha_of(self, ch):
        if ch in self.light_channels and self.light_alpha is not None:
            return self.light_alpha
        return self.alpha

    def _lr_of(self, ch):
        if ch in self.light_channels and self.light_lr is not None:
            return self.light_lr
        return self.lr

    def fit(self, l_X_INPUTS, arr_MASK, vector_Y):
        self.train()
        l_X_INPUTS = copy.deepcopy(l_X_INPUTS)
        arr_MASK = copy.deepcopy(arr_MASK)
        vector_Y = copy.deepcopy(vector_Y)

        self.dyam = SharedMatrix(
            attention_gate_enabled=self.attention_gate_enabled, init_seed=self.init_seed,
            logvar_clamp=self.logvar_clamp, shared=self.shared_channels)
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

        # nhóm tham số: mỗi kênh một lr riêng; weight đã buộc chỉ vào optimizer một lần
        uniq = self.dyam.unique_risk_weights()
        groups = [{"params": [w], "lr": self._lr_of(ch)} for w, ch in uniq]
        for i, lin in enumerate(self.dyam.l_risk_linears):
            groups.append({"params": [lin.bias], "lr": self._lr_of(i)})
        groups.append({"params": list(self.dyam.l_logvar_linears.parameters()), "lr": self.lr})
        self.optimizer = torch.optim.Adam(groups)

        loader = DataLoader(MaskedMultiModalLoader(inputs, mask, targets),
                            batch_size=256, shuffle=True, drop_last=False)
        n_batch = len(loader)

        for _ in range(self.epochs + 1):
            for b_inputs, b_mask, b_labels in loader:
                self.optimizer.zero_grad()
                output, _, _, attn_norm, _ = self.dyam(b_inputs, b_mask)
                loss = self.criterion(output, b_labels)
                pen = sum(self._alpha_of(ch) * w.norm(p=2) for w, ch in uniq)
                pen = pen + self.alpha * sum(m.weight.norm(p=2)
                                             for m in self.dyam.l_logvar_linears)
                ((loss + pen + self.beta * attn_norm) / n_batch).backward()
                self.optimizer.step()

        output, _, _, _, _ = self.dyam(inputs, mask)
        return self.response_zscore(output, targets).detach().cpu().numpy()


def pooled_rad_filter(l1_filter, mask_columns):
    """Bộ lọc L1 dùng tổn thương GỘP cho mọi vị trí -> cả ba kênh nhận cùng một bộ đặc trưng.

    lesion_index của PC/PL/LN là 1-2 / 3-4 / 5-6 nên gộp không làm lẫn nhóm khi tính độ bền.
    """
    pos = [i for i, c in enumerate(mask_columns) if c in RAD_SITES and i in l1_filter]
    if len(pos) < 2:
        return l1_filter, []
    pooled = pd.concat([l1_filter[i]["l1_selection_df"] for i in pos])
    new = dict(l1_filter)
    for i in pos:
        new[i] = {"l1_selection_df": pooled, "kwargs": l1_filter[i]["kwargs"]}
    return new, pos


def train_shared(data, mask, labels, l1_filter, model_params, folds=10, seed=42,
                 share=True, light=True, light_lr=0.005, light_alpha=0.0001):
    """train_uniform_avg với ModelShared. `share`: buộc trọng số radiomics; `light`: chế độ nhẹ
    cho các kênh không phải radiomics."""
    cols = list(mask.columns)
    params = dict(model_params)
    if share:
        l1_filter, pos = pooled_rad_filter(l1_filter, cols)
        params["shared_channels"] = pos
    if light:
        params["light_channels"] = [i for i, c in enumerate(cols) if c not in RAD_SITES]
        params["light_lr"], params["light_alpha"] = light_lr, light_alpha

    saved = mua.MultiModalDynamicModelUncertainty
    mua.MultiModalDynamicModelUncertainty = ModelShared
    try:
        return mua.train_uniform_avg(data, mask, labels, l1_filter, params,
                                     folds=folds, seed=seed)
    finally:
        mua.MultiModalDynamicModelUncertainty = saved
