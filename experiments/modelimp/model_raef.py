"""RAEF — Reliability- and Availability-aware Expert Fusion.

Một họ mô hình nhận `uniform_avg` làm TRƯỜNG HỢP ĐẶC BIỆT (tại khởi tạo), nên không thể tệ hơn
nó lúc bắt đầu; mỗi thành phần thêm vào đều có căn cứ từ thí nghiệm của dự án:

    s(x, m) = Σ_i m_i · ρ_i · tanh(W_i x_i) / Σ_i m_i · ρ_i   +   v · m
              └────────── (R) hợp nhất theo độ tin cậy ──────────┘   └ (A) tiên nghiệm sẵn-có ┘

  Chuyên gia theo modality  r_i = tanh(W_i x_i). Radiomics có chuyên gia + bộ chọn đặc trưng RIÊNG
      cho từng vị trí tổn thương (đã kiểm: gộp vị trí làm AUC tụt 0.11 — §2.17).

  (R) Trọng số độ tin cậy ρ_i = softplus(θ_i): MỘT số vô hướng cho mỗi modality, KHÔNG phụ thuộc
      đầu vào. Căn cứ: attention phụ thuộc đầu vào (DyAM N×N, OvO N ánh xạ tuyến tính) không hơn
      trung bình cộng ở n=247 (§2.5), tức dữ liệu không đủ để ước lượng trọng số theo từng bệnh
      nhân; nhưng độ tin cậy theo modality thì khác nhau rõ (ngoại kiểm pathology 0.77, radiomics
      ~0.45 — §2.3). θ khởi tạo bằng nhau -> đúng bằng uniform_avg.

  (A) Tiên nghiệm sẵn-có-dữ-liệu v·m: số hạng tuyến tính trên vector mask. Căn cứ: mẫu khuyết
      mang tín hiệu tiên lượng (chỉ riêng "có tiêu bản pathology" đạt AUC 0.668 — §2.14).
      v khởi tạo 0 -> đúng bằng uniform_avg.

Tổng tham số thêm vào: 2N (N = số modality) — ngang OvO, ít hơn DyAM (N²).
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


class RAEFMatrix(mua.AttentionMatrixUncertainty):
    def __init__(self, *a, use_rel=True, use_avail=True, **kw):
        super().__init__(*a, **kw)
        self.use_rel = use_rel
        self.use_avail = use_avail

    def setup_matrix(self):
        super().setup_matrix()
        n = self.n_input_channels
        self.rel_logit = nn.Parameter(torch.zeros(n))        # ρ bằng nhau -> uniform_avg
        self.avail = nn.Linear(n, 1, bias=False)
        torch.nn.init.zeros_(self.avail.weight)              # v = 0     -> uniform_avg

    def reliability(self):
        if getattr(self, "rel_fixed", None) is not None:
            return self.rel_fixed                            # ước lượng bằng CV, không học
        return F.softplus(self.rel_logit)

    def forward(self, inputs, mask):
        risk_scores = torch.cat([self.l_risk_linears[i](x) for i, x in enumerate(inputs)],
                                axis=1)
        risk_weights = mask * self.tanh(risk_scores)
        w = mask * self.reliability() if self.use_rel else mask
        attn_weight = F.normalize(w, p=1, dim=1)
        total = torch.sum(risk_weights * attn_weight, dim=1)
        if self.use_avail:
            total = total + self.avail(mask).squeeze(1)
        attn_norm = attn_weight.norm(p=2, dim=1).mean()
        risk_norm = risk_scores.norm(p=2, dim=1).mean()
        return total, risk_weights, attn_weight, attn_norm, risk_norm

    def get_l2_weight_sum(self):
        base = torch.stack([m.weight.norm(p=2) for m in self.l_risk_linears]).sum()
        if not self.use_avail:
            return base
        # v khởi tạo = 0: norm tại 0 có gradient không xác định -> dùng dạng có epsilon
        return base + torch.sqrt((self.avail.weight ** 2).sum() + 1e-12)


class ModelRAEF(_Base):
    def __init__(self, *a, use_rel=True, use_avail=True, rel_mode="learn", **kw):
        super().__init__(*a, **kw)
        self.use_rel, self.use_avail, self.rel_mode = use_rel, use_avail, rel_mode

    @staticmethod
    def _cv_reliability(inputs, mask, targets, floor=0.02):
        """ρ_i = max(AUC_i − 0.5, floor): AUC out-of-fold của hồi quy logistic 3-fold trên TỪNG
        modality, chỉ dùng bệnh nhân có modality đó. Không có tham số nào học bằng gradient."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        y_all = targets.detach().cpu().numpy()
        rho = []
        for i, x in enumerate(inputs):
            rows = mask[:, i].detach().cpu().numpy() > 0
            X, y = x.detach().cpu().numpy()[rows], y_all[rows]
            if rows.sum() < 30 or min((y == 0).sum(), (y == 1).sum()) < 6:
                rho.append(floor)
                continue
            oof = np.zeros(len(y))
            for tr, te in StratifiedKFold(3, shuffle=True, random_state=0).split(X, y):
                clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=300)
                oof[te] = clf.fit(X[tr], y[tr]).decision_function(X[te])
            rho.append(max(roc_auc_score(y, oof) - 0.5, floor))
        return torch.tensor(rho, dtype=torch.float32)

    def fit(self, l_X_INPUTS, arr_MASK, vector_Y):
        self.train()
        l_X_INPUTS = copy.deepcopy(l_X_INPUTS)
        arr_MASK = copy.deepcopy(arr_MASK)
        vector_Y = copy.deepcopy(vector_Y)

        self.dyam = RAEFMatrix(
            attention_gate_enabled=self.attention_gate_enabled, init_seed=self.init_seed,
            logvar_clamp=self.logvar_clamp, use_rel=self.use_rel, use_avail=self.use_avail)
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
        if self.use_rel and self.rel_mode == "cv":
            self.dyam.rel_fixed = self._cv_reliability(inputs, mask, targets).to(self.device)
            self.dyam.rel_logit.requires_grad_(False)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=sum(targets == 0) / sum(targets == 1))
        self.optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad], lr=self.lr)

        loader = DataLoader(MaskedMultiModalLoader(inputs, mask, targets),
                            batch_size=256, shuffle=True, drop_last=False)
        n_batch = len(loader)
        for _ in range(self.epochs + 1):
            for b_inputs, b_mask, b_labels in loader:
                self.optimizer.zero_grad()
                output, _, _, attn_norm, _ = self.dyam(b_inputs, b_mask)
                loss = self.criterion(output, b_labels)
                l2 = self.dyam.get_l2_weight_sum()
                ((loss + self.alpha * l2 + self.beta * attn_norm) / n_batch).backward()
                self.optimizer.step()

        output, _, _, _, _ = self.dyam(inputs, mask)
        return self.response_zscore(output, targets).detach().cpu().numpy()


def train_raef(data, mask, labels, l1_filter, model_params, folds=10, seed=42,
               use_rel=True, use_avail=True, rel_mode="learn"):
    params = {**model_params, "use_rel": use_rel, "use_avail": use_avail,
              "rel_mode": rel_mode}
    saved = mua.MultiModalDynamicModelUncertainty
    mua.MultiModalDynamicModelUncertainty = ModelRAEF
    try:
        return mua.train_uniform_avg(data, mask, labels, l1_filter, params,
                                     folds=folds, seed=seed)
    finally:
        mua.MultiModalDynamicModelUncertainty = saved
