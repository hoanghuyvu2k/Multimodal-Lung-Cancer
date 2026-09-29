"""
Baseline `uniform_avg` — KHÔNG CÓ ATTENTION.

    rᵢ = tanh(Wᵢ · xᵢ)
    aᵢ = maskᵢ / Σmask          (chia đều cho các modality có sẵn, KHÔNG học)
    total = Σ aᵢ rᵢ

Đây chính là file đã sinh ra các con số `uniform_avg` trong results/method_B.json —
giữ nguyên để tái lập chính xác. Nó vốn là `model_uncertainty.py` của phương pháp B chạy ở
`logvar_clamp = 0`: khi đó log σ² bị chặn về đúng 0, precision hằng số 1, nên aᵢ = maskᵢ/Σmask.
Đầu head log-variance vẫn tồn tại nhưng KHÔNG ảnh hưởng đầu ra (gradient qua nó bằng 0).

Phương pháp B (precision weighting, logvar_clamp = 0.25) đã được đánh giá đầy đủ và BỊ LOẠI:
Δ ≥ +0.015 chỉ trên 2/4 config, p = 0.116 trên BM1. Chênh giữa có và không có attention là
+0.005 / +0.005 / −0.001 / +0.007 trên BM1–BM4, tức không phân biệt được.

Mặc định logvar_clamp đổi thành 0.0 để dùng làm baseline.
"""

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

_CODE_DIR = Path(__file__).resolve().parents[2]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from lung_helpers import (  # noqa: E402
    MaskedMultiModalLoader,
    auc_roc_ci,
    find_optimal_cutoff,
    get_summary_df,
    l1_filter_features_list,
    set_global_seed,
)

# Chặn log-variance để precision không tràn: exp(±LOGVAR_CLAMP) -> [0.018, 54.6]
LOGVAR_CLAMP = 0.0


class AttentionMatrixUncertainty(nn.Module):
    """Trọng số attention suy ra từ precision dự đoán, thay vì từ score học trực tiếp."""

    def __init__(self, attention_gate_enabled=True, init_seed=42, logvar_clamp=LOGVAR_CLAMP):
        super().__init__()
        torch.manual_seed(init_seed)
        self.logvar_clamp = logvar_clamp
        self.input_channel_shapes = []
        self.tanh = nn.Tanh()
        self.attention_gate_enabled = attention_gate_enabled

    def add_channel(self, channel_template):
        self.input_channel_shapes.append(channel_template.shape[1])
        self.n_input_channels = len(self.input_channel_shapes)

    def setup_matrix(self):
        self.l_risk_linears = nn.ModuleList()
        self.l_logvar_linears = nn.ModuleList()
        for channel_shape in self.input_channel_shapes:
            self.l_risk_linears.append(nn.Linear(channel_shape, 1))
            self.l_logvar_linears.append(nn.Linear(channel_shape, 1))
        for m in self.l_risk_linears:
            torch.nn.init.zeros_(m.bias)
        # bias = 0 -> log σ² = 0 -> σ² = 1 -> precision = 1 cho mọi modality lúc khởi đầu,
        # tức khởi điểm trùng đúng với "chia đều theo mask". Mọi sai khác về sau là học được.
        for m in self.l_logvar_linears:
            torch.nn.init.zeros_(m.bias)

    def get_l2_weight_sum(self):
        return torch.stack([m.weight.norm(p=2) for m in
                            list(self.l_risk_linears) + list(self.l_logvar_linears)]).sum()

    def forward(self, inputs, mask):
        risks, logvars = [], []
        for i, x in enumerate(inputs):
            risks.append(self.l_risk_linears[i](x))
            logvars.append(self.l_logvar_linears[i](x))

        risk_scores = torch.cat(risks, axis=1)
        log_var = torch.clamp(torch.cat(logvars, axis=1), -self.logvar_clamp, self.logvar_clamp)

        risk_weights = mask * self.tanh(risk_scores)      # GIỮ tanh như bản gốc
        precision = mask * torch.exp(-log_var)            # pᵢ = 1/σᵢ²
        attn_weight = F.normalize(precision, p=1, dim=1)  # GIỮ ràng buộc Σaᵢ = 1

        if self.attention_gate_enabled:
            total_risk = torch.sum(risk_weights * attn_weight, dim=1)
        else:
            total_risk = torch.sum(risk_weights, dim=1)

        attn_norm = attn_weight.norm(p=2, dim=1).mean()
        risk_norm = risk_scores.norm(p=2, dim=1).mean()

        return total_risk, risk_weights, attn_weight, attn_norm, risk_norm


class MultiModalDynamicModelUncertainty(nn.Module):
    def __init__(self, epochs=100, alpha=1.0, beta=1.0, lr=0.01, hidden_factor=2,
                 class_weight="balanced", print_on=50, no_scale=(),
                 attention_gate_enabled=True, init_seed=42, logvar_clamp=LOGVAR_CLAMP,
                 **_ignored):
        super().__init__()
        self.epochs = epochs
        self.alpha = alpha
        self.beta = beta
        self.lr = lr
        self.class_weight = class_weight
        self.print_on = print_on
        self.noscale = list(no_scale)
        self.attention_gate_enabled = attention_gate_enabled
        self.init_seed = init_seed
        self.logvar_clamp = logvar_clamp
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def response_zscore(self, input, target=None):
        if self.training:
            threshold = find_optimal_cutoff(target.detach(), input.detach())
            self.mu = torch.tensor(threshold).to(self.device)
            self.std = input.std(dim=0)
        return (input - self.mu) / self.std

    @staticmethod
    def _to_numeric(X):
        if isinstance(X, pd.DataFrame):
            return X.select_dtypes(include=[np.number]).values
        return np.asarray(X).astype(np.float64)

    def _prepare(self, l_X_INPUTS, fitting):
        out = []
        for i, X in enumerate(l_X_INPUTS):
            X = self._to_numeric(X)
            if i in self.noscale:
                arr = np.nan_to_num(X)
            elif fitting:
                scaler = RobustScaler()
                arr = np.nan_to_num(scaler.fit_transform(X))
                self.d_scalers[i] = scaler
            else:
                arr = np.nan_to_num(self.d_scalers[i].transform(X))
            out.append((torch.tensor(arr).float().to(self.device), X))
        return out

    def fit(self, l_X_INPUTS, arr_MASK, vector_Y):
        self.train()
        l_X_INPUTS = copy.deepcopy(l_X_INPUTS)
        arr_MASK = copy.deepcopy(arr_MASK)
        vector_Y = copy.deepcopy(vector_Y)

        self.dyam = AttentionMatrixUncertainty(
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
                output, _, _, attn_norm, _ = self.dyam(b_inputs, b_mask)
                loss = self.criterion(output, b_labels)
                l2 = self.dyam.get_l2_weight_sum()
                total_loss = (loss + (self.alpha * l2) + (self.beta * attn_norm)) / n_batch
                total_loss.backward()
                self.optimizer.step()

        output, _, _, _, _ = self.dyam(inputs, mask)
        return self.response_zscore(output, targets).detach().cpu().numpy()

    def _forward_eval(self, l_X_INPUTS, arr_MASK):
        self.eval()
        prepared = self._prepare(copy.deepcopy(l_X_INPUTS), fitting=False)
        inputs = [t for t, _ in prepared]
        mask = torch.tensor(copy.deepcopy(arr_MASK)).float().to(self.device)
        with torch.no_grad():
            return self.dyam(inputs, mask), mask

    def predict_proba(self, l_X_INPUTS, arr_MASK):
        (output, _, _, _, _), _ = self._forward_eval(l_X_INPUTS, arr_MASK)
        return self.response_zscore(output).detach().cpu().numpy()

    def get_summary_scores(self, l_X_INPUTS, arr_MASK):
        (output, risk_scores, attn_weight, _, _), mask = self._forward_eval(
            l_X_INPUTS, arr_MASK)
        share = mask.sum(dim=1, keepdim=True) * attn_weight
        d = lambda t: t.detach().cpu().numpy()  # noqa: E731
        return (d(self.response_zscore(output)), d(risk_scores),
                d(attn_weight), d(share))

    def get_coefs(self, modality_list, modality_mask):
        coefs = torch.cat([x.weight.flatten() for x in self.dyam.l_risk_linears],
                          dim=0).flatten().detach().cpu().numpy()
        columns = []
        for i, df in enumerate(modality_list):
            columns.extend(
                df.add_prefix("risk___" + modality_mask.columns[i] + "__").columns)
        return pd.DataFrame([coefs, np.sign(coefs)], columns=columns,
                            index=["coef", "sign"])


def train_uniform_avg(modality_list_in, modality_mask, outcomes, l1_dfs_filter,
                      model_params, folds=10, seed=42):
    set_global_seed(seed)
    l_v_scores, l_v_labels, d_summarys_all = [], [], {}
    if folds == "LOO":
        folds = len(outcomes.index)
    df_coef_agg = pd.DataFrame()
    params = {k: v for k, v in model_params.items() if k != "cross_modality_enabled"}

    kf = KFold(n_splits=folds, random_state=0, shuffle=True)
    for fold, (tr, te) in enumerate(tqdm(list(kf.split(outcomes.index)), file=sys.stdout)):
        modality_list = [df.copy(deep=True) for df in modality_list_in]
        train_px, valid_px = outcomes.index[tr], outcomes.index[te]

        for pos, filt in l1_dfs_filter.items():
            l1_filter_features_list(modality_list, filt["l1_selection_df"], outcomes,
                                    valid_px, pos, **filt["kwargs"])
        if any(len(df.columns) == 0 for df in modality_list):
            continue

        tr_in = [df.loc[train_px].values for df in modality_list]
        va_in = [df.loc[valid_px].values for df in modality_list]
        tr_mask = modality_mask.loc[train_px].astype(int).values
        va_mask = modality_mask.loc[valid_px].astype(int).values
        tr_y = outcomes.loc[train_px, "label"]
        va_y = outcomes.loc[valid_px, "label"].values

        clf = MultiModalDynamicModelUncertainty(**params)
        clf.fit(tr_in, tr_mask, tr_y)
        df_coef_agg = pd.concat([df_coef_agg, clf.get_coefs(modality_list, modality_mask)])

        valid_scores = clf.predict_proba(va_in, va_mask)
        l_v_scores.extend(valid_scores)
        l_v_labels.extend(va_y)

        _, risks, attns, shares = clf.get_summary_scores(va_in, va_mask)
        for idx, px in enumerate(valid_px):
            row = {"label": va_y[idx], "score": valid_scores[idx], "fold": fold}
            for i, v in enumerate(risks[idx]):
                row[f"risk_{modality_mask.columns[i]}"] = v
            for i, v in enumerate(attns[idx]):
                row[f"attn_{modality_mask.columns[i]}"] = v
            for i, v in enumerate(shares[idx]):
                row[f"share_{modality_mask.columns[i]}"] = v
            d_summarys_all[px] = row

        if 1.0 in l_v_labels and 0.0 in l_v_labels:
            auc, ci = auc_roc_ci(l_v_labels, l_v_scores, 0.95)
    print(f"Uncertainty Fold [{fold + 1}] AUC = {auc:.3f} "
          f"+/- {(ci[1] - ci[0]) / 2.0:.3f} 95% CL")

    return get_summary_df(d_summarys_all), df_coef_agg
