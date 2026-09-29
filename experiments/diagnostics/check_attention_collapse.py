"""
Step 2 — Chẩn đoán: OvO gate có thật sự hoạt động không?

Nghi ngờ: `attn_score / l_feature_factor[i]` (lung_helpers.py:1209) chia logit cho SỐ LƯỢNG
FEATURE. Với radiomics vài trăm feature, `score_i - mean_others` co về ~0 nên
`sigmoid(≈0) ≈ 0.5` cho mọi modality → gate phẳng, OvO không làm gì cả.

Script probe trực tiếp vào clf.dyam thay vì đọc cột attn_* của summary_df, vì với OvO
`attentions` trả về là mixing_matrix 3D (outer product) chứ không phải attention weight.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))
sys.path.insert(0, str(_HERE.parents[1]))

from common import benchmarks as B          # noqa: E402
from common.data_setup import load_all      # noqa: E402
from common.evaluate import save_result     # noqa: E402

from lung_helpers import (                  # noqa: E402
    MultiModalDynamicModel,
    MultiModalDynamicModelOvO,
    get_training_data,
    l1_filter_features_list,
)

BM = B.PRIMARY  # BM1 = Rad+IHC-G+Gen+PDL1


def scale_inputs(clf, arrays):
    """Lặp lại đúng cách get_summary_scores chuẩn hoá đầu vào."""
    out = []
    for i, X in enumerate(arrays):
        X = np.asarray(X, dtype=np.float64)
        if i in clf.noscale:
            out.append(torch.tensor(np.nan_to_num(X)).float().to(clf.device))
        else:
            out.append(torch.tensor(
                np.nan_to_num(clf.l_scalers[i].transform(X))).float().to(clf.device))
    return out


def probe_ovo(clf, inputs, mask):
    """Tái hiện AttentionMatrixOvO.forward, giữ lại các giá trị trung gian."""
    dyam = clf.dyam
    raw_pre, raw_post = [], []
    for i, x in enumerate(inputs):
        pre = dyam.l_attn_linears[i](x)
        raw_pre.append(pre)
        raw_post.append(pre / dyam.l_feature_factor[i].to(x.device))

    pre_t = torch.cat(raw_pre, axis=1)
    post_t = torch.cat(raw_post, axis=1)

    n_mod = post_t.shape[1]
    ovo = []
    for i in range(n_mod):
        score_i = post_t[:, i:i + 1]
        others_mask = mask.clone()
        others_mask[:, i] = 0
        score_others = post_t.clone()
        score_others[:, i] = 0
        n_others = torch.clamp(others_mask.sum(dim=1, keepdim=True), min=1)
        mean_others = (score_others * others_mask).sum(dim=1, keepdim=True) / n_others
        ovo.append(torch.sigmoid(score_i - mean_others))
    ovo_t = torch.cat(ovo, dim=1)

    attn_scores = mask * F.softplus(ovo_t)
    attn_weight = F.normalize(attn_scores, p=1, dim=1)

    d = lambda t: t.detach().cpu().numpy()  # noqa: E731
    return d(pre_t), d(post_t), d(ovo_t), d(attn_weight)


def main():
    ctx = load_all()
    bm = B.BENCHMARKS[BM]
    print(f"Benchmark: {BM} = {bm['name']}")

    data, mask_df, labels = get_training_data(
        bm["modalities"], ctx.modality_dict, ctx.modality_MASK, ctx.df_outcomes)
    l1_filter = B.get_filters(BM, ctx)

    # Dùng đúng fold 0 của KFold(seed=42) như trong train()/train_ovo()
    kf = KFold(n_splits=B.FOLDS, shuffle=True, random_state=42)
    train_idx, test_idx = next(iter(kf.split(labels.index)))
    train_px, valid_px = labels.index[train_idx], labels.index[test_idx]

    modality_list = [df.copy(deep=True) for df in data]
    for pos, filt in l1_filter.items():
        l1_filter_features_list(modality_list, filt["l1_selection_df"], labels,
                                valid_px, pos, **filt["kwargs"])

    names = list(mask_df.columns)
    n_feats = [len(df.columns) for df in modality_list]
    print("\nSố feature sau L1 filter:")
    for nm, nf in zip(names, n_feats):
        print(f"   {nm:<22} {nf:>5}")

    tr_in = [df.loc[train_px].values for df in modality_list]
    va_in = [df.loc[valid_px].values for df in modality_list]
    tr_mask = mask_df.loc[train_px].astype(int).values
    va_mask = mask_df.loc[valid_px].astype(int).values
    tr_y = labels.loc[train_px, "label"]

    params = dict(B.MODEL_PARAMS)
    params_ovo = {k: v for k, v in params.items() if k != "cross_modality_enabled"}

    torch.manual_seed(42)
    np.random.seed(42)
    print("\n[1/2] Fit OvO ...")
    clf_ovo = MultiModalDynamicModelOvO(**params_ovo)
    clf_ovo.fit(tr_in, tr_mask, tr_y)

    torch.manual_seed(42)
    np.random.seed(42)
    print("[2/2] Fit Original ...")
    clf_org = MultiModalDynamicModel(**params)
    clf_org.fit(tr_in, tr_mask, tr_y)

    mask_t = torch.tensor(va_mask).float().to(clf_ovo.device)

    ovo_in = scale_inputs(clf_ovo, va_in)
    pre, post, ovo_sig, attn_ovo = probe_ovo(clf_ovo, ovo_in, mask_t)

    org_in = scale_inputs(clf_org, va_in)
    with torch.no_grad():
        _, _, attn_org, _, _ = clf_org.dyam(org_in, mask_t)
    attn_org = attn_org.detach().cpu().numpy()

    n_mod = len(names)
    uniform = 1.0 / n_mod

    report = {
        "benchmark": BM,
        "benchmark_name": bm["name"],
        "n_valid": int(len(valid_px)),
        "n_modalities": n_mod,
        "uniform_weight": uniform,
        "n_features": {nm: int(nf) for nm, nf in zip(names, n_feats)},
        "modalities": {},
        "global": {},
    }

    print("\n" + "=" * 78)
    print(f"{'modality':<22} {'#feat':>6} {'sd(raw)':>10} {'sd(÷feat)':>11} "
          f"{'sd(sigm)':>10} {'mean(a)':>9} {'sd(a)':>8}")
    print("-" * 78)
    for i, nm in enumerate(names):
        row = {
            "n_features": int(n_feats[i]),
            "sd_raw_pre_division": float(pre[:, i].std()),
            "sd_raw_post_division": float(post[:, i].std()),
            "sd_ovo_sigmoid": float(ovo_sig[:, i].std()),
            "mean_ovo_sigmoid": float(ovo_sig[:, i].mean()),
            "mean_attn_ovo": float(attn_ovo[:, i].mean()),
            "sd_attn_ovo": float(attn_ovo[:, i].std()),
            "mean_attn_original": float(attn_org[:, i].mean()),
            "sd_attn_original": float(attn_org[:, i].std()),
        }
        report["modalities"][nm] = row
        print(f"{nm:<22} {n_feats[i]:>6} {row['sd_raw_pre_division']:>10.3e} "
              f"{row['sd_raw_post_division']:>11.3e} {row['sd_ovo_sigmoid']:>10.3e} "
              f"{row['mean_attn_ovo']:>9.4f} {row['sd_attn_ovo']:>8.4f}")
    print("=" * 78)

    sd_ovo_mean = float(np.mean([r["sd_attn_ovo"] for r in report["modalities"].values()]))
    sd_org_mean = float(np.mean([r["sd_attn_original"] for r in report["modalities"].values()]))
    sig_dev = float(np.abs(ovo_sig - 0.5).max())

    # Kiểm chứng then chốt: attention có khác gì so với "chia đều cho các modality có sẵn"?
    # Nếu gate phẳng thì attn_weight -> mask_i / sum(mask), tức attention không học được gì
    # ngoài việc đếm modality nào có mặt.
    mask_np = va_mask.astype(float)
    uniform_mask = mask_np / np.clip(mask_np.sum(axis=1, keepdims=True), 1, None)
    dev_ovo = float(np.abs(attn_ovo - uniform_mask).max())
    dev_org = float(np.abs(attn_org - uniform_mask).max())
    mad_ovo = float(np.abs(attn_ovo - uniform_mask).mean())
    mad_org = float(np.abs(attn_org - uniform_mask).mean())
    report["vs_mask_uniform"] = {
        "max_abs_dev_ovo": dev_ovo,
        "max_abs_dev_original": dev_org,
        "mean_abs_dev_ovo": mad_ovo,
        "mean_abs_dev_original": mad_org,
        "ovo_is_mask_uniform": bool(dev_ovo < 0.02),
    }
    print(f"\n--- So với 'chia đều cho modality có sẵn' (mask_i / Σmask) ---")
    print(f"OvO      : lệch trung bình = {mad_ovo:.5f} | lệch lớn nhất = {dev_ovo:.5f}")
    print(f"Original : lệch trung bình = {mad_org:.5f} | lệch lớn nhất = {dev_org:.5f}")

    report["global"] = {
        "mean_sd_attn_ovo": sd_ovo_mean,
        "mean_sd_attn_original": sd_org_mean,
        "max_abs_sigmoid_deviation_from_half": sig_dev,
        "max_abs_attn_ovo_deviation_from_uniform":
            float(np.abs(attn_ovo - uniform).max()),
        "collapse_confirmed": bool(sd_ovo_mean < 0.02),
    }

    print(f"\nsd trung bình của attention weight   OvO = {sd_ovo_mean:.5f} | "
          f"Original = {sd_org_mean:.5f}")
    print(f"|sigmoid - 0.5| lớn nhất             = {sig_dev:.5f}")
    print(f"|a_i - 1/N| lớn nhất (N={n_mod}, 1/N={uniform:.4f}) = "
          f"{report['global']['max_abs_attn_ovo_deviation_from_uniform']:.5f}")
    print(f"\nCỔNG QUYẾT ĐỊNH sd(attn_ovo) < 0.02 -> "
          f"{'GATE COLLAPSE ĐƯỢC XÁC NHẬN' if sd_ovo_mean < 0.02 else 'KHÔNG collapse'}")

    save_result("diagnostics", report)


if __name__ == "__main__":
    main()
