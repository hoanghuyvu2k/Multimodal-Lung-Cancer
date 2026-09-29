"""Kiểm chứng: embedding (pathology Phikon + radiology MedicalNet) khi KẾT HỢP đa nguồn có tốt hơn thủ công?

Thay imaging thủ công bằng embedding (PCA-64 để công bằng với feature đã lọc), giữ Gen/PDL1/Labs. Chạy fusion
uniform_avg + DyAM, so 5 cấu hình. Discovery-CV 5-seed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lung_helpers import get_training_data                      # noqa: E402
from baselines.model_uniform_avg import train_uniform_avg       # noqa: E402
from common import benchmarks as B                              # noqa: E402
from common.data_setup import load_all                          # noqa: E402
from common.evaluate import save_result                         # noqa: E402
from allcombo.run_combo import train as train_dyam             # noqa: E402

DS = Path(__file__).resolve().parents[2].parent / "datasets"
RAD = ["rad_lesion_pc", "rad_lesion_pl", "rad_lesion_ln"]
RAD_FILT_NAMES = {RAD[i]: i for i in range(3)}


def add_embed_modality(ctx, name, parquet, dim=64):
    emb = pd.read_parquet(DS / parquet)
    emb = emb[emb.index.isin(ctx.df_outcomes.index)]
    X = RobustScaler().fit_transform(np.nan_to_num(emb.values))
    k = min(dim, X.shape[1], X.shape[0] - 1)
    red = PCA(n_components=k, random_state=0).fit_transform(X)
    df = pd.DataFrame(red, index=emb.index, columns=[f"{name}_{i}" for i in range(k)])
    ctx.modality_dict[name] = df.reindex(ctx.df_outcomes.index)
    ctx.modality_MASK[name] = ctx.df_outcomes.index.isin(emb.index)
    return int(ctx.modality_MASK[name].sum())


def build_l1(modality_list, ctx):
    return {modality_list.index(n): ctx.rad_filters[RAD_FILT_NAMES[n]]
            for n in modality_list if n in RAD_FILT_NAMES}


def run_cfg(train_fn, modality_list, ctx, seeds=B.SEEDS):
    data, mask, labels = get_training_data(modality_list, ctx.modality_dict,
                                           ctx.modality_MASK, ctx.df_outcomes)
    l1 = build_l1(modality_list, ctx)
    params = dict(B.MODEL_PARAMS)
    cols, y = {}, None
    for s in seeds:
        rng = np.random.default_rng(s)
        sh = labels.iloc[rng.permutation(len(labels))]
        summ, _ = train_fn(data, mask, sh, l1, params, folds=B.FOLDS, seed=s)
        cols[s] = pd.Series(summ["score"].astype(float).values, index=summ.index)
        if y is None:
            y = pd.Series(summ["label"].astype(float).values, index=summ.index)
    sc = pd.DataFrame(cols)
    per = [roc_auc_score(y.loc[sc.index], sc[s]) for s in seeds]
    return float(np.mean(per)), float(np.std(per, ddof=1))


def main():
    ctx = load_all()
    n_p = add_embed_modality(ctx, "path_emb", "path_fm_embed_discovery.parquet")
    n_r = add_embed_modality(ctx, "rad_emb", "ct3d_fm_embed_discovery.parquet")
    print(f"embedding modalities: path_emb n={n_p}, rad_emb n={n_r}")

    G, D, L = "gen_driver_mut_amp", "cnl_pdl1_score", "cnl_dem_labs"
    P_HC, R_HC = ["path_ihc_glcm"], RAD
    configs = {
        "A hand-crafted (BM1)":   R_HC + P_HC + [G, D],
        "B embed cả 2":           ["rad_emb", "path_emb", G, D],
        "C embed path (rad HC)":  R_HC + ["path_emb", G, D],
        "D embed rad (path HC)":  ["rad_emb"] + P_HC + [G, D],
        "A+Labs hand-crafted":    R_HC + P_HC + [G, D, L],
        "B+Labs embed cả 2":      ["rad_emb", "path_emb", G, D, L],
    }
    out = {}
    print(f"\n{'=' * 78}\nFUSION: embedding vs hand-crafted (discovery-CV 5-seed)\n{'=' * 78}")
    print(f"{'config':<24}{'uniform_avg':>18}{'DyAM':>18}")
    for name, mods in configs.items():
        um, us = run_cfg(train_uniform_avg, mods, ctx)
        dm, ds = run_cfg(train_dyam, mods, ctx)
        out[name] = {"modalities": mods, "uniform": [um, us], "dyam": [dm, ds]}
        print(f"{name:<24}{f'{um:.4f}±{us:.4f}':>18}{f'{dm:.4f}±{ds:.4f}':>18}", flush=True)
    print("=" * 78)
    save_result("fm_fusion", out)
    a = out["A hand-crafted (BM1)"]["uniform"][0]
    b = out["B embed cả 2"]["uniform"][0]
    print(f"\nĐIỂM CHỐT: embed cả 2 (B={b:.4f}) vs hand-crafted (A={a:.4f}) -> Δ={b-a:+.4f}")


if __name__ == "__main__":
    main()
