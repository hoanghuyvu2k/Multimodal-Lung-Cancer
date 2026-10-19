"""Đặc trưng mô hình nền từ ẢNH GỐC (embedding đã trích sẵn 2026-07) đưa vào bộ tốt nhất
(uniform_avg + NLP + miss_pattern + C18). Mốc so: c18stack__{slug}.csv cùng seed.

    f1  thêm chuyên gia pathology thứ hai `path_phikon`: Phikon 768d -> PCA-16 (163 bệnh nhân, so với 105 của GLCM/IHC-A)
    f2  GHÉP embedding CT (fmcib 4096d -> PCA-16) vào chuyên gia radiomics PC, cùng một đầu tuyến tính với đặc trưng
        thủ công đã lọc (kiểu "foundation-enhanced radiomics"); cột embedding không qua bộ lọc radiomics

PCA khớp không giám sát trên toàn bộ bệnh nhân có embedding (không dùng nhãn), như đợt 2026-07 (§2.6).

    PYTHONIOENCODING=utf-8 python experiments/archloop/run_fm.py f1 --seeds=5 --only=0,1
"""
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
_CODE = _EXP.parent
for _p in (str(_CODE), str(_EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
import torch                                                      # noqa: E402
from sklearn.decomposition import PCA                             # noqa: E402
from sklearn.preprocessing import RobustScaler                    # noqa: E402

from archloop import harness as H                                 # noqa: E402  (áp bản vá scipy)
from archloop import cands as C                                   # noqa: E402
from allcombo.run_paper_combos import PAPER_COMBOS, make_filter   # noqa: E402
import baselines.model_uniform_avg as mua                         # noqa: E402
from common import benchmarks as B                                # noqa: E402
from common import data_setup as ds                               # noqa: E402
from common.data_setup import load_all                            # noqa: E402
from missingness.run_miss import MISS, add_miss_modality          # noqa: E402
from nlp_verify.nlp_modality import add_nlp_modality              # noqa: E402
from lung_helpers import get_training_data                        # noqa: E402

COMBOS = ["Rad+IHC-G+Gen+PDL1", "Rad+IHC-A+Gen+TMB+PDL1", "Rad+IHC-A+Gen+PDL1",
          "Rad+IHC-A+MutAmp"]
PHIKON, DIM = "path_phikon", 16


def reduced(parquet, prefix, ctx, dim=DIM):
    emb = pd.read_parquet(Path(ds.BASE_DB_DIR) / parquet)
    emb = emb[emb.index.isin(ctx.df_outcomes.index)]
    X = RobustScaler().fit_transform(np.nan_to_num(emb.values))
    red = PCA(n_components=dim, random_state=0).fit_transform(X)
    return pd.DataFrame(red, index=emb.index, columns=[f"{prefix}_{i}" for i in range(dim)])


def keep_fm_filter(modality_list, df_input, df_outcome, exclude_index, position, **kw):
    """Bộ lọc radiomics gốc, nhưng giữ lại các cột embedding `ctfm_*` đã ghép vào chuyên gia."""
    before = modality_list[position]
    extra = [c for c in before.columns if c.startswith("ctfm_")]
    out = _ORIG_FILTER(modality_list, df_input, df_outcome, exclude_index, position, **kw)
    if extra:
        modality_list[position] = pd.concat([modality_list[position], before[extra]], axis=1)
    return out


_ORIG_FILTER = mua.l1_filter_features_list


class ModelFoldPCA(C._BaseModel):
    """f1cv — như f1 nhưng RobustScaler + PCA của kênh Phikon được khớp TRONG từng fold, chỉ trên bệnh nhân train
    (bản f1 khớp PCA trên toàn bộ bệnh nhân -> rò rỉ không giám sát)."""

    def __init__(self, *a, pca_idx=None, pca_dim=DIM, **kw):
        super().__init__(*a, **kw)
        self.pca_idx, self.pca_dim = pca_idx, pca_dim

    def get_coefs(self, modality_list, modality_mask):
        return pd.DataFrame()

    def _prepare(self, l_X_INPUTS, fitting):
        l_X = list(l_X_INPUTS)
        raw = np.asarray(l_X[self.pca_idx], dtype=float)
        ok = ~np.isnan(raw).all(axis=1)
        if fitting:
            self._sc = RobustScaler().fit(np.nan_to_num(raw[ok]))
            self._pca = PCA(n_components=self.pca_dim, random_state=0).fit(
                self._sc.transform(np.nan_to_num(raw[ok])))
        red = np.full((raw.shape[0], self.pca_dim), np.nan)
        if ok.any():
            red[ok] = self._pca.transform(self._sc.transform(np.nan_to_num(raw[ok])))
        l_X[self.pca_idx] = pd.DataFrame(red)
        return super()._prepare(l_X, fitting)


def train_f1cv(data, mask, labels, l1, params, folds=10, seed=42):
    p = {**params, "pca_idx": list(mask.columns).index(PHIKON)}
    return C._with("MultiModalDynamicModelUncertainty", ModelFoldPCA, C.train_pdl1fill,
                   data, mask, labels, l1, p, folds=folds, seed=seed)


def train_f2(data, mask, labels, l1, params, folds=10, seed=42):
    return C._with("l1_filter_features_list", keep_fm_filter, C.train_pdl1fill,
                   data, mask, labels, l1, params, folds=folds, seed=seed)


def run_all21(ctx, part):
    """Mô hình đầy đủ + Phikon (PCA trong fold) trên 21 tổ hợp, 5 seed. Phikon chỉ được thêm ở tổ hợp CÓ pathology
    (IHC-A hoặc IHC-G) để không đổi nghĩa của tổ hợp; các tổ hợp còn lại trùng experiments/results/full21."""
    import json
    from sklearn.metrics import roc_auc_score
    out = _CODE / "experiments" / "results" / "full21_phikon"
    out.mkdir(parents=True, exist_ok=True)
    ph = pd.read_parquet(Path(ds.BASE_DB_DIR) / "path_fm_embed_discovery.parquet")
    ph = ph[ph.index.isin(ctx.df_outcomes.index)]
    ctx.modality_dict[PHIKON] = ph.reindex(ctx.df_outcomes.index)
    ctx.modality_MASK[PHIKON] = ctx.df_outcomes.index.isin(ph.index)
    i, n = (int(x) for x in part.split("/"))
    todo = [(lb, m, f) for lb, m, f in PAPER_COMBOS if any(x in C.PATH_CH for x in m)]
    for k, (label, mods, ftype) in enumerate(todo):
        f = out / f"{H.slug(label)}.json"
        if k % n != i or f.exists():
            continue
        base = [m for m in mods if m != "cnl_dem_labs"]
        add_miss_modality(ctx, base)
        full = base + ["cnl_nlp", MISS, PHIKON]
        data, mask, labels = get_training_data(full, ctx.modality_dict, ctx.modality_MASK,
                                               ctx.df_outcomes)
        params = {**B.MODEL_PARAMS, "no_scale": [full.index("cnl_nlp"), full.index(MISS)]}
        y = labels["label"].astype(float)
        aucs, cols = [], {}
        for seed in B.SEEDS:
            rng = np.random.default_rng(seed)
            summ, _ = train_f1cv(data, mask, labels.iloc[rng.permutation(len(labels))],
                                 make_filter(ftype, ctx), params, folds=B.FOLDS, seed=seed)
            s = pd.Series(summ["score"].astype(float).values, index=summ.index)
            cols[seed] = s
            aucs.append(float(roc_auc_score(y.loc[s.index], s)))
        z = pd.DataFrame(cols)
        ens = ((z - z.mean()) / z.std()).mean(axis=1)
        res = {"label": label, "n": int(len(y)), "per_seed": aucs, "mean": float(np.mean(aucs)),
               "sd": float(np.std(aucs, ddof=1)), "ens5": float(roc_auc_score(y.loc[ens.index], ens))}
        f.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[all21] {label}: {res['mean']:.4f} ± {res['sd']:.4f}  ens5 {res['ens5']:.4f}", flush=True)
    print("XONG")


def main():
    torch.set_num_threads(2)
    variant = sys.argv[1]
    arg = {a.split("=")[0]: a.split("=")[1] for a in sys.argv[2:] if a.startswith("--")}
    seeds = H.SEEDS20[:int(arg.get("--seeds", 20))]
    combos = [COMBOS[int(i)] for i in arg["--only"].split(",")] if "--only" in arg else COMBOS
    ctx = load_all()
    add_nlp_modality(ctx)
    if variant == "all21":
        return run_all21(ctx, arg.get("--part", "0/1"))
    restrict = variant == "f1cov"            # f1cov: Phikon CHỈ ở bệnh nhân đã có pathology thủ công (không thêm độ phủ)
    if restrict:
        variant = "f1cv"
    avail = variant == "f1avail"             # f1avail: đối chứng — chuyên gia CHỈ CÓ BIAS trên đúng 163 bệnh nhân có tiêu bản
    if avail:                                #          (không có nội dung ảnh) -> đo phần tín hiệu "có tiêu bản hay không"
        ph = pd.read_parquet(Path(ds.BASE_DB_DIR) / "path_fm_embed_discovery.parquet")
        ph = ph[ph.index.isin(ctx.df_outcomes.index)]
        ctx.modality_dict[PHIKON] = pd.DataFrame({"const": 1.0}, index=ph.index).reindex(ctx.df_outcomes.index)
        ctx.modality_MASK[PHIKON] = ctx.df_outcomes.index.isin(ph.index)
        variant = "f1"
    elif variant in ("f1", "f1cv"):
        if variant == "f1":
            ph = reduced("path_fm_embed_discovery.parquet", "phikon", ctx)
        else:                                            # 768 chiều thô; PCA khớp trong fold
            ph = pd.read_parquet(Path(ds.BASE_DB_DIR) / arg.get("--emb", "path_fm_embed_discovery.parquet"))
            ph = ph[ph.index.isin(ctx.df_outcomes.index)]
        ctx.modality_dict[PHIKON] = ph.reindex(ctx.df_outcomes.index)
        ctx.modality_MASK[PHIKON] = ctx.df_outcomes.index.isin(ph.index)
        print(f"path_phikon n={int(ctx.modality_MASK[PHIKON].sum())}", flush=True)
    else:
        ct = reduced("ct_fmcib_embed_discovery.parquet", "ctfm", ctx)
        pc = ctx.modality_dict["rad_lesion_pc"]
        ctx.modality_dict["rad_lesion_pc"] = pd.concat([pc, ct.reindex(pc.index)], axis=1)
        have = ctx.modality_MASK["rad_lesion_pc"].astype(bool)
        print(f"ctfm ghép vào rad_lesion_pc: {int(ct.index.isin(have.index[have.values]).sum())}/"
              f"{int(have.sum())} bệnh nhân PC có embedding", flush=True)
    train = {"f1": C.train_pdl1fill, "f1cv": train_f1cv, "f2": train_f2}[variant]
    for label in combos:
        mods, ftype = next((m, f) for lb, m, f in PAPER_COMBOS if lb == label)
        add_miss_modality(ctx, mods)                      # miss_pattern tính trên modality GỐC
        full = mods + ["cnl_nlp", MISS] + ([PHIKON] if variant in ("f1", "f1cv") else [])
        data, mask, labels = get_training_data(full, ctx.modality_dict, ctx.modality_MASK,
                                               ctx.df_outcomes)
        if restrict:
            hc = next(m for m in mods if m in C.PATH_CH)
            mask = mask.copy()
            mask[PHIKON] = mask[PHIKON].astype(bool) & mask[hc].astype(bool)
            print(f"f1cov {label}: Phikon dùng ở {int(mask[PHIKON].sum())} bệnh nhân (đã có {hc})", flush=True)
        params = {**B.MODEL_PARAMS, "no_scale": [full.index("cnl_nlp"), full.index(MISS)]}
        tag = arg.get('--tag') or ('f1cov' if restrict else 'f1avail' if avail else variant)
        f = H.OUT / f"{tag}stack__{H.slug(label)}.csv"
        df = (pd.read_csv(f, index_col=0) if f.exists()
              else pd.DataFrame({"label": labels["label"].astype(float)}))
        for seed in seeds:
            col = f"seed_{seed}"
            if col in df.columns:
                continue
            rng = np.random.default_rng(seed)
            summ, _ = train(data, mask, labels.iloc[rng.permutation(len(labels))],
                            make_filter(ftype, ctx), params, folds=B.FOLDS, seed=seed)
            df[col] = pd.Series(summ["score"].astype(float).values, index=summ.index)
            df.to_csv(f)
        n = sum(k.startswith("seed_") for k in df.columns)
        print(f"[{variant}stack] {label}: {n} seed", flush=True)
    print("XONG")


if __name__ == "__main__":
    main()
