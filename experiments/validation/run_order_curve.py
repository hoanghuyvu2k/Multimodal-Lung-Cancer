"""T1 (forward chaining theo số chụp CT — proxy, KHÔNG xác minh là thời gian) và T2 (đường cong học theo cỡ mẫu). Kế hoạch ở ledger.md.

    PYTHONIOENCODING=utf-8 python experiments/validation/run_order_curve.py t1
    PYTHONIOENCODING=utf-8 python experiments/validation/run_order_curve.py t2 stack_nomiss
    PYTHONIOENCODING=utf-8 python experiments/validation/run_order_curve.py t2 dyam
Kết quả: experiments/results/order_curve_{t1,t2_<kind>}.json
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
_EXP = Path(__file__).resolve().parents[1]
_CODE = _EXP.parent
for _p in (str(_CODE), str(_EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
import torch                                                      # noqa: E402
from sklearn.metrics import roc_auc_score                         # noqa: E402
from sklearn.model_selection import KFold                         # noqa: E402

from archloop import harness as H                                 # noqa: E402,F401  (vá scipy)
from archloop import cands as C                                   # noqa: E402
import baselines.model_uniform_avg as mua                         # noqa: E402
from allcombo.run_paper_combos import PAPER_COMBOS, make_filter   # noqa: E402
from common import benchmarks as B                                # noqa: E402
from common import data_setup as ds                               # noqa: E402
from common.data_setup import load_all                            # noqa: E402
from missingness.run_miss import MISS, add_miss_modality          # noqa: E402
from nlp_verify.nlp_modality import add_nlp_modality              # noqa: E402
import lung_helpers as LH                                         # noqa: E402
from lung_helpers import get_training_data, l1_filter_features_list   # noqa: E402

OUT = _CODE / "experiments" / "results"
INIT_SEEDS = [42, 7, 123]
KINDS = ["dyam", "uniform", "stack_miss", "stack_nomiss"]


class Setup:
    def __init__(self, ctx, label, kind):
        mods, ftype = next((m, f) for lb, m, f in PAPER_COMBOS if lb == label)
        add_miss_modality(ctx, mods)                              # miss_pattern từ mask GỐC
        stack = kind.startswith("stack")
        full = mods + (["cnl_nlp"] + ([MISS] if kind == "stack_miss" else []) if stack else [])
        data, mask, labels = get_training_data(full, ctx.modality_dict, ctx.modality_MASK, ctx.df_outcomes)
        data, mask = [d.copy() for d in data], mask.copy()
        if stack and "cnl_pdl1_score" in mask.columns:              # C18: lấp PD-L1 thiếu bằng điểm lâm sàng (không dùng nhãn)
            i = list(mask.columns).index("cnl_pdl1_score")
            cl = C._clinical_pdl1().reindex(data[i].index)
            have = mask["cnl_pdl1_score"].astype(bool)
            use = (~have.values) & cl.notna().values
            data[i].iloc[use, 0] = cl.values[use]
            mask.loc[mask.index[use], "cnl_pdl1_score"] = True
        self.kind, self.data, self.mask, self.labels, self.l1 = kind, data, mask, labels, make_filter(ftype, ctx)
        p = dict(B.MODEL_PARAMS)
        if stack:
            p["no_scale"] = [full.index("cnl_nlp")] + ([full.index(MISS)] if kind == "stack_miss" else [])
        self.params = p

    def fit_eval(self, train_ids, test_ids, init_seed):
        ml = [d.copy() for d in self.data]
        excl = self.labels.index.difference(train_ids)               # loại cả bệnh nhân kiểm tra lẫn tương lai khỏi chọn đặc trưng
        mk = self.mask.copy()
        for pos, filt in self.l1.items():
            try:
                l1_filter_features_list(ml, filt["l1_selection_df"], self.labels, excl, pos, **filt["kwargs"])
            except ValueError:                                           # quá ít bệnh nhân (một lớp) để chọn đặc trưng -> chuyên gia đó coi như không có
                ml[pos] = ml[pos].iloc[:, :1]
                mk.iloc[:, pos] = 0
                self.n_dropped = getattr(self, "n_dropped", 0) + 1
        for pos, d in enumerate(ml):                                      # chọn đặc trưng trả 0 cột -> chuyên gia đó coi như không có (không bỏ cả fold)
            if len(d.columns) == 0:
                ml[pos] = self.data[pos].iloc[:, :1].copy()
                mk.iloc[:, pos] = 0
                self.n_dropped = getattr(self, "n_dropped", 0) + 1
        tr = [d.loc[train_ids].values for d in ml]
        te = [d.loc[test_ids].values for d in ml]
        mtr, mte = mk.loc[train_ids].astype(int).values, mk.loc[test_ids].astype(int).values
        ytr = self.labels.loc[train_ids, "label"]
        if self.kind == "dyam":
            LH.set_global_seed(init_seed)
            clf = LH.MultiModalDynamicModel(**self.params)
        else:
            p = {k: v for k, v in self.params.items() if k != "cross_modality_enabled"}
            clf = mua.MultiModalDynamicModelUncertainty(**{**p, "init_seed": init_seed})
        clf.fit(tr, mtr, ytr)
        return pd.Series(np.asarray(clf.predict_proba(te, mte), dtype=float), index=test_ids)


def avg_scores(S, train_ids, test_ids):
    return pd.concat([S.fit_eval(train_ids, test_ids, s) for s in INIT_SEEDS], axis=1).mean(axis=1)


def boot_ci(y, s, rng, k=2000):
    v = []
    for _ in range(k):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            v.append(roc_auc_score(y[i], s[i]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def t1(ctx):
    cl = LH.get_clinical_table_v2(
        path=f"{ds.BASE_DB_DIR}/18193MSKMINDProjectM-OmnibusInventory_DATA_2021-12-20_1540-WITH-TB-and-SCANNER.csv",
        main_index_col="dmp_pt_id", cohort=pd.read_csv(f"{ds.BASE_DB_DIR}/final_cohort_listing.csv").set_index("main_index").query("cohort == 'discovery'"))
    order = pd.to_numeric(cl["radiology_accession_number"], errors="coerce").sort_values().index
    n = len(order)
    cuts = [int(n * 0.50), int(n * 0.667), int(n * 0.833), n]
    rng = np.random.default_rng(0)
    res = {}
    for label in ("Rad+IHC-G+Gen+PDL1", "Rad+IHC-A+Gen+TMB+PDL1"):
        res[label] = {}
        for kind in KINDS:
            S = Setup(ctx, label, kind)
            blocks, parts = [], []
            for a, b in zip(cuts[:-1], cuts[1:]):
                tr_ids, te_ids = order[:a], order[a:b]
                sc = avg_scores(S, tr_ids, te_ids)
                y = S.labels.loc[te_ids, "label"].values
                blocks.append(float(roc_auc_score(y, sc.values)) if len(np.unique(y)) == 2 else float("nan"))
                parts.append(pd.DataFrame({"s": sc.values, "y": y}, index=te_ids))
            P = pd.concat(parts)
            R = {"auc_pooled": float(roc_auc_score(P.y, P.s)), "ci": boot_ci(P.y.values, P.s.values, rng), "blocks": blocks,
                 "n": int(len(P)), "n_pos": int(P.y.sum())}
            res[label][kind] = R
            print(f"[T1] {label:<26}{kind:<14} AUC gộp (n={R['n']}, không đáp ứng {R['n_pos']}) {R['auc_pooled']:.3f} [{R['ci'][0]:.3f}, {R['ci'][1]:.3f}]  khối: "
                  + ", ".join(f"{x:.3f}" for x in blocks), flush=True)
    (OUT / "order_curve_t1.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def t2(ctx, kind):
    S = Setup(ctx, "Rad+IHC-G+Gen+PDL1", kind)
    ids = S.labels.index
    y_all = S.labels["label"]
    out = {}
    for frac in (0.4, 0.6, 0.8, 1.0):
        per_seed = []
        for seed in (42, 7, 123, 2024, 31337):
            rng = np.random.default_rng(seed)
            oof = pd.Series(np.nan, index=ids)
            for tr, te in KFold(10, shuffle=True, random_state=seed).split(ids):
                tr_ids = ids[tr]
                if frac < 1.0:                                           # lấy phân tầng theo nhãn
                    keep = []
                    for cls in (0.0, 1.0):
                        c = tr_ids[y_all.loc[tr_ids].values == cls]
                        keep.extend(rng.choice(c, int(round(frac * len(c))), replace=False))
                    tr_ids = pd.Index(keep)
                oof.loc[ids[te]] = avg_scores_single(S, tr_ids, ids[te])
            m = oof.notna()
            per_seed.append(float(roc_auc_score(y_all[m], oof[m])))
        out[str(frac)] = {"mean": float(np.mean(per_seed)), "sd": float(np.std(per_seed, ddof=1)), "per_seed": per_seed,
                          "n_train_median": int(round(frac * len(ids) * 0.9))}
        out[str(frac)]["experts_dropped_total"] = int(getattr(S, "n_dropped", 0))
        print(f"[T2] {kind:<13} {int(frac * 100):>3}% (≈ {out[str(frac)]['n_train_median']} bệnh nhân huấn luyện): AUC {out[str(frac)]['mean']:.4f} ± {out[str(frac)]['sd']:.4f}"
              f"  (lần chuyên gia radiomics bị bỏ vì quá ít bệnh nhân: {out[str(frac)]['experts_dropped_total']})", flush=True)
    (OUT / f"order_curve_t2_{kind}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def avg_scores_single(S, train_ids, test_ids):
    return S.fit_eval(train_ids, test_ids, 42)


def main():
    torch.set_num_threads(2)
    ctx = load_all()
    add_nlp_modality(ctx)
    if sys.argv[1] == "t1":
        t1(ctx)
    else:
        t2(ctx, sys.argv[2])


if __name__ == "__main__":
    main()
