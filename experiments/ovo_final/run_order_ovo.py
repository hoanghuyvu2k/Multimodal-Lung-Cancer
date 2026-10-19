"""W3 — giữ-ngoài theo số chụp (T1) và đường cong học (T2) cho DyAM+NLP và OvO+NLP (cùng đường chạy build_l1).
    python experiments/ovo_final/run_order_ovo.py t1 | t2
Kết quả: results/order_ovo_t1.json, order_ovo_t2.json"""
import json
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXP)); sys.path.insert(0, str(_EXP.parent))

import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402
import torch                                                  # noqa: E402
from sklearn.metrics import roc_auc_score                     # noqa: E402
from sklearn.model_selection import KFold                     # noqa: E402

import lung_helpers as LH                                     # noqa: E402
from validation import run_order_curve as OC                  # noqa: E402
from allcombo.run_paper_combos import PAPER_COMBOS             # noqa: E402
from common import benchmarks as B                            # noqa: E402
from common.data_setup import load_all                        # noqa: E402
from lung_helpers import get_training_data, l1_filter_features_list   # noqa: E402
from nlp_verify.nlp_modality import add_nlp_modality          # noqa: E402
from nlp_verify.nlp_run import build_l1                       # noqa: E402

KINDS = ["dyam_nlp", "ovo_nlp"]
OUT = OC.OUT


class S2:
    def __init__(self, ctx, label, kind):
        mods = next(m for lb, m, f in PAPER_COMBOS if lb == label) + ["cnl_nlp"]
        self.kind = kind
        self.data, self.mask, self.labels = get_training_data(mods, ctx.modality_dict, ctx.modality_MASK, ctx.df_outcomes)
        self.l1 = build_l1(mods, ctx)
        self.params = {**B.MODEL_PARAMS, "no_scale": [mods.index("cnl_nlp")]}
        self.n_dropped = 0

    def fit_eval(self, train_ids, test_ids, init_seed):
        ml = [d.copy() for d in self.data]
        excl = self.labels.index.difference(train_ids)
        mk = self.mask.copy()
        for pos, filt in self.l1.items():
            try:
                l1_filter_features_list(ml, filt["l1_selection_df"], self.labels, excl, pos, **filt["kwargs"])
            except ValueError:
                ml[pos] = ml[pos].iloc[:, :1]; mk.iloc[:, pos] = 0; self.n_dropped += 1
        for pos, d in enumerate(ml):
            if len(d.columns) == 0:
                ml[pos] = self.data[pos].iloc[:, :1].copy(); mk.iloc[:, pos] = 0; self.n_dropped += 1
        tr = [d.loc[train_ids].values for d in ml]; te = [d.loc[test_ids].values for d in ml]
        mtr, mte = mk.loc[train_ids].astype(int).values, mk.loc[test_ids].astype(int).values
        LH.set_global_seed(init_seed)
        p = {k: v for k, v in self.params.items() if k != "cross_modality_enabled"}
        if self.kind == "dyam_nlp":
            clf = LH.MultiModalDynamicModel(**self.params)
        else:
            clf = LH.MultiModalDynamicModelOvO(**p)
        clf.fit(tr, mtr, self.labels.loc[train_ids, "label"])
        return pd.Series(np.asarray(clf.predict_proba(te, mte), dtype=float), index=test_ids)


def t1(ctx):
    import validation.run_order_curve as oc
    cl = LH.get_clinical_table_v2(
        path=f"{oc.ds.BASE_DB_DIR}/18193MSKMINDProjectM-OmnibusInventory_DATA_2021-12-20_1540-WITH-TB-and-SCANNER.csv",
        main_index_col="dmp_pt_id", cohort=pd.read_csv(f"{oc.ds.BASE_DB_DIR}/final_cohort_listing.csv").set_index("main_index").query("cohort == 'discovery'"))
    order = pd.to_numeric(cl["radiology_accession_number"], errors="coerce").sort_values().index
    n = len(order); cuts = [int(n * 0.50), int(n * 0.667), int(n * 0.833), n]
    rng = np.random.default_rng(0); res = {}
    for label in ("Rad+IHC-G+Gen+PDL1", "Rad+IHC-A+Gen+TMB+PDL1"):
        res[label] = {}
        for kind in KINDS:
            S = S2(ctx, label, kind); blocks, parts = [], []
            for a, b in zip(cuts[:-1], cuts[1:]):
                tr_ids, te_ids = order[:a], order[a:b]
                sc = pd.concat([S.fit_eval(tr_ids, te_ids, s) for s in oc.INIT_SEEDS], axis=1).mean(axis=1)
                y = S.labels.loc[te_ids, "label"].values
                blocks.append(float(roc_auc_score(y, sc.values)) if len(np.unique(y)) == 2 else float("nan"))
                parts.append(pd.DataFrame({"s": sc.values, "y": y}, index=te_ids))
            P = pd.concat(parts)
            res[label][kind] = {"auc_pooled": float(roc_auc_score(P.y, P.s)), "ci": oc.boot_ci(P.y.values, P.s.values, rng), "blocks": blocks,
                                "n": int(len(P)), "n_pos": int(P.y.sum()), "experts_dropped": S.n_dropped}
            r = res[label][kind]
            print(f"[T1] {label:<26}{kind:<10} AUC {r['auc_pooled']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] n={r['n']} 21pos? {r['n_pos']}", flush=True)
    (OUT / "order_ovo_t1.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def t2(ctx):
    out = {}
    for kind in KINDS:
        S = S2(ctx, "Rad+IHC-G+Gen+PDL1", kind); ids = S.labels.index; y_all = S.labels["label"]; out[kind] = {}
        for frac in (0.4, 0.6, 0.8, 1.0):
            per_seed = []
            for seed in (42, 7, 123, 2024, 31337):
                rng = np.random.default_rng(seed); oof = pd.Series(np.nan, index=ids)
                for tr, te in KFold(10, shuffle=True, random_state=seed).split(ids):
                    tr_ids = ids[tr]
                    if frac < 1.0:
                        keep = []
                        for cls in (0.0, 1.0):
                            c = tr_ids[y_all.loc[tr_ids].values == cls]
                            keep.extend(rng.choice(c, int(round(frac * len(c))), replace=False))
                        tr_ids = pd.Index(keep)
                    oof.loc[ids[te]] = S.fit_eval(tr_ids, ids[te], 42)
                m = oof.notna(); per_seed.append(float(roc_auc_score(y_all[m], oof[m])))
            out[kind][str(frac)] = {"mean": float(np.mean(per_seed)), "sd": float(np.std(per_seed, ddof=1)), "per_seed": per_seed,
                                    "n_train_median": int(round(frac * len(ids) * 0.9))}
            print(f"[T2] {kind:<9} {int(frac * 100):>3}% (~{out[kind][str(frac)]['n_train_median']}): {out[kind][str(frac)]['mean']:.4f} ± {out[kind][str(frac)]['sd']:.4f}", flush=True)
    (OUT / "order_ovo_t2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    torch.set_num_threads(2)
    ctx = load_all(); add_nlp_modality(ctx)
    t1(ctx) if sys.argv[1] == "t1" else t2(ctx)
    print("XONG")
