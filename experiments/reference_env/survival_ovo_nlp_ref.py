"""Sống còn cho BM1 (Rad+IHC-G+Gen+PDL1) x {DyAM, OvO, DyAM+NLP, OvO+NLP}, 5 seed (hoán vị nhãn như harness chuẩn), ref-linux-py38.

    run_ref.sh experiments/reference_env/survival_ovo_nlp_ref.py
Điểm: experiments/results/ref/bm1_surv_scores/<model>.csv ; kết quả: experiments/results/ref/survival_ovo_nlp.json
Chỉ số: C-index, log HR/1SD (z-score, hiệu chỉnh age, ecog, albumin, dnlr, liver_mets), KM chi2 (trung vị) từng seed; hiệu ghép cặp;
bootstrap bệnh nhân ghép cặp trên điểm ensemble; LRT thêm điểm.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy import stats

warnings.filterwarnings("ignore")
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1])); sys.path.insert(0, str(_HERE.parents[2]))
from common import benchmarks as B                                          # noqa: E402
from common.data_setup import load_all                                      # noqa: E402
from nlp_verify import verify_nlp_5seed as V                                # noqa: E402
from nlp_verify.nlp_modality import add_nlp_modality                        # noqa: E402
from allcombo.run_paper_combos import PAPER_COMBOS, make_filter             # noqa: E402
from allcombo.run_combo import train as train_dyam                          # noqa: E402
from allcombo.run_ovo_singlerun import train_ovo_safe                       # noqa: E402
from lung_helpers import get_training_data                                  # noqa: E402

import os
RES = _HERE.parents[1] / "results" / os.environ.get("CHK_OUT", "ref")
RES.mkdir(parents=True, exist_ok=True)
SD = RES / "bm1_surv_scores"; SD.mkdir(parents=True, exist_ok=True)
COVS = ["age", "ecog", "albumin", "dnlr", "liver_mets"]
clin = V.load_clinical()
SEEDS = list(B.SEEDS)


def z(s):
    return (s - s.mean()) / s.std()


def cidx(s):
    d = pd.DataFrame({"s": s}).join(clin[["pfs", "pfs_censor"]]).dropna()
    return float(concordance_index(d["pfs"], -d["s"], d["pfs_censor"]))


def logHR(s):
    d = pd.DataFrame({"s": z(s)}).join(clin[COVS + ["pfs", "pfs_censor"]]).dropna()
    return float(CoxPHFitter().fit(d, "pfs", "pfs_censor").params_["s"])


def main():
    ctx = load_all(); add_nlp_modality(ctx)
    mods0, ftype = next((m, f) for lb, m, f in PAPER_COMBOS if lb == "Rad+IHC-G+Gen+PDL1")
    base = [m for m in mods0 if m != "cnl_dem_labs"]
    MOD = {"dyam": (train_dyam, mods0), "ovo": (train_ovo_safe, mods0), "dyam_nlp": (train_dyam, base + ["cnl_nlp"]), "ovo_nlp": (train_ovo_safe, base + ["cnl_nlp"])}
    res = {"per_seed": {}, "paired": {}, "boot": {}, "lrt": {}}
    sc = {}
    for nm, (fn, mods) in MOD.items():
        data, mask, labels = get_training_data(mods, ctx.modality_dict, ctx.modality_MASK, ctx.df_outcomes)
        params = dict(B.MODEL_PARAMS); params["no_scale"] = [mods.index("cnl_nlp")] if "cnl_nlp" in mods else []
        cols = {}
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            shuffled = labels.iloc[rng.permutation(len(labels))]
            summ, _ = fn(data, mask, shuffled, make_filter(ftype, ctx), params, folds=B.FOLDS, seed=seed)
            cols[f"seed_{seed}"] = pd.Series(summ["score"].astype(float).values, index=summ.index)
            print("[run]", nm, seed, flush=True)
        df = pd.DataFrame(cols); df.to_csv(SD / f"{nm}.csv"); sc[nm] = df
        res["per_seed"][nm] = {"c": [cidx(df[c]) for c in df], "logHR": [logHR(df[c]) for c in df],
                               "km": [V.km_chi2(pd.DataFrame({"score": df[c]}), clin) for c in df]}
        print("[metrics]", nm, {k: round(float(np.mean(v)), 4) for k, v in res["per_seed"][nm].items()}, flush=True)
    for a, b in (("ovo_nlp", "dyam_nlp"), ("ovo_nlp", "ovo"), ("ovo_nlp", "dyam"), ("ovo", "dyam"), ("dyam_nlp", "dyam")):
        r = {}
        for m in ("c", "logHR", "km"):
            d = np.array(res["per_seed"][a][m]) - np.array(res["per_seed"][b][m])
            r[m] = {"mean": round(float(d.mean()), 4), "sd": round(float(d.std(ddof=1)), 4), "n_pos": int((d > 0).sum())}
        res["paired"][f"{a}-{b}"] = r
        # bootstrap ghép cặp trên ensemble
        ea, eb = sc[a].apply(z).mean(axis=1), sc[b].apply(z).mean(axis=1)
        idx = ea.index; rng = np.random.default_rng(0); dc, dh = [], []
        for _ in range(1000):
            i = rng.integers(0, len(idx), len(idx)); ia = idx[i]
            t = clin.loc[ia, ["pfs", "pfs_censor"]].reset_index(drop=True); cv = clin.loc[ia, COVS].reset_index(drop=True)
            try:
                sa, sb = ea.values[i], eb.values[i]
                dc.append(concordance_index(t["pfs"], -sa, t["pfs_censor"]) - concordance_index(t["pfs"], -sb, t["pfs_censor"]))
                fa = pd.concat([pd.Series(sa).pipe(z).rename("s"), cv, t], axis=1).dropna()
                fb = pd.concat([pd.Series(sb).pipe(z).rename("s"), cv, t], axis=1).dropna()
                dh.append(CoxPHFitter().fit(fa, "pfs", "pfs_censor").params_["s"] - CoxPHFitter().fit(fb, "pfs", "pfs_censor").params_["s"])
            except Exception:
                continue
        res["boot"][f"{a}-{b}"] = {"dC_ci": [round(float(x), 4) for x in np.percentile(dc, [2.5, 97.5])], "dC_pos": round(float(np.mean(np.array(dc) > 0)), 3),
                                    "dlogHR_ci": [round(float(x), 4) for x in np.percentile(dh, [2.5, 97.5])], "dlogHR_pos": round(float(np.mean(np.array(dh) > 0)), 3)}
        d = pd.DataFrame({"b": z(eb), "a": z(ea)}).join(clin[COVS + ["pfs", "pfs_censor"]]).dropna()
        m0 = CoxPHFitter().fit(d.drop(columns="a"), "pfs", "pfs_censor"); m1 = CoxPHFitter().fit(d, "pfs", "pfs_censor")
        lr = 2 * (m1.log_likelihood_ - m0.log_likelihood_)
        res["lrt"][f"add_{a}_to_{b}"] = {"LR": round(float(lr), 3), "p": round(float(stats.chi2.sf(lr, 1)), 4), "corr": round(float(np.corrcoef(d["a"], d["b"])[0, 1]), 3)}
        print("[pair]", a, b, r, res["boot"][f"{a}-{b}"], res["lrt"][f"add_{a}_to_{b}"], flush=True)
    (RES / "survival_ovo_nlp.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("XONG")


if __name__ == "__main__":
    main()
