"""Song con cho 21 to hop goc, 5 seed x 10-fold (hoan vi nhan nhu harness chuan), ref-linux-py38. +NLP chi chay cho 2 to hop co Labs.
  run_ref.sh experiments/survival/run_survival_21combos_5seed.py --part=0/3
Ra: experiments/results/ref/multi21/scores/<idx>__<model>.csv (cot seed_*), experiments/results/ref/multi21/surv_part<i>.json
Moi seed: AUC, C-index, log HR/1SD (z-score, hieu chinh age, ecog, albumin, dnlr, liver_mets), KM chi2 (trung vi)."""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from lung_helpers import get_training_data                                              # noqa: E402
from common import benchmarks as B                                                      # noqa: E402
from common.data_setup import load_all                                                  # noqa: E402
from nlp_verify import nlp_run                                                          # noqa: E402,F401
from nlp_verify import verify_nlp_5seed as V                                            # noqa: E402
from nlp_verify.nlp_modality import add_nlp_modality                                    # noqa: E402
from allcombo.run_paper_combos import PAPER_COMBOS, make_filter                         # noqa: E402
from allcombo.run_combo import train as train_dyam                                      # noqa: E402
from allcombo.run_ovo_singlerun import train_ovo_safe                                   # noqa: E402

OUT = ROOT / "experiments" / "results" / "ref" / "multi21"
(OUT / "scores").mkdir(parents=True, exist_ok=True)
COVS = ["age", "ecog", "albumin", "dnlr", "liver_mets"]
clin = V.load_clinical()


def z(s):
    return (s - s.mean()) / s.std()


def cidx(s):
    d = pd.DataFrame({"s": s}).join(clin[["pfs", "pfs_censor"]]).dropna()
    return float(concordance_index(d["pfs"], -d["s"], d["pfs_censor"]))


def loghr(s):
    d = pd.DataFrame({"s": z(s)}).join(clin[COVS + ["pfs", "pfs_censor"]]).dropna()
    return float(CoxPHFitter().fit(d, "pfs", "pfs_censor").params_["s"])


def main():
    part = next((a.split("=")[1] for a in sys.argv if a.startswith("--part=")), "0/1")
    i, n = (int(x) for x in part.split("/"))
    f = OUT / f"surv_part{i}.json"
    done = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    ctx = load_all(); add_nlp_modality(ctx)
    for k, (label, mods, ftype) in enumerate(PAPER_COMBOS):
        if k % n != i or label in done:
            continue
        has_labs = "cnl_dem_labs" in mods
        base = [m for m in mods if m != "cnl_dem_labs"]
        variants = {"dyam": (train_dyam, mods), "ovo": (train_ovo_safe, mods)}
        if has_labs:
            variants.update({"dyam_nlp": (train_dyam, base + ["cnl_nlp"]), "ovo_nlp": (train_ovo_safe, base + ["cnl_nlp"])})
        e = {"idx": k + 1}
        for v, (fn, m) in variants.items():
            data, mask, labels = get_training_data(m, ctx.modality_dict, ctx.modality_MASK, ctx.df_outcomes)
            params = dict(B.MODEL_PARAMS)
            params["no_scale"] = [m.index("cnl_nlp")] if "cnl_nlp" in m else []
            cols, met = {}, {"auc": [], "cindex": [], "loghr": [], "km": []}
            for sd in B.SEEDS:
                sh = labels.iloc[np.random.default_rng(sd).permutation(len(labels))]
                s, _ = fn(data, mask, sh, make_filter(ftype, ctx), params, folds=B.FOLDS, seed=sd)
                sc = pd.Series(s["score"].astype(float).values, index=s.index)
                cols[f"seed_{sd}"] = sc
                met["auc"].append(float(roc_auc_score(s["label"].astype(float), sc)))
                met["cindex"].append(cidx(sc)); met["loghr"].append(loghr(sc)); met["km"].append(V.km_chi2(pd.DataFrame({"score": sc}), clin))
            pd.DataFrame(cols).to_csv(OUT / "scores" / f"{k + 1:02d}__{v}.csv")
            e[v] = met
        done[label] = e
        f.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[surv5] {k + 1} {label}: " + " | ".join(f"{v} auc {np.mean(x['auc']):.3f} C {np.mean(x['cindex']):.3f} logHR {np.mean(x['loghr']):.3f} KM {np.mean(x['km']):.1f}" for v, x in e.items() if v != "idx"), flush=True)
    print("XONG")


if __name__ == "__main__":
    main()
