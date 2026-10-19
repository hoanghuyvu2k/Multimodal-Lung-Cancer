"""Chi so song con cho 21 to hop goc x {DyAM, OvO, DyAM+NLP, OvO+NLP} o MOT phan hoach (seed 42, thu tu tu nhien), ref-linux-py38.
Cung duong chay voi bang AUC 21 to hop (run_notebook_repro.py: train_dyam / train_ovo_safe, MODEL_PARAMS, make_filter).
NLP chi chay rieng cho 2 to hop co Labs (NLP thay Labs); cac to hop khac +NLP == khong NLP (khong chay lai).

  run_ref.sh experiments/survival/run_survival_21combos_single.py --part=0/3
Ra: experiments/results/ref/single21/scores/<idx>__<model>.csv ; experiments/results/ref/single21/surv_part<i>.json
Chi so: AUC (kiem tra khop bang AUC), C-index, Cox HR (lung_helpers), KM chi2 (trung vi). tdAUC them bang compute_tdauc_21_single.py."""
import json
import sys
from pathlib import Path

from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from lung_helpers import compute_cindex, compute_multivariate_cox, get_training_data   # noqa: E402
from common import benchmarks as B                                                      # noqa: E402
from common.data_setup import load_all                                                  # noqa: E402
from nlp_verify import nlp_run                                                          # noqa: E402,F401
from nlp_verify import verify_nlp_5seed as V                                            # noqa: E402
from nlp_verify.nlp_modality import add_nlp_modality                                    # noqa: E402
from allcombo.run_paper_combos import PAPER_COMBOS, make_filter                         # noqa: E402
from allcombo.run_combo import train as train_dyam                                      # noqa: E402
from allcombo.run_ovo_singlerun import train_ovo_safe                                   # noqa: E402

OUT = ROOT / "experiments" / "results" / "ref" / "single21"
(OUT / "scores").mkdir(parents=True, exist_ok=True)


def main():
    part = next((a.split("=")[1] for a in sys.argv if a.startswith("--part=")), "0/1")
    i, n = (int(x) for x in part.split("/"))
    f = OUT / f"surv_part{i}.json"
    done = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    ctx = load_all(); add_nlp_modality(ctx)
    clin = V.load_clinical()
    for k, (label, mods, ftype) in enumerate(PAPER_COMBOS):
        if k % n != i or label in done:
            continue
        has_labs = "cnl_dem_labs" in mods
        base = [m for m in mods if m != "cnl_dem_labs"]
        variants = {"dyam": (train_dyam, mods), "ovo": (train_ovo_safe, mods)}
        if has_labs:
            variants.update({"dyam_nlp": (train_dyam, base + ["cnl_nlp"]), "ovo_nlp": (train_ovo_safe, base + ["cnl_nlp"])})
        e = {}
        for v, (fn, m) in variants.items():
            data, mask, labels = get_training_data(m, ctx.modality_dict, ctx.modality_MASK, ctx.df_outcomes)
            params = dict(B.MODEL_PARAMS)
            params["no_scale"] = [m.index("cnl_nlp")] if "cnl_nlp" in m else []
            s, _ = fn(data, mask, labels, make_filter(ftype, ctx), params, folds=B.FOLDS, seed=42)
            s[["label", "score"]].to_csv(OUT / "scores" / f"{k + 1:02d}__{v}.csv")
            try:
                hr = float(compute_multivariate_cox(s, clin).summary.loc["score", "exp(coef)"])
            except Exception:
                hr = float("nan")
            e[v] = {"auc": float(roc_auc_score(s["label"].astype(float), s["score"].astype(float))),
                    "cindex": float(compute_cindex(s, clin)), "cox_hr": hr, "km_chi2": V.km_chi2(s, clin)}
        done[label] = {"idx": k + 1, **e}
        f.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[surv] {k + 1} {label}: " + " | ".join(f"{v} auc {x['auc']:.4f} C {x['cindex']:.3f} HR {x['cox_hr']:.2f} KM {x['km_chi2']:.1f}" for v, x in e.items()), flush=True)
    print("XONG")


if __name__ == "__main__":
    main()
