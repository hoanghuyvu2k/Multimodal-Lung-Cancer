"""Tái tạo con số ~0.8 của bài báo gốc theo ĐÚNG cấu hình Figures-Finalized.ipynb (cell 13/18):
epochs 125, lr 0.01, alpha 0.001, beta 0.0, cross_modality off, KFold 10 shuffle random_state=0,
bộ lọc radiomics l1_strength 0.1 (ctx.rad_filters, KHÔNG dùng clean_rad_filters), 1 lần chạy seed 42 thứ tự tự nhiên.
Rồi cùng 21 tổ hợp: DyAM, OvO, DyAM+NLP, OvO+NLP (NLP thay Labs / thêm vào base), mỗi cái:
  single-run (seed 42, không hoán vị) + 5 seed x 10-fold (hoán vị nhãn).

    PYTHONIOENCODING=utf-8 python experiments/nlp_verify/run_notebook_repro.py --part=0/2
Kết quả: experiments/results/notebook_repro_part{i}.json (lưu tăng dần, chạy tiếp được).
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lung_helpers import get_training_data                  # noqa: E402
from common import benchmarks as B                          # noqa: E402
from common.data_setup import load_all                      # noqa: E402
from nlp_verify import nlp_run                              # noqa: E402,F401  (áp bản vá scipy>=1.14)
from nlp_verify.nlp_modality import add_nlp_modality         # noqa: E402
from allcombo.run_paper_combos import PAPER_COMBOS, make_filter  # noqa: E402
from allcombo.run_combo import train as train_dyam           # noqa: E402
from allcombo.run_ovo_singlerun import train_ovo_safe        # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"


def auc(summ):
    return float(roc_auc_score(summ["label"].astype(float).values, summ["score"].astype(float).values))


def run(train_fn, mods, ftype, ctx):
    data, mask, labels = get_training_data(mods, ctx.modality_dict, ctx.modality_MASK, ctx.df_outcomes)
    l1 = make_filter(ftype, ctx)
    params = dict(B.MODEL_PARAMS)
    params["no_scale"] = [mods.index("cnl_nlp")] if "cnl_nlp" in mods else []
    s, _ = train_fn(data, mask, labels, l1, params, folds=B.FOLDS, seed=42)
    single = auc(s)
    per = []
    for sd in B.SEEDS:
        sh = labels.iloc[np.random.default_rng(sd).permutation(len(labels))]
        s, _ = train_fn(data, mask, sh, l1, params, folds=B.FOLDS, seed=sd)
        per.append(auc(s))
    return {"single_run": single, "mean": float(np.mean(per)), "sd": float(np.std(per, ddof=1)),
            "per_seed": per}


def main():
    part = next((a.split("=")[1] for a in sys.argv if a.startswith("--part=")), "0/1")
    i, n = (int(x) for x in part.split("/"))
    f = OUT / f"notebook_repro_part{i}.json"
    done = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    ctx = load_all()
    add_nlp_modality(ctx)
    for k, (label, mods, ftype) in enumerate(PAPER_COMBOS):
        if k % n != i or label in done:
            continue
        base = [m for m in mods if m != "cnl_dem_labs"]
        variants = {"dyam": (train_dyam, mods), "ovo": (train_ovo_safe, mods),
                    "dyam_nlp": (train_dyam, base + ["cnl_nlp"]), "ovo_nlp": (train_ovo_safe, base + ["cnl_nlp"])}
        done[label] = {v: run(fn, m, ftype, ctx) for v, (fn, m) in variants.items()}
        f.write_text(json.dumps(done, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        e = done[label]
        print(f"[repro] {label}: " + " | ".join(f"{v} 1r {e[v]['single_run']:.4f} 5s {e[v]['mean']:.4f}" for v in e), flush=True)
    print("XONG")


if __name__ == "__main__":
    main()
