"""21 tổ hợp của bài báo gốc trong MÔI TRƯỜNG THAM CHIẾU (ref-linux-py38):
DyAM, OvO, NLP+DyAM, NLP+OvO; mỗi biến thể: 1 lần chạy đơn (seed 42, thứ tự tự nhiên như notebook) + 5 seed (hoán vị nhãn).

Cấu hình (xem document/2026-10-13_reference-training-config/): epochs 125, lr 0.01, alpha 0.001, beta 0.0, cross_modality False,
attention_gate_enabled=False cho đơn nguồn, ctx.rad_filters (l1_strength 0.1), KFold 10 random_state=0, discovery n=247.
NLP: thay Labs (hoặc thêm vào base) bằng cnl_nlp (MiniLM 384-d, no_scale).

    run_ref.sh experiments/reference_env/run_all21_ref.py --part=0/4
Kết quả: experiments/results/ref_all21_part{i}.json (lưu tăng dần, chạy tiếp được).
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))        # experiments/
sys.path.insert(0, str(_HERE.parents[2]))        # code/

import torch, sklearn, scipy, pandas as pd         # noqa: E402
from lung_helpers import get_training_data        # noqa: E402
from common import benchmarks as B                # noqa: E402
from common.data_setup import load_all            # noqa: E402
from nlp_verify.nlp_modality import add_nlp_modality   # noqa: E402
from allcombo.run_paper_combos import PAPER_COMBOS, make_filter  # noqa: E402
from allcombo.run_combo import train as train_dyam     # noqa: E402
from allcombo.run_ovo_singlerun import train_ovo_safe  # noqa: E402

OUT = _HERE.parents[1] / "results"
ENV = (f"ref-linux-py38 py{sys.version.split()[0]} torch {torch.__version__} sklearn {sklearn.__version__} "
       f"numpy {np.__version__} scipy {scipy.__version__} pandas {pd.__version__}")


def auc(summ):
    return float(roc_auc_score(summ["label"].astype(float).values, summ["score"].astype(float).values))


def run(train_fn, mods, ftype, ctx):
    data, mask, labels = get_training_data(mods, ctx.modality_dict, ctx.modality_MASK, ctx.df_outcomes)
    l1 = make_filter(ftype, ctx)
    params = dict(B.MODEL_PARAMS)
    params["no_scale"] = [mods.index("cnl_nlp")] if "cnl_nlp" in mods else []
    if len(mods) == 1:
        params["attention_gate_enabled"] = False       # như notebook gốc cho đơn nguồn
    s, _ = train_fn(data, mask, labels, l1, params, folds=B.FOLDS, seed=42)
    single = auc(s)
    per = []
    for sd in B.SEEDS:
        sh = labels.iloc[np.random.default_rng(sd).permutation(len(labels))]
        s, _ = train_fn(data, mask, sh, l1, params, folds=B.FOLDS, seed=sd)
        per.append(auc(s))
    return {"single_run": single, "mean": float(np.mean(per)), "sd": float(np.std(per, ddof=1)), "per_seed": per}


def main():
    part = next((a.split("=")[1] for a in sys.argv if a.startswith("--part=")), "0/1")
    i, n = (int(x) for x in part.split("/"))
    f = OUT / f"ref_all21_part{i}.json"
    done = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    print("ENV", ENV, flush=True)
    ctx = load_all()
    add_nlp_modality(ctx)
    for k, (label, mods, ftype) in enumerate(PAPER_COMBOS):
        if k % n != i or label in done:
            continue
        base = [m for m in mods if m != "cnl_dem_labs"]
        variants = {"dyam": (train_dyam, mods), "ovo": (train_ovo_safe, mods),
                    "dyam_nlp": (train_dyam, base + ["cnl_nlp"]), "ovo_nlp": (train_ovo_safe, base + ["cnl_nlp"])}
        done[label] = {v: run(fn, m, ftype, ctx) for v, (fn, m) in variants.items()}
        done[label]["_env"] = ENV
        f.write_text(json.dumps(done, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        e = done[label]
        print(f"[ref21] {label}: " + " | ".join(f"{v} 1r {e[v]['single_run']:.4f} 5s {e[v]['mean']:.4f}" for v in ("dyam", "ovo", "dyam_nlp", "ovo_nlp")), flush=True)
    print("XONG")


if __name__ == "__main__":
    main()
