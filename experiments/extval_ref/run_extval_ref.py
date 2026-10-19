"""Kiểm chứng NGOÀI (path_valid) cho DyAM / OvO / DyAM+NLP / OvO+NLP trong ref-linux-py38 (2026-10-14).
Kế hoạch định trước: document/2026-10-04_arch-search-loop/ledger.md (mục 2026-10-14). Chạy MỘT lần.

Huấn luyện trên TOÀN BỘ discovery (n=247), dự đoán path_valid (n=71) một lần. Bộ nhớ đệm từ make_cache.py.
  A1 = GLCM + PD-L1 · A2 = GLCM + PD-L1 + Labs6 thô · A3 = GLCM + PD-L1 + NLP6 · P = PD-L1 đơn nguồn (chỉ DyAM)
  1 seed = seed 42, thứ tự tự nhiên · 5 seed = SEEDS với hoán vị thứ tự hàng (như harness).
Đối chiếu nội bộ: CV 10-fold cùng bộ nguồn rút gọn, cùng env, cùng seed.

    run_ref.sh experiments/extval_ref/run_extval_ref.py
Kết quả: experiments/results/ref/extval_ref.json
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))        # experiments/
sys.path.insert(0, str(_HERE.parents[2]))        # code/

import torch, sklearn, scipy, pandas as pd                      # noqa: E402
from sklearn.metrics import roc_auc_score                       # noqa: E402
from lung_helpers import MultiModalDynamicModel, MultiModalDynamicModelOvO, set_global_seed  # noqa: E402
from common import benchmarks as B                              # noqa: E402
from allcombo.run_combo import train as train_dyam              # noqa: E402
from allcombo.run_ovo_singlerun import train_ovo_safe           # noqa: E402

CACHE = Path("/mnt/d/repro_linux/extval_cache.pkl")
OUT = _HERE.parents[1] / "results" / "ref" / "extval_ref.json"
ENV = (f"ref-linux-py38 py{sys.version.split()[0]} torch {torch.__version__} sklearn {sklearn.__version__} "
       f"numpy {np.__version__} scipy {scipy.__version__} pandas {pd.__version__}")
N_BOOT = 5000
ARMS = {"A1 GLCM+PDL1": ["glcm", "pdl1"],
        "A2 GLCM+PDL1+Labs6": ["glcm", "pdl1", "labs6"],
        "A3 GLCM+PDL1+NLP6": ["glcm", "pdl1", "nlp6"],
        "P PDL1": ["pdl1"]}
RUNS = [("A1 GLCM+PDL1", "dyam"), ("A1 GLCM+PDL1", "ovo"),
        ("A2 GLCM+PDL1+Labs6", "dyam"), ("A2 GLCM+PDL1+Labs6", "ovo"),
        ("A3 GLCM+PDL1+NLP6", "dyam"), ("A3 GLCM+PDL1+NLP6", "ovo"),
        ("P PDL1", "dyam")]


def mod_arrays(d, mods):
    """Danh sách ma trận + mặt nạ (n x len(mods)) cho một cohort."""
    X = []
    for m in mods:
        a = np.asarray(d[m], dtype=float)
        X.append(a.reshape(len(a), -1))
    M = np.column_stack([np.asarray(d["m_" + m] if "m_" + m in d else np.ones(len(d["y"]), bool)).astype(int)
                         for m in mods])
    return X, M


def params_for(mods):
    p = dict(B.MODEL_PARAMS)
    p["no_scale"] = [mods.index("nlp6")] if "nlp6" in mods else []
    if "nlp6" in mods:
        assert mods[-1] == "nlp6"                       # no_scale + l_scalers[i] chỉ đúng khi nlp là nguồn cuối
    if len(mods) == 1:
        p["attention_gate_enabled"] = False             # như notebook gốc cho đơn nguồn
    return p


def make_model(kind, p):
    if kind == "dyam":
        return MultiModalDynamicModel(**p)
    return MultiModalDynamicModelOvO(**{k: v for k, v in p.items() if k != "cross_modality_enabled"})


MODEL_DIR = Path("/mnt/d/repro_linux/extval_models")      # model đã train, nạp lại bằng predict_new.py (không phải train lại)


def model_path(arm, kind, tag):
    return MODEL_DIR / (arm.split()[0] + "__" + kind + "__" + tag + ".pt")


def fit_predict(kind, mods, disc, ext, seed, perm=None, save_to=None):
    Xtr, Mtr = mod_arrays(disc, mods)
    Xte, Mte = mod_arrays(ext, mods)
    y = np.asarray(disc["y"], dtype=float)
    if perm is not None:
        Xtr = [a[perm] for a in Xtr]; Mtr = Mtr[perm]; y = y[perm]
    set_global_seed(seed)
    clf = make_model(kind, params_for(mods))
    clf.fit(Xtr, Mtr, y)
    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": clf, "mods": mods, "kind": kind, "seed": seed, "permuted": perm is not None,
                    "env": ENV, "n_train": int(len(y))}, str(save_to))
    return np.asarray(clf.predict_proba(Xte, Mte), dtype=float).ravel()


def cv_auc(kind, mods, disc, seed, perm=None):
    """CV 10-fold nội bộ cùng bộ nguồn; trả AUC (dùng đúng train_dyam / train_ovo_safe của harness)."""
    idx = pd.Index(disc["index"])
    dfs = []
    for m in mods:
        a = np.asarray(disc[m], dtype=float)
        dfs.append(pd.DataFrame(a.reshape(len(a), -1), index=idx))
    mask = pd.DataFrame(np.column_stack([np.asarray(disc["m_" + m] if "m_" + m in disc else np.ones(len(idx), bool))
                                         for m in mods]).astype(bool), index=idx, columns=mods)
    out = pd.DataFrame({"label": np.asarray(disc["y"], dtype=float)}, index=idx)
    if perm is not None:
        out = out.iloc[perm]
    fn = train_dyam if kind == "dyam" else train_ovo_safe
    s, _ = fn(dfs, mask, out, {}, params_for(mods), folds=B.FOLDS, seed=seed)
    return float(roc_auc_score(s["label"].astype(float).values, s["score"].astype(float).values))


def boot_auc(y, s, rng):
    v = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            v.append(roc_auc_score(y[i], s[i]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def boot_delta(y, sa, sb):
    rng = np.random.default_rng(0)
    base = roc_auc_score(y, sa) - roc_auc_score(y, sb)
    d = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            d.append(roc_auc_score(y[i], sa[i]) - roc_auc_score(y[i], sb[i]))
    d = np.array(d)
    return {"delta": float(base), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p": float(2 * min((d <= 0).mean(), (d >= 0).mean()))}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cache = pickle.loads(CACHE.read_bytes())
    disc, ext = cache["disc"], cache["ext"]
    yv = np.asarray(ext["y"], dtype=float)
    rng = np.random.default_rng(0)
    print("ENV", ENV, flush=True)
    print(f"discovery n={len(disc['y'])} · path_valid n={len(yv)} (không đáp ứng {int(yv.sum())}, đáp ứng {int((1-yv).sum())})", flush=True)
    res = {"_env": ENV, "n_ext": int(len(yv)), "n_ext_pos": int(yv.sum()), "note": cache["note"], "runs": {}}
    scores = {}
    n = len(disc["y"])
    for arm, kind in RUNS:
        mods = ARMS[arm]
        key = f"{arm} | {kind}"
        s1 = fit_predict(kind, mods, disc, ext, 42, save_to=model_path(arm, kind, "seed42_natural"))
        per, sc5 = [], []
        for sd in B.SEEDS:
            perm = np.random.default_rng(sd).permutation(n)
            s = fit_predict(kind, mods, disc, ext, sd, perm, save_to=model_path(arm, kind, f"seed{sd}_perm"))
            sc5.append(s)
            per.append(float(roc_auc_score(yv, s)))
        ens = np.mean(sc5, axis=0)
        scores[key] = {"single": s1, "ens": ens}
        cv1 = cv_auc(kind, mods, disc, 42)
        cv5 = []
        for sd in B.SEEDS:
            perm = np.random.default_rng(sd).permutation(n)
            cv5.append(cv_auc(kind, mods, disc, sd, perm))
        r = {"ext_single": float(roc_auc_score(yv, s1)), "ext_single_ci": boot_auc(yv, s1, rng),
             "ext_5seed_per": per, "ext_5seed_mean": float(np.mean(per)), "ext_5seed_sd": float(np.std(per, ddof=1)),
             "ext_ens": float(roc_auc_score(yv, ens)), "ext_ens_ci": boot_auc(yv, ens, rng),
             "cv_single": cv1, "cv_5seed_per": cv5, "cv_5seed_mean": float(np.mean(cv5)), "cv_5seed_sd": float(np.std(cv5, ddof=1))}
        res["runs"][key] = r
        print(f"[{key:<34}] ext 1s {r['ext_single']:.3f} [{r['ext_single_ci'][0]:.3f},{r['ext_single_ci'][1]:.3f}] | "
              f"5s {r['ext_5seed_mean']:.3f}±{r['ext_5seed_sd']:.3f} | ens {r['ext_ens']:.3f} || "
              f"CV 1s {cv1:.3f} 5s {r['cv_5seed_mean']:.3f}±{r['cv_5seed_sd']:.3f}", flush=True)
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    pairs = []
    for a in ("A1 GLCM+PDL1", "A2 GLCM+PDL1+Labs6", "A3 GLCM+PDL1+NLP6"):
        pairs.append((f"OvO − DyAM [{a}]", f"{a} | ovo", f"{a} | dyam"))
    for kind in ("dyam", "ovo"):
        pairs.append((f"NLP6 − Labs6 [{kind}]", f"A3 GLCM+PDL1+NLP6 | {kind}", f"A2 GLCM+PDL1+Labs6 | {kind}"))
        pairs.append((f"NLP6 − không lâm sàng [{kind}]", f"A3 GLCM+PDL1+NLP6 | {kind}", f"A1 GLCM+PDL1 | {kind}"))
    pairs.append(("OvO+NLP6 − DyAM+Labs6", "A3 GLCM+PDL1+NLP6 | ovo", "A2 GLCM+PDL1+Labs6 | dyam"))
    res["deltas"] = {}
    for name, a, b in pairs:
        res["deltas"][name] = {w: boot_delta(yv, scores[a][w], scores[b][w]) for w in ("single", "ens")}
        d1, de = res["deltas"][name]["single"], res["deltas"][name]["ens"]
        print(f"  Δ {name:<36} 1s {d1['delta']:+.3f} [{d1['ci'][0]:+.3f},{d1['ci'][1]:+.3f}] p={d1['p']:.2f} | "
              f"ens {de['delta']:+.3f} [{de['ci'][0]:+.3f},{de['ci'][1]:+.3f}] p={de['p']:.2f}", flush=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez_compressed(OUT.with_suffix(".scores.npz"), y=yv, **{k.replace(" ", "_") + "__" + w: v[w] for k, v in scores.items() for w in v})
    print("XONG ->", OUT)


if __name__ == "__main__":
    main()
