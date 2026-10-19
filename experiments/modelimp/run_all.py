"""Chạy nối tiếp bốn hướng cải thiện mô hình (kế hoạch: document/2026-10-01_model-improvement).

    2  ensemble theo seed            (rẻ nhất, tính từ điểm đã lưu)
    4  dò siêu tham số               (BM1, 3 seed)
    3  pairwise loss + modality dropout (BM1, BM4, 5 seed)
    1  ComBat radiomics              (BM1, BM3, 5 seed)

    PYTHONIOENCODING=utf-8 python experiments/modelimp/run_all.py [2] [4] [3] [1]
"""
import sys
import time
import traceback
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
_CODE = _EXP.parent
for _p in (str(_CODE), str(_EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sklearn.preprocessing._data as _skd                        # noqa: E402

_orig_yj = _skd.PowerTransformer._yeo_johnson_optimize


def _safe_yj(self, x):
    try:
        return _orig_yj(self, x)
    except Exception:
        return 1.0


_skd.PowerTransformer._yeo_johnson_optimize = _safe_yj

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from sklearn.metrics import roc_auc_score                         # noqa: E402

from allcombo.run_paper_combos import PAPER_COMBOS, make_filter    # noqa: E402
from baselines.model_uniform_avg import train_uniform_avg          # noqa: E402
from common import benchmarks as B                                 # noqa: E402
from common.data_setup import load_all                             # noqa: E402
from common.evaluate import save_result                            # noqa: E402
from modelimp.model_plus import train_plus                         # noqa: E402
from lung_helpers import get_training_data                         # noqa: E402

SCORES = _CODE / "experiments" / "results" / "scores"
BM1, BM3, BM4 = "Rad+IHC-G+Gen+PDL1", "Rad+Gen", "PDL1+Gen"
REF = {BM1: 0.7730, BM3: 0.7160, BM4: 0.7191}
PAYLOAD = {"huong": {}}


def combo(ctx, label):
    mods, ftype = next((m, f) for lb, m, f in PAPER_COMBOS if lb == label)
    data, mask, labels = get_training_data(mods, ctx.modality_dict,
                                           ctx.modality_MASK, ctx.df_outcomes)
    return data, mask, labels, make_filter(ftype, ctx)


def run_seeds(train_fn, ctx, label, seeds=None, **extra):
    """5-seed (hoán vị nhãn) -> AUC từng seed + điểm gộp để làm ensemble."""
    data, mask, labels, l1 = combo(ctx, label)
    params = {**B.MODEL_PARAMS, **extra}
    y = labels["label"].astype(float)
    cols, per_seed = {}, []
    for seed in (seeds or B.SEEDS):
        rng = np.random.default_rng(seed)
        summ, _ = train_fn(data, mask, labels.iloc[rng.permutation(len(labels))],
                           l1, params, folds=B.FOLDS, seed=seed)
        s = pd.Series(summ["score"].astype(float).values, index=summ.index).dropna()
        cols[f"seed_{seed}"] = s
        per_seed.append(float(roc_auc_score(y.loc[s.index], s)))
    sc = pd.DataFrame(cols)
    zs = sc.apply(lambda c: (c - c.mean()) / c.std())            # chuẩn hoá trước khi gộp
    m = zs.mean(axis=1, skipna=True)
    ok = m.notna()
    ens = float(roc_auc_score(y.loc[sc.index][ok], m[ok]))
    return {"mean": float(np.mean(per_seed)), "sd": float(np.std(per_seed, ddof=1)),
            "per_seed": per_seed, "ensemble": ens}


# ── Hướng 2 ────────────────────────────────────────────────────────────────────
def huong_2(ctx):
    """Ensemble theo seed cho cả 21 tổ hợp, tính từ điểm ĐÃ LƯU (không train lại)."""
    rows = {}
    print(f"{'tổ hợp':<30}{'1 seed TB':>12}{'ensemble':>11}{'Δ':>9}")
    for label, _m, _f in PAPER_COMBOS:
        slug = label.replace("+", "_").replace("-", "").replace(" ", "")
        f = SCORES / f"uni_{slug}__PAPER.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, index_col=0)
        seeds = [c for c in df.columns if c.startswith("seed_")]
        y = df["label"]
        # điểm có thể NaN: seed nào bỏ fold 0-feature thì bệnh nhân đó không được chấm
        aucs = []
        for c in seeds:
            ok = df[c].notna()
            if ok.sum() > 10 and y[ok].nunique() > 1:
                aucs.append(roc_auc_score(y[ok], df.loc[ok, c]))
        if not aucs:
            continue
        single = float(np.mean(aucs))
        z = df[seeds].apply(lambda c: (c - c.mean()) / c.std())
        m = z.mean(axis=1, skipna=True)
        ok = m.notna()
        ens = float(roc_auc_score(y[ok], m[ok]))
        rows[label] = {"single_mean": single, "ensemble": ens, "delta": ens - single}
        print(f"{label:<30}{single:>12.4f}{ens:>11.4f}{ens - single:>+9.4f}")
    d = [v["delta"] for v in rows.values()]
    print(f"\nTrung bình Δ = {np.mean(d):+.4f} · số tổ hợp cải thiện: "
          f"{sum(1 for x in d if x > 0)}/{len(d)}")
    return rows


# ── Hướng 4 ────────────────────────────────────────────────────────────────────
def huong_4(ctx):
    """Dò theo từng trục quanh cấu hình hiện tại, BM1, 3 seed."""
    seeds = B.SEEDS[:3]
    grid = [("epochs", e) for e in (75, 200)] + \
           [("lr", v) for v in (0.005, 0.02)] + \
           [("alpha", v) for v in (0.0001, 0.01)] + \
           [("beta", v) for v in (0.001,)]
    out = {}
    base = run_seeds(train_uniform_avg, ctx, BM1, seeds)
    out["base"] = base
    print(f"{'cấu hình':<22}{'AUC':>16}{'ensemble':>11}")
    print(f"{'base (125/0.01/0.001)':<22}{base['mean']:.4f}±{base['sd']:.3f}"
          f"{base['ensemble']:>11.4f}")
    for k, v in grid:
        r = run_seeds(train_uniform_avg, ctx, BM1, seeds, **{k: v})
        out[f"{k}={v}"] = r
        print(f"{k + '=' + str(v):<22}{r['mean']:.4f}±{r['sd']:.3f}{r['ensemble']:>11.4f}"
              f"   Δ={r['mean'] - base['mean']:+.4f}", flush=True)
    return out


# ── Hướng 3 ────────────────────────────────────────────────────────────────────
def huong_3(ctx):
    out = {}
    print(f"{'tổ hợp':<24}{'cấu hình':<22}{'AUC':>16}{'ensemble':>11}{'Δ':>9}")
    for label in (BM1, BM4):
        base = run_seeds(train_uniform_avg, ctx, label)
        out[f"{label}|bce"] = base
        print(f"{label:<24}{'BCE (nền)':<22}{base['mean']:.4f}±{base['sd']:.3f}"
              f"{base['ensemble']:>11.4f}{0.0:>+9.4f}", flush=True)
        for name, kw in (("pairwise", {"loss": "pairwise"}),
                         ("dropout 0.2", {"modality_dropout": 0.2}),
                         ("pairwise+dropout", {"loss": "pairwise",
                                               "modality_dropout": 0.2})):
            r = run_seeds(train_plus, ctx, label, **kw)
            out[f"{label}|{name}"] = r
            print(f"{'':<24}{name:<22}{r['mean']:.4f}±{r['sd']:.3f}{r['ensemble']:>11.4f}"
                  f"{r['mean'] - base['mean']:>+9.4f}", flush=True)
    return out


# ── Hướng 1 ────────────────────────────────────────────────────────────────────
def huong_1(ctx):
    from modelimp.combat_rad import load_harmonized
    ctx_h = load_harmonized()
    out = {}
    print(f"{'tổ hợp':<24}{'gốc':>16}{'ComBat':>16}{'Δ':>9}")
    for label in (BM1, BM3):
        a = run_seeds(train_uniform_avg, ctx, label)
        b = run_seeds(train_uniform_avg, ctx_h, label)
        out[label] = {"goc": a, "combat": b}
        print(f"{label:<24}{a['mean']:.4f}±{a['sd']:.3f}{b['mean']:.4f}±{b['sd']:.3f}"
              f"{b['mean'] - a['mean']:>+9.4f}", flush=True)
    return out


# ── Hướng 5: gộp những gì thắng, kiểm trên nhiều tổ hợp với đủ 5 seed ─────────
BEST = {"epochs": 75, "lr": 0.005, "alpha": 0.0001}


def huong_5(ctx):
    """Hướng 4 chỉ dò trên BM1 với 3 seed -> phải xác nhận cấu hình gộp không phải may mắn."""
    combos = [BM1, BM4, "Rad+IHC-G+Gen", "Rad+IHC-A+Gen+PDL1", "IHC-G+Gen", BM3]
    out = {}
    print(f"{'tổ hợp':<24}{'nền':>16}{'tham số mới':>16}{'+pairwise':>16}{'ens tốt nhất':>14}")
    for label in combos:
        base = run_seeds(train_uniform_avg, ctx, label)
        hp = run_seeds(train_uniform_avg, ctx, label, **BEST)
        hp_pw = run_seeds(train_plus, ctx, label, loss="pairwise", **BEST)
        out[label] = {"base": base, "hp": hp, "hp_pairwise": hp_pw}
        best_ens = max(base["ensemble"], hp["ensemble"], hp_pw["ensemble"])
        print(f"{label:<24}{base['mean']:.4f}±{base['sd']:.3f}{hp['mean']:.4f}±{hp['sd']:.3f}"
              f"{hp_pw['mean']:.4f}±{hp_pw['sd']:.3f}{best_ens:>14.4f}", flush=True)
    return out


def huong_6(ctx):
    """Con số chốt: NLP + miss_pattern (tốt nhất hiện tại) CÓ ensemble theo seed."""
    from missingness.run_miss import MISS, add_miss_modality
    from nlp_verify.nlp_modality import add_nlp_modality
    add_nlp_modality(ctx)
    out = {}
    print(f"{'biến thể':<26}{'AUC 5-seed':>16}{'ensemble':>11}")
    for label in (BM1, "Rad+IHC-G+Gen"):
        mods = next(m for lb, m, _f in PAPER_COMBOS if lb == label)
        base = [m for m in mods if m != "cnl_dem_labs"]
        add_miss_modality(ctx, base)
        for name, extra in (("nền", []), ("+NLP+miss", ["cnl_nlp", MISS])):
            data, mask, labels, l1 = combo(ctx, label)
            if extra:                       # dựng lại tổ hợp có thêm modality
                data, mask, labels = get_training_data(
                    base + extra, ctx.modality_dict, ctx.modality_MASK, ctx.df_outcomes)
            ns = [i for i, c in enumerate(mask.columns) if c in ("cnl_nlp", MISS)]
            params = {"no_scale": ns} if ns else {}
            y = labels["label"].astype(float)
            cols, per = {}, []
            for seed in B.SEEDS:
                rng = np.random.default_rng(seed)
                summ, _ = train_uniform_avg(data, mask,
                                            labels.iloc[rng.permutation(len(labels))],
                                            l1, {**B.MODEL_PARAMS, **params},
                                            folds=B.FOLDS, seed=seed)
                sc = pd.Series(summ["score"].astype(float).values, index=summ.index).dropna()
                cols[f"seed_{seed}"] = sc
                per.append(float(roc_auc_score(y.loc[sc.index], sc)))
            df = pd.DataFrame(cols)
            z = df.apply(lambda c: (c - c.mean()) / c.std()).mean(axis=1, skipna=True)
            ok = z.notna()
            ens = float(roc_auc_score(y.loc[df.index][ok], z[ok]))
            out[f"{label}|{name}"] = {"mean": float(np.mean(per)),
                                      "sd": float(np.std(per, ddof=1)), "ensemble": ens}
            print(f"{label[:12] + ' ' + name:<26}{np.mean(per):.4f}±{np.std(per, ddof=1):.3f}"
                  f"{ens:>11.4f}", flush=True)
    return out


PHASES = [("6", "con số chốt: NLP+miss + ensemble", huong_6),
          ("5", "gộp cấu hình tốt nhất", huong_5),
          ("2", "ensemble theo seed", huong_2),
          ("4", "dò siêu tham số", huong_4),
          ("3", "pairwise loss + modality dropout", huong_3),
          ("1", "ComBat radiomics", huong_1)]


def main(argv):
    wanted = {a for a in argv[1:] if a in {p[0] for p in PHASES}} or {p[0] for p in PHASES}
    ctx = load_all()
    report = []
    for key, desc, fn in PHASES:
        if key not in wanted:
            continue
        print(f"\n{'=' * 92}\nHƯỚNG {key} — {desc}\n{'=' * 92}", flush=True)
        t0 = time.time()
        try:
            PAYLOAD["huong"][key] = fn(ctx)
            report.append((key, "OK", time.time() - t0, ""))
        except Exception as exc:                                   # noqa: BLE001
            traceback.print_exc()
            report.append((key, "LỖI", time.time() - t0, f"{type(exc).__name__}: {exc}"))
        save_result("modelimp", PAYLOAD)
        print(f"[hướng {key}] {report[-1][1]} sau {report[-1][2] / 60:.1f} phút", flush=True)

    print(f"\n{'=' * 92}\nTỔNG KẾT\n{'=' * 92}")
    for key, st, secs, err in report:
        print(f"  hướng {key:<3}{st:<5}{secs / 60:6.1f} phút  {err}")


if __name__ == "__main__":
    import torch
    torch.set_num_threads(2)
    main(sys.argv)
