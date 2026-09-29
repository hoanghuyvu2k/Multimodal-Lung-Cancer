"""uniform_avg vs DyAM(original) trên MỌI tổ hợp >=2 nguồn dữ liệu.

5 domain (Rad đặt đầu để ctx.rad_filters khớp vị trí pc/pl/ln). Inject combo vào B.BENCHMARKS rồi
dùng nguyên run_repeated_cv (seed-permutation, xử lý score) — không sửa lung_helpers.py.

CLI: python -m experiments.allcombo.run_combo <k|reproduce>
  reproduce -> chạy combo DG(=BM4) đối chiếu locked
  2|3|4|5   -> chạy mọi combo có k domain, append results/allcombo.json + scores
"""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.model_selection import KFold                       # noqa: E402
from lung_helpers import (                                      # noqa: E402
    MultiModalDynamicModel, get_summary_df, l1_filter_features_list, set_global_seed,
)
from baselines.model_uniform_avg import train_uniform_avg       # noqa: E402
from common import benchmarks as B                              # noqa: E402


def train(modality_list_in, modality_mask, outcomes, l1_dfs_filter,
          model_params, folds=10, seed=42):
    """DyAM (MultiModalDynamicModel) — sao y vòng fold của lung_helpers.train (cùng KFold
    random_state=0, cùng model/filter) NHƯNG thêm guard 0-feature như uniform_avg để công bằng
    (uniform_avg skip fold khi modality bị L1 về 0; train() gốc thì crash). Combo không filter
    -> trùng khít train() gốc (đã kiểm: DG=BM4 khớp 0.7072)."""
    set_global_seed(seed)
    d = {}
    kf = KFold(n_splits=folds, random_state=0, shuffle=True)
    for fold, (tr, te) in enumerate(kf.split(outcomes.index)):
        mlist = [df.copy(deep=True) for df in modality_list_in]
        train_px, valid_px = outcomes.index[tr], outcomes.index[te]
        for pos, filt in l1_dfs_filter.items():
            l1_filter_features_list(mlist, filt["l1_selection_df"], outcomes, valid_px, pos, **filt["kwargs"])
        if any(len(df.columns) == 0 for df in mlist):
            continue
        clf = MultiModalDynamicModel(**model_params)
        clf.fit([df.loc[train_px].values for df in mlist],
                modality_mask.loc[train_px].astype(int).values,
                outcomes.loc[train_px, "label"])
        vs = clf.predict_proba([df.loc[valid_px].values for df in mlist],
                               modality_mask.loc[valid_px].astype(int).values)
        vy = outcomes.loc[valid_px, "label"].values
        for i, px in enumerate(valid_px):
            d[px] = {"label": vy[i], "score": vs[i], "fold": fold}
    return get_summary_df(d), None
from common.data_setup import load_all                          # noqa: E402
from common.evaluate import (                                   # noqa: E402
    paired_bootstrap, run_repeated_cv, save_result, save_scores, strip_internals,
)

try:
    from common.evaluate import load_result
except ImportError:  # pragma: no cover
    load_result = None

# order cố định, Rad đầu tiên
DOMAINS = {
    "R": ("Rad", ["rad_lesion_pc", "rad_lesion_pl", "rad_lesion_ln"]),
    "P": ("Path", ["path_ihc_glcm"]),
    "G": ("Gen", ["gen_driver_mut_amp"]),
    "D": ("PDL1", ["cnl_pdl1_score"]),
    "L": ("Labs", ["cnl_dem_labs"]),
}
ORDER = ["R", "P", "G", "D", "L"]
# combo trùng benchmark khoá (để đối chiếu)
LOCKED = {"RG": "BM3", "DG": "BM4", "RPGD": "BM1", "RPGDL": "BM2"}


def combos_of_k(k):
    """Trả list mã combo (chuỗi ký hiệu theo ORDER) có k domain."""
    out = []
    for combo in itertools.combinations(ORDER, k):
        out.append("".join(sorted(combo, key=ORDER.index)))
    return out


def inject_benchmark(code):
    """Đưa combo vào B.BENCHMARKS với modality Rad-first; trả key."""
    mods = []
    for c in code:
        mods.extend(DOMAINS[c][1])
    B.BENCHMARKS[code] = {
        "name": "+".join(DOMAINS[c][0] for c in code),
        "modalities": mods,
        "use_rad_filters": "R" in code,
    }
    return code


def run_combo(code, ctx):
    inject_benchmark(code)
    res_u = run_repeated_cv(train_uniform_avg, code, ctx, verbose=False)
    res_d = run_repeated_cv(train, code, ctx, verbose=False)
    delta, lo, hi, p = paired_bootstrap(res_u, res_d)  # uniform - dyam
    save_scores("uniform", code, res_u)
    save_scores("dyam", code, res_d)
    entry = {
        "code": code, "k": len(code),
        "domains": [DOMAINS[c][0] for c in code],
        "uniform": strip_internals(res_u),
        "dyam": strip_internals(res_d),
        "delta_mean": res_u["mean_auc"] - res_d["mean_auc"],
        "paired": {"comparison": "uniform - dyam",
                   "delta_auc": delta, "ci_low": lo, "ci_high": hi, "p_value": p},
    }
    return entry


def load_payload():
    try:
        return load_result("allcombo")
    except Exception:
        return {"domains": {c: DOMAINS[c][0] for c in ORDER}, "combos": {}}


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "reproduce"
    ctx = load_all()

    if arg == "reproduce":
        e = run_combo("DG", ctx)
        u, d = e["uniform"]["mean_auc"], e["dyam"]["mean_auc"]
        # locked 5-seed: uniform (backbone_lock.json)=0.7191, DyAM (baseline.json)=0.7072
        print(f"\nREPRODUCE DG(=BM4): uniform={u:.4f} (locked 0.7191) | "
              f"DyAM={d:.4f} (locked 0.7072)")
        ok = abs(u - 0.7191) < 0.005 and abs(d - 0.7072) < 0.005
        print("  -> KHỚP, harness đúng" if ok else "  -> LỆCH, cần kiểm tra!")
        return

    k = int(arg)
    payload = load_payload()
    codes = combos_of_k(k)
    print(f"\n{'=' * 74}\nk={k}: {len(codes)} combo\n{'=' * 74}")
    print(f"{'combo':<8}{'domains':<26}{'uniform':>10}{'DyAM':>10}{'Δ(u-d)':>10}{'p':>8}{'note':>8}")
    for code in codes:
        e = run_combo(code, ctx)
        payload["combos"][code] = e
        save_result("allcombo", payload)  # append per combo
        note = LOCKED.get(code, "")
        print(f"{code:<8}{'+'.join(e['domains']):<26}"
              f"{e['uniform']['mean_auc']:>10.4f}{e['dyam']['mean_auc']:>10.4f}"
              f"{e['delta_mean']:>+10.4f}{e['paired']['p_value']:>8.3f}{note:>8}", flush=True)
    print("=" * 74)


if __name__ == "__main__":
    main()
