"""OvO vs DyAM on the Institut Curie NSCLC 1L-ICI cohort (Captier 2025, Zenodo 14293431, CC-BY-NC-4.0).
Second cohort, NOT external validation (PET not CT, RNA not mutation, no PD-L1, mixed mono/chemo-IO).
Label: best RECIST PR/CR -> 0, SD/POD -> 1 (same convention as the main paper). 23 pts w/o RECIST dropped.
Usage (from code/): python experiments/mmist/run_curie.py <Data_NSCLC_Immuno_Curie dir> [seeds]
"""
import itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lung_helpers import train, train_ovo, get_training_data, prepare_other_modalities  # noqa: E402

d = Path(sys.argv[1])
seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [42, 7, 123, 2024, 31337]

rd = lambda f: pd.read_csv(d / f, index_col=0)
clin = rd("clinical_targets.csv")
lab = clin["Best response"].map({"Complete": 0, "Partial": 0, "Stable": 1, "Progression": 1}).dropna()
outcomes = pd.DataFrame({"label": lab.astype(float)}); outcomes.index.name = "main_index"

tx = rd("transcriptomics.csv")
tx = pd.get_dummies(tx, columns=["Biopsy site"], dtype=float)
raw = {"rad": rd("radiomics.csv"), "path": rd("pathomics.csv"), "tx": tx}

mod, mask = {}, pd.DataFrame(index=outcomes.index)
for k, df in raw.items():
    df = df.loc[df.index.intersection(outcomes.index)].replace([np.inf, -np.inf], np.nan).astype(float)
    df = df.loc[:, df.notna().any()]
    df = df.fillna(df.median())
    df.index.name = "main_index"
    prepare_other_modalities(mod, df, mask, k)
print({k: (v.shape, int(mask[k].sum())) for k, v in mod.items()}, "labelled", len(outcomes))

params = {"epochs": 125, "lr": 0.01, "alpha": 0.001, "beta": 0.0, "cross_modality_enabled": False}
res = {}
for combo in [c for r in (1, 2, 3) for c in itertools.combinations(raw, r)]:
    data, m, y = get_training_data(list(combo), mod, mask, outcomes)
    row = {}
    for tag, fn in [("DyAM", train), ("OvO", train_ovo)]:
        aucs = []
        for seed in seeds:
            shuf = y.iloc[np.random.default_rng(seed).permutation(len(y))]
            s, _ = fn(data, m, shuf, {}, dict(params), folds=10, seed=seed)
            aucs.append(roc_auc_score(s["label"].astype(float).astype(int), s["score"].astype(float)))
        row[tag] = {"mean": float(np.mean(aucs)), "sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0, "per_seed": aucs}
    res["+".join(combo)] = {"n": int(len(y)), **row}
    print("+".join(combo), len(y), {k: f"{v['mean']:.3f}±{v['sd']:.3f}" for k, v in row.items()}, flush=True)

out = Path(__file__).resolve().parents[1] / "results" / "curie_ici_pilot.json"
out.write_text(json.dumps({"seeds": seeds, "results": res}, indent=1)); print("saved", out)
