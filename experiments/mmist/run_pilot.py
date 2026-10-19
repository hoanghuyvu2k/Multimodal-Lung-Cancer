"""Pilot: OvO vs DyAM on MMIST-LUNG (TCGA+CPTAC), 12-month mortality. NOT part of the paper's ICI cohort.

Usage (from code/): python experiments/mmist/run_pilot.py <data_dir> [n_genes] [seeds]
data_dir holds demographics.csv, diagnosis.csv, tx.csv (git-lfs materialized selected-transcriptomics).
label = 1 means DEAD within 12 months (vital_status_12 == 0 in the source), so AUC is risk-oriented.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lung_helpers import train, train_ovo, get_training_data, prepare_other_modalities  # noqa: E402

data_dir = Path(sys.argv[1])
n_genes = int(sys.argv[2]) if len(sys.argv) > 2 else 200
seeds = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [42]

LEAK = ["demographic.days_to_death", "days_to_last_fup", "vital_status_12", "vital_status_dss"]

demo = pd.read_csv(data_dir / "demographics.csv").set_index("cases.submitter_id")
diag = pd.read_csv(data_dir / "diagnosis.csv").set_index("cases.submitter_id")
tx = pd.read_csv(data_dir / "tx.csv").set_index("cases.submitter_id")

outcomes = pd.DataFrame({"label": (1 - demo["vital_status_12"]).astype(float)})
outcomes.index.name = "main_index"
demo = demo.drop(columns=LEAK)

# unsupervised variance filter only (no labels used)
tx = tx[tx.var().sort_values(ascending=False).index[:n_genes]]

mod_dict, mask = {}, pd.DataFrame(index=outcomes.index)
for name, df in [("clin_demo", demo), ("clin_diag", diag), ("tx", tx)]:
    df = df.copy()
    df.index.name = "main_index"
    df = df.fillna(df.median())
    prepare_other_modalities(mod_dict, df, mask, name)
print({k: (v.shape, int(mask[k].sum())) for k, v in mod_dict.items()})

params = {"epochs": 125, "lr": 0.01, "alpha": 0.001, "beta": 0.0, "cross_modality_enabled": False}
names = ["clin_demo", "clin_diag", "tx"]
combos = [c for r in (1, 2, 3) for c in itertools.combinations(names, r)]

res = {}
for combo in combos:
    data, m, lab = get_training_data(list(combo), mod_dict, mask, outcomes)
    row = {}
    for tag, fn in [("DyAM", train), ("OvO", train_ovo)]:
        aucs = []
        for seed in seeds:
            shuf = lab.iloc[np.random.default_rng(seed).permutation(len(lab))]
            s, _ = fn(data, m, shuf, {}, dict(params), folds=10, seed=seed)
            aucs.append(roc_auc_score(s["label"].astype(float).astype(int), s["score"].astype(float)))
        row[tag] = {"mean": float(np.mean(aucs)), "per_seed": aucs}
    res["+".join(combo)] = row
    print("+".join(combo), {k: round(v["mean"], 4) for k, v in row.items()}, flush=True)

out = Path(__file__).resolve().parents[1] / "results" / "mmist_pilot.json"
out.write_text(json.dumps({"n_genes": n_genes, "seeds": seeds, "n": int(len(outcomes)), "results": res}, indent=1))
print("saved", out)
