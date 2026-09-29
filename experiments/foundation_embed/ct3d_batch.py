"""Batch 3D MedicalNet embedding cho 1 cohort -> parquet (patient x 2048).
python -m experiments.foundation_embed.ct3d_batch <discovery|rad_valid>
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.evaluate import RESULTS_DIR                         # noqa: E402
from foundation_embed.ct3d_embedding import embed_volume_3d, load_medicalnet  # noqa: E402

CACHE = Path(__file__).resolve().parent / "cache"
DS = Path(__file__).resolve().parents[2].parent / "datasets"


def main():
    cohort = sys.argv[1] if len(sys.argv) > 1 else "discovery"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mp = pd.read_csv(RESULTS_DIR / "ct_slide_map.csv", dtype=str)
    sub = mp[mp.cohort == cohort]
    m = load_medicalnet(dev)
    recs, fails = [], []
    t0 = time.time()
    for i, r in enumerate(sub.itertuples(), 1):
        try:
            v, n = embed_volume_3d(r.acc, m, dev, cache_dir=str(CACHE))
        except Exception as e:
            fails.append((r.acc, str(e)[:100])); continue
        if v is None:
            fails.append((r.acc, "no_lesion")); continue
        recs.append((r.patient, v))
        if i % 40 == 0:
            print(f"  {i}/{len(sub)} ({time.time()-t0:.0f}s)", flush=True)
    by = {}
    for pid, v in recs:
        by.setdefault(pid, []).append(v)
    idx = sorted(by)
    mat = np.vstack([np.mean(by[p], axis=0) for p in idx])
    df = pd.DataFrame(mat, index=idx, columns=[f"mnet_{j}" for j in range(mat.shape[1])])
    df.index.name = "patient"
    out = DS / f"ct3d_fm_embed_{cohort}.parquet"
    df.to_parquet(out)
    print(f"[{cohort}] ok={len(recs)} patients={len(idx)} fail={len(fails)} -> {out}")
    if fails:
        print("  fails:", fails[:8])


if __name__ == "__main__":
    main()
