"""Step 2/3 CT — batch trích BiomedCLIP embedding cho 1 cohort -> parquet (patient x 512).

python -m experiments.foundation_embed.ct_step2_batch <discovery|rad_valid>
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.evaluate import RESULTS_DIR                         # noqa: E402
from foundation_embed.ct_fm_embedding import embed_volume, load_biomedclip  # noqa: E402

CACHE = Path(__file__).resolve().parent / "cache"
DS = Path(__file__).resolve().parents[2].parent / "datasets"


def main():
    cohort = sys.argv[1] if len(sys.argv) > 1 else "discovery"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mp = pd.read_csv(RESULTS_DIR / "ct_slide_map.csv", dtype=str)
    sub = mp[mp.cohort == cohort]
    model, prep = load_biomedclip(dev)

    recs, fails = [], []
    t0 = time.time()
    for i, r in enumerate(sub.itertuples(), 1):
        try:
            vec, n = embed_volume(r.acc, model, prep, dev, cache_dir=str(CACHE))
        except Exception as e:
            fails.append((r.acc, str(e)[:120])); continue
        if vec is None:
            fails.append((r.acc, "no_lesion/series")); continue
        recs.append((r.patient, vec))
        if i % 40 == 0:
            print(f"  {i}/{len(sub)} ({time.time()-t0:.0f}s)", flush=True)

    by = {}
    for pid, v in recs:
        by.setdefault(pid, []).append(v)
    idx = sorted(by)
    mat = np.vstack([np.mean(by[p], axis=0) for p in idx])
    df = pd.DataFrame(mat, index=idx, columns=[f"bmclip_{j}" for j in range(mat.shape[1])])
    df.index.name = "patient"
    out = DS / f"ct_fm_embed_{cohort}.parquet"
    df.to_parquet(out)
    print(f"[{cohort}] vol={len(sub)} ok={len(recs)} patients={len(idx)} fail={len(fails)} -> {out}")
    if fails:
        print("  fails:", fails[:10])


if __name__ == "__main__":
    main()
