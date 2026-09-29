"""Step 2/3 — Batch trích Phikon embedding cho 1 cohort -> parquet (patient x 768).

python -m experiments.foundation_embed.step2_batch <discovery|path_valid>
Cache .npy mỗi slide -> chạy lại không mất công. Gộp nhiều slide/patient bằng mean.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.evaluate import RESULTS_DIR                         # noqa: E402
from foundation_embed.pathology_fm_embedding import embed_slide, load_phikon  # noqa: E402

CACHE = Path(__file__).resolve().parent / "cache"
DS = Path(__file__).resolve().parents[2].parent / "datasets"


def main():
    cohort = sys.argv[1] if len(sys.argv) > 1 else "discovery"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mp = pd.read_csv(RESULTS_DIR / "fm_slide_map.csv")
    sub = mp[mp.cohort == cohort].copy()
    sub["slide"] = sub["slide"].astype(str)
    sub["patient"] = sub["patient"].astype(str)
    model, proc = load_phikon(dev)
    rng = np.random.default_rng(0)

    recs, fails = [], []
    t0 = time.time()
    for i, r in enumerate(sub.itertuples(), 1):
        try:
            vec, n, src = embed_slide(r.slide, model, proc, dev, rng, cache_dir=str(CACHE))
        except Exception as e:
            fails.append((r.slide, str(e)[:120]))
            continue
        if vec is None:
            fails.append((r.slide, f"no_tiles({src})"))
            continue
        recs.append((r.patient, vec))
        if i % 25 == 0:
            print(f"  {i}/{len(sub)}  ({time.time() - t0:.0f}s)", flush=True)

    # gộp theo patient (mean nếu nhiều slide)
    by = {}
    for pid, v in recs:
        by.setdefault(pid, []).append(v)
    idx = sorted(by)
    mat = np.vstack([np.mean(by[p], axis=0) for p in idx])
    df = pd.DataFrame(mat, index=idx, columns=[f"phikon_{j}" for j in range(mat.shape[1])])
    df.index.name = "patient"
    out = DS / f"path_fm_embed_{cohort}.parquet"
    df.to_parquet(out)
    print(f"[{cohort}] slides={len(sub)} ok={len(recs)} patients={len(idx)} fail={len(fails)} -> {out}")
    if fails:
        print("  fails:", fails[:10])


if __name__ == "__main__":
    main()
