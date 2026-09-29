"""Step 1 smoke — 5 slide discovery: shape đúng, n_tiles hợp lý, thời gian/slide."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.evaluate import RESULTS_DIR                         # noqa: E402
from foundation_embed.pathology_fm_embedding import embed_slide, load_phikon  # noqa: E402


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mp = pd.read_csv(RESULTS_DIR / "fm_slide_map.csv")
    slides = mp[mp.cohort == "discovery"]["slide"].astype(str).head(5).tolist()
    model, proc = load_phikon(dev)
    rng = np.random.default_rng(0)
    print(f"{'slide':<12}{'n_tiles':>8}{'source':>8}{'dim':>6}{'norm':>9}{'sec':>8}")
    for s in slides:
        t0 = time.time()
        vec, n, src = embed_slide(s, model, proc, dev, rng)
        dt = time.time() - t0
        if vec is None:
            print(f"{s:<12}{'0':>8}{src:>8}  (no tiles)")
            continue
        print(f"{s:<12}{n:>8}{src:>8}{vec.shape[0]:>6}{np.linalg.norm(vec):>9.2f}{dt:>8.1f}")


if __name__ == "__main__":
    main()
