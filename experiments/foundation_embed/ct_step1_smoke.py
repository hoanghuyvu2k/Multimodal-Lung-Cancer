"""Step 1 CT smoke — 5 volume discovery."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.evaluate import RESULTS_DIR                         # noqa: E402
from foundation_embed.ct_fm_embedding import embed_volume, load_biomedclip  # noqa: E402


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mp = pd.read_csv(RESULTS_DIR / "ct_slide_map.csv", dtype=str)
    accs = mp[mp.cohort == "discovery"]["acc"].head(5).tolist()
    model, prep = load_biomedclip(dev)
    print(f"{'acc':<10}{'n_slice':>8}{'dim':>6}{'norm':>9}{'sec':>8}")
    for a in accs:
        t0 = time.time()
        vec, n = embed_volume(a, model, prep, dev)
        dt = time.time() - t0
        if vec is None:
            print(f"{a:<10}{'0':>8}  (no lesion/series)")
            continue
        print(f"{a:<10}{n:>8}{vec.shape[0]:>6}{np.linalg.norm(vec):>9.2f}{dt:>8.1f}")


if __name__ == "__main__":
    main()
