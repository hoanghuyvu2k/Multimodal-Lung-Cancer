"""Gộp đặc trưng mức ô (tile_fm_extract.py) thành một vector / bệnh nhân bằng trung bình.

    python experiments/foundation_embed/tile_pool.py            # cả hai mô hình, cohort discovery

Kết quả: ../datasets/path_{model}_mean_discovery.parquet (index = patient).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_FE = Path(__file__).resolve().parent
TILES = _FE / "cache" / "tiles"
DS = _FE.parents[2] / "datasets"
MAP = _FE.parent / "results" / "fm_slide_map.csv"


def main():
    cohort = sys.argv[1] if len(sys.argv) > 1 else "discovery"
    mp = pd.read_csv(MAP)
    mp = mp[mp.cohort == cohort].astype({"slide": str, "patient": str})
    for model in ("phikon2", "midnight"):
        by, ntile = {}, []
        for r in mp.itertuples():
            f = TILES / model / f"{r.slide}.npy"
            if f.exists():
                a = np.load(f).astype(np.float32)
                by.setdefault(r.patient, []).append(a)
                ntile.append(len(a))
        idx = sorted(by)
        mat = np.vstack([np.concatenate(by[p]).mean(axis=0) for p in idx])
        df = pd.DataFrame(mat, index=idx, columns=[f"{model}_{j}" for j in range(mat.shape[1])])
        df.index.name = "patient"
        out = DS / f"path_{model}_mean_{cohort}.parquet"
        df.to_parquet(out)
        print(f"{model}: {len(idx)} bệnh nhân, {mat.shape[1]} chiều, ô/slide trung vị {int(np.median(ntile))} "
              f"(min {min(ntile)}, max {max(ntile)}) -> {out.name}")


if __name__ == "__main__":
    main()
