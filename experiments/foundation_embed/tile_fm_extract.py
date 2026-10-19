"""Trích đặc trưng MỨC Ô (tile) từ tiêu bản gốc bằng hai mô hình nền mở, không cần đăng ký:

    phikon2   owkin/phikon-v2    ViT-L, CLS 1024 chiều
    midnight  kaiko-ai/midnight  ViT-g, CLS + trung bình patch = 3072 chiều

Khác đợt 2026-07 (Phikon v1, lưu sẵn trung bình/slide): ở đây LƯU TỪNG Ô để thử cách gộp có học (attention MIL),
và lấy tối đa 500 ô/slide trong vùng u HALO. Cùng một bộ ô cho cả hai mô hình.

Chạy bằng môi trường có GPU (torch CUDA):
    C:/Users/Admin/miniconda3/envs/doan-env/python.exe experiments/foundation_embed/tile_fm_extract.py [discovery]

Kết quả: experiments/foundation_embed/cache/tiles/{model}/{slide}.npy  (n_ô × D, float16) — tiếp tục được chỗ dở.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_FE = Path(__file__).resolve().parent
sys.path.insert(0, str(_FE.parent))
from foundation_embed import pathology_fm_embedding as P          # noqa: E402

P.MAX_TILES = 500
OUT = _FE / "cache" / "tiles"
MAP = _FE.parent / "results" / "fm_slide_map.csv"
BATCH = 32
NORM = {"phikon2": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "midnight": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])}
REPO = {"phikon2": "owkin/phikon-v2", "midnight": "kaiko-ai/midnight"}


def to_batch(tiles, name, dev):
    arr = np.stack([np.asarray(t.resize((224, 224)), dtype=np.float32) / 255.0 for t in tiles])
    mean, std = (np.array(v, dtype=np.float32) for v in NORM[name])
    x = torch.from_numpy(((arr - mean) / std).transpose(0, 3, 1, 2))
    return x.to(dev).half()


@torch.no_grad()
def embed(model, name, tiles, dev):
    feats = []
    for i in range(0, len(tiles), BATCH):
        h = model(pixel_values=to_batch(tiles[i:i + BATCH], name, dev)).last_hidden_state
        f = h[:, 0] if name == "phikon2" else torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1)
        feats.append(f.float().cpu().numpy())
    return np.concatenate(feats).astype(np.float16)


def main():
    import openslide
    from transformers import AutoModel
    cohort = sys.argv[1] if len(sys.argv) > 1 else "discovery"
    dev = "cuda"
    mp = pd.read_csv(MAP)
    slides = mp[mp.cohort == cohort]["slide"].astype(str).tolist()
    for n in REPO:
        (OUT / n).mkdir(parents=True, exist_ok=True)
    todo = [s for s in slides if not all((OUT / n / f"{s}.npy").exists() for n in REPO)]
    print(f"[{cohort}] {len(slides)} slide, còn {len(todo)}", flush=True)
    if not todo:
        return
    models = {n: AutoModel.from_pretrained(r).to(dev).eval().half() for n, r in REPO.items()}
    t0 = time.time()
    for k, sid in enumerate(todo, 1):
        try:
            slide = openslide.OpenSlide(str(P.SLIDES / f"{sid}.svs"))
            cand = list(P.HALO.glob(f"{sid}_*.annotations"))
            regions, src = P.parse_regions(cand[0]) if cand else ([], "none")
            tiles = P.read_tissue_tiles(slide, P.tile_positions(slide, regions),
                                        np.random.default_rng(int(sid) % (2 ** 32)))
            if not tiles:
                print(f"  {sid}: không có ô ({src})", flush=True)
                continue
            for n, m in models.items():
                np.save(OUT / n / f"{sid}.npy", embed(m, n, tiles, dev))
        except Exception as e:                                    # noqa: BLE001
            print(f"  {sid}: LỖI {type(e).__name__} {str(e)[:120]}", flush=True)
            continue
        if k % 10 == 0 or k == len(todo):
            print(f"  {k}/{len(todo)}  {time.time() - t0:.0f}s  (slide {sid}: {len(tiles)} ô, vùng {src})",
                  flush=True)
    print("XONG")


if __name__ == "__main__":
    main()
