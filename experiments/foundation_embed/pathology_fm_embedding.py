"""Pathology WSI -> Phikon embedding. Tile 20x trong vùng Tumor (HALO) -> Phikon CLS -> mean-pool/slide.

Mirror tinh thần clinical_nlp_embedding.py: sinh vector/bệnh nhân để cắm vào modality_dict (no_scale).
Cache embedding từng slide ra .npy để không tính lại.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

TILE = 256          # px @ level0 (20x); Phikon processor sẽ resize về 224
STRIDE = 256
MAX_TILES = 300     # cap để giới hạn compute trên RTX 3060 (sample nếu vượt)
TISSUE_MEAN_MAX = 220   # bỏ tile nền trắng
BATCH = 32

BASE = Path(__file__).resolve().parents[2].parent / "datasets" / "pathology" / "LUNG_18-193"
SLIDES = BASE / "slides"
HALO = BASE / "halo"


def parse_regions(annotation_path):
    """Trả (list polygon Nx2 @level0, source). Ưu tiên annotation tên chứa 'Tumor'; fallback 'Layer 1'."""
    try:
        root = ET.parse(annotation_path).getroot()
    except Exception:
        return [], "none"
    tumor, other = [], []
    for ann in root.iter("Annotation"):
        name = (ann.get("Name") or "").lower()
        for reg in ann.iter("Region"):
            verts = [(float(v.get("X")), float(v.get("Y"))) for v in reg.iter("V")]
            if len(verts) < 2:
                continue
            poly = np.array(verts)
            if reg.get("Type") == "Rectangle" and len(poly) == 2:  # bbox -> 4 góc
                (x0, y0), (x1, y1) = poly
                poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            (tumor if "tumor" in name else other).append(poly)
    if tumor:
        return tumor, "tumor"
    if other:
        return other, "layer"
    return [], "none"


def tumor_mask(regions, dims, ds=64):
    """Rasterize polygon vào mask nhỏ (level0/ds) bằng cv2.fillPoly."""
    import cv2
    W, H = dims
    mask = np.zeros((H // ds + 1, W // ds + 1), np.uint8)
    for poly in regions:
        cv2.fillPoly(mask, [(poly / ds).astype(np.int32)], 1)
    return mask, ds


def tile_positions(slide, regions):
    """Vị trí top-left (x,y) @level0 nằm trong vùng u."""
    W, H = slide.level_dimensions[0]
    if regions:
        mask, ds = tumor_mask(regions, (W, H))
        xs0 = int(min(p[:, 0].min() for p in regions))
        ys0 = int(min(p[:, 1].min() for p in regions))
        xs1 = int(max(p[:, 0].max() for p in regions))
        ys1 = int(max(p[:, 1].max() for p in regions))
    else:
        mask, ds, xs0, ys0, xs1, ys1 = None, 1, 0, 0, W, H
    pos = []
    for y in range(max(0, ys0), min(H - TILE, ys1), STRIDE):
        for x in range(max(0, xs0), min(W - TILE, xs1), STRIDE):
            if mask is not None:
                cy, cx = (y + TILE // 2) // ds, (x + TILE // 2) // ds
                if cy >= mask.shape[0] or cx >= mask.shape[1] or mask[cy, cx] == 0:
                    continue
            pos.append((x, y))
    return pos


def read_tissue_tiles(slide, positions, rng):
    """Đọc tile, lọc nền trắng; sample tối đa MAX_TILES."""
    if len(positions) > MAX_TILES * 3:
        idx = rng.choice(len(positions), MAX_TILES * 3, replace=False)
        positions = [positions[i] for i in idx]
    tiles = []
    for (x, y) in positions:
        img = slide.read_region((x, y), 0, (TILE, TILE)).convert("RGB")
        arr = np.asarray(img)
        if arr.mean() < TISSUE_MEAN_MAX and arr.std() > 10:
            tiles.append(img)
        if len(tiles) >= MAX_TILES:
            break
    return tiles


def embed_slide(slide_id, model, processor, device, rng, cache_dir=None):
    """Trả (vector 768, n_tiles, source). Cache .npy nếu có cache_dir."""
    import torch
    if cache_dir:
        cp = Path(cache_dir) / f"{slide_id}.npy"
        if cp.exists():
            v = np.load(cp)
            return v, -1, "cache"
    import openslide
    slide = openslide.OpenSlide(str(SLIDES / f"{slide_id}.svs"))
    cand = list(HALO.glob(f"{slide_id}_*.annotations"))
    regions, source = parse_regions(cand[0]) if cand else ([], "none")
    pos = tile_positions(slide, regions)
    tiles = read_tissue_tiles(slide, pos, rng)
    if not tiles:
        return None, 0, source
    feats = []
    for i in range(0, len(tiles), BATCH):
        batch = tiles[i:i + BATCH]
        inp = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inp)
        feats.append(out.last_hidden_state[:, 0, :].cpu().numpy())  # CLS token
    vec = np.concatenate(feats).mean(axis=0)  # mean-pool tiles
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.save(Path(cache_dir) / f"{slide_id}.npy", vec)
    return vec, len(tiles), source


def load_phikon(device):
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained("owkin/phikon")
    model = AutoModel.from_pretrained("owkin/phikon").to(device).eval()
    return model, proc
