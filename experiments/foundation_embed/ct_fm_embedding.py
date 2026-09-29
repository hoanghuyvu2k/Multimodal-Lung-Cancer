"""CT volume + mask -> BiomedCLIP embedding (2.5D). Lát axial qua lesion -> encode_image -> mean-pool/volume.

Chọn series .nii khớp size mask. Window HU soft-tissue. Crop bbox lesion. Cache .npy.
"""

from pathlib import Path

import numpy as np

DS = Path(__file__).resolve().parents[2].parent / "datasets" / "radiology" / "LUNG_18-193"
VOLS = DS / "volumes"
SEGS = DS / "segmentations"

WL, WW = 40, 400          # soft-tissue window (tổn thương mô mềm trong phổi)
MARGIN = 16               # px margin quanh bbox lesion
MAX_SLICES = 40
BATCH = 32


def seg_path(acc):
    for suf in ("nh", "amp"):
        p = SEGS / f"{acc}_{suf}.mha"
        if p.exists():
            return p
    c = list(SEGS.glob(f"{acc}_*.mha"))
    return c[0] if c else None


def load_ct_and_mask(acc):
    import SimpleITK as sitk
    sp = seg_path(acc)
    if sp is None:
        return None, None
    mask = sitk.ReadImage(str(sp))
    msize = mask.GetSize()
    nii = None
    for n in (VOLS / acc).glob("SCANS/*/*_volumetric_image.nii"):
        if sitk.ReadImage(str(n), sitk.sitkInt16).GetSize() == msize:
            nii = n
            break
    if nii is None:
        return None, None
    ct = sitk.GetArrayFromImage(sitk.ReadImage(str(nii), sitk.sitkFloat32))  # [z,y,x] HU
    m = sitk.GetArrayFromImage(mask)                                          # [z,y,x]
    return ct, (m > 0)


def window_to_rgb(sl):
    lo, hi = WL - WW / 2, WL + WW / 2
    x = np.clip((sl - lo) / (hi - lo), 0, 1)
    u8 = (x * 255).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)


def embed_volume(acc, model, preprocess, device, cache_dir=None):
    import torch
    from PIL import Image
    if cache_dir:
        cp = Path(cache_dir) / f"ct_{acc}.npy"
        if cp.exists():
            return np.load(cp), -1
    ct, mask = load_ct_and_mask(acc)
    if ct is None:
        return None, 0
    zsl = np.where(mask.any(axis=(1, 2)))[0]
    if len(zsl) == 0:
        return None, 0
    ys, xs = np.where(mask.any(axis=0))
    y0, y1 = max(0, ys.min() - MARGIN), min(ct.shape[1], ys.max() + MARGIN)
    x0, x1 = max(0, xs.min() - MARGIN), min(ct.shape[2], xs.max() + MARGIN)
    if len(zsl) > MAX_SLICES:
        zsl = zsl[np.linspace(0, len(zsl) - 1, MAX_SLICES).astype(int)]
    imgs = [Image.fromarray(window_to_rgb(ct[z, y0:y1, x0:x1])) for z in zsl]
    feats = []
    for i in range(0, len(imgs), BATCH):
        batch = torch.stack([preprocess(im) for im in imgs[i:i + BATCH]]).to(device)
        with torch.no_grad():
            feats.append(model.encode_image(batch).cpu().numpy())
    vec = np.concatenate(feats).mean(axis=0)
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.save(Path(cache_dir) / f"ct_{acc}.npy", vec)
    return vec, len(imgs)


def load_biomedclip(device):
    import open_clip
    model, prep = open_clip.create_model_from_pretrained(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    return model.to(device).eval(), prep
