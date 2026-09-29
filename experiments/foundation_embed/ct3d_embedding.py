"""CT lesion → 3D foundation embedding (MONAI MedicalNet resnet50 pretrained, fallback của fmcib).

Patch 3D quanh centroid lesion → MedicalNet resnet50 (feat 2048). Tái dùng load_ct_and_mask của Phase 2.
"""

from pathlib import Path

import numpy as np

from foundation_embed.ct_fm_embedding import load_ct_and_mask   # noqa: E402

PATCH = 64            # voxel box quanh centroid
HU_LO, HU_HI = -1000, 400


def _crop(ct, center, p=PATCH):
    z, y, x = [int(round(c)) for c in center]
    h = p // 2
    out = np.full((p, p, p), HU_LO, np.float32)
    zs, ys, xs = [slice(max(0, c - h), c - h + p) for c in (z, y, x)]
    src = ct[zs, ys, xs]
    out[:src.shape[0], :src.shape[1], :src.shape[2]] = src
    return out


def embed_volume_3d(acc, model, device, cache_dir=None):
    import torch
    if cache_dir:
        cp = Path(cache_dir) / f"ct3d_{acc}.npy"
        if cp.exists():
            return np.load(cp), -1
    ct, mask = load_ct_and_mask(acc)
    if ct is None or not mask.any():
        return None, 0
    center = np.argwhere(mask).mean(axis=0)          # (z,y,x)
    patch = _crop(ct, center)
    patch = np.clip(patch, HU_LO, HU_HI)
    patch = (patch - patch.mean()) / (patch.std() + 1e-6)   # z-score per patch
    t = torch.tensor(patch).float()[None, None].to(device)  # [1,1,P,P,P]
    with torch.no_grad():
        vec = model(t).cpu().numpy().ravel()
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.save(Path(cache_dir) / f"ct3d_{acc}.npy", vec)
    return vec, 1


def load_medicalnet(device):
    from monai.networks.nets import resnet50
    m = resnet50(spatial_dims=3, n_input_channels=1, feed_forward=False,
                 shortcut_type="B", bias_downsample=False, pretrained=True)
    return m.to(device).eval()
