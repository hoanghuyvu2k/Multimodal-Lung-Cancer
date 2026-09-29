"""Cắt patch 3D lesion-centered 50mm@1mm (chuẩn fmcib) tại local -> npz nhỏ để upload Colab.

python -m experiments.foundation_embed.ct_fmcib_patches <discovery|rad_valid>
Mỗi patch = 50x50x50 int16 HU, đã resample về 1mm isotropic, center tại centroid lesion.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.evaluate import RESULTS_DIR                         # noqa: E402
from foundation_embed.ct_fm_embedding import seg_path           # noqa: E402

DS = Path(__file__).resolve().parents[2].parent / "datasets"
VOLS = DS / "radiology" / "LUNG_18-193" / "volumes"
SIZE_MM = 50


def matched_ct(acc, mask_img):
    for n in (VOLS / acc).glob("SCANS/*/*_volumetric_image.nii"):
        img = sitk.ReadImage(str(n), sitk.sitkInt16)
        if img.GetSize() == mask_img.GetSize():
            return img
    return None


def resample_iso(img, interp):
    sp = img.GetSpacing()
    newsize = [int(round(img.GetSize()[i] * sp[i])) for i in range(3)]
    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing((1.0, 1.0, 1.0))
    r.SetSize(newsize)
    r.SetOutputOrigin(img.GetOrigin())
    r.SetOutputDirection(img.GetDirection())
    r.SetInterpolator(interp)
    r.SetDefaultPixelValue(-1024 if interp == sitk.sitkLinear else 0)
    return r.Execute(img)


def extract_patch(acc):
    sp = seg_path(acc)
    if sp is None:
        return None
    mask = sitk.ReadImage(str(sp))
    ct = matched_ct(acc, mask)
    if ct is None:
        return None
    ct1 = sitk.GetArrayFromImage(resample_iso(ct, sitk.sitkLinear))      # [z,y,x] @1mm
    m1 = sitk.GetArrayFromImage(resample_iso(mask, sitk.sitkNearestNeighbor)) > 0
    if not m1.any():
        return None
    cz, cy, cx = [int(round(c)) for c in np.argwhere(m1).mean(axis=0)]
    h = SIZE_MM // 2
    out = np.full((SIZE_MM, SIZE_MM, SIZE_MM), -1024, np.int16)
    src = ct1[max(0, cz - h):cz - h + SIZE_MM, max(0, cy - h):cy - h + SIZE_MM,
              max(0, cx - h):cx - h + SIZE_MM]
    out[:src.shape[0], :src.shape[1], :src.shape[2]] = src.astype(np.int16)
    return out


def main():
    cohort = sys.argv[1] if len(sys.argv) > 1 else "discovery"
    mp = pd.read_csv(RESULTS_DIR / "ct_slide_map.csv", dtype=str)
    sub = mp[mp.cohort == cohort]
    patches, pids, fails = {}, [], []
    for i, r in enumerate(sub.itertuples(), 1):
        try:
            p = extract_patch(r.acc)
        except Exception as e:
            fails.append((r.acc, str(e)[:80])); continue
        if p is None:
            fails.append((r.acc, "no_patch")); continue
        patches[r.patient] = p
        pids.append(r.patient)
        if i % 40 == 0:
            print(f"  {i}/{len(sub)}", flush=True)
    arr = np.stack([patches[p] for p in pids]).astype(np.int16)
    out = DS / f"fmcib_patches_{cohort}.npz"
    np.savez_compressed(out, patches=arr, patients=np.array(pids))
    mb = out.stat().st_size / 1e6
    print(f"[{cohort}] ok={len(pids)} fail={len(fails)} shape={arr.shape} -> {out} ({mb:.0f}MB)")
    if fails:
        print("  fails:", fails[:6])


if __name__ == "__main__":
    main()
