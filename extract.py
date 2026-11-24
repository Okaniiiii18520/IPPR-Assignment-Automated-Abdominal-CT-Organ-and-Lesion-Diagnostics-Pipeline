"""
Extract a small, shareable slice set from LiTS NIfTI volumes.

Usage:
    python extract_slices.py <folder_with_nii_files> <output_folder>

For every N that has BOTH volume-N.nii and segmentation-N.nii it saves
<output_folder>/patient_N.npz containing:
    hu    : (n, 512, 512) int16   raw Hounsfield Units (no windowing applied)
    mask  : (n, 512, 512) uint8   0 = background, 1 = liver, 2 = tumor
    index : (n,) original axial slice numbers
and writes <output_folder>/manifest.csv. Zip the output folder and share it.
"""
import csv
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

MIN_LIVER_PX = 3000      # skip slices with almost no liver
MIN_TUMOR_PX = 50        # a slice counts as "tumor slice" above this
N_TUMOR = 6              # max tumor slices per patient
N_NORMAL = 4             # max liver-only slices per patient


def pick_evenly(idx, k):
    """Choose up to k indices spread evenly across the list idx."""
    idx = np.asarray(idx)
    if len(idx) <= k:
        return idx
    return idx[np.linspace(0, len(idx) - 1, k).round().astype(int)]


def main(source, dst):
    source, dst = Path(source), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    vols = {int(re.search(r"volume-(\d+)", p.name).group(1)): p
            for p in source.glob("volume-*.nii*")}
    segs = {int(re.search(r"segmentation-(\d+)", p.name).group(1)): p
            for p in source.glob("segmentation-*.nii*")}
    pairs = sorted(set(vols) & set(segs))
    print("volumes:", sorted(vols), "| segmentations:", sorted(segs))
    print("usable pairs:", pairs)
    if not pairs:
        sys.exit("No matching volume/segmentation pairs found.")

    rows = []
    for n in pairs:
        # get_fdata applies the NIfTI scaling, so values are true HU
        vol = nib.load(str(vols[n])).get_fdata(dtype=np.float32)
        seg = np.rint(nib.load(str(segs[n])).get_fdata()).astype(np.uint8)
        if vol.shape != seg.shape:
            print(f"patient {n}: shape mismatch {vol.shape} vs {seg.shape}, skipped")
            continue

        liver_px = (seg >= 1).sum(axis=(0, 1))
        tumor_px = (seg == 2).sum(axis=(0, 1))
        tumor_idx = np.where(tumor_px >= MIN_TUMOR_PX)[0]
        normal_idx = np.where((liver_px >= MIN_LIVER_PX) & (tumor_px == 0))[0]
        chosen = np.sort(np.concatenate([pick_evenly(tumor_idx, N_TUMOR),
                                         pick_evenly(normal_idx, N_NORMAL)]))
        if len(chosen) == 0:
            print(f"patient {n}: no suitable slices, skipped")
            continue

        # rot90(k=3) matches the orientation used in the original repo's display code
        hu = np.stack([np.rot90(vol[:, :, i], 3) for i in chosen]).astype(np.int16)
        mk = np.stack([np.rot90(seg[:, :, i], 3) for i in chosen])
        np.savez_compressed(dst / f"patient_{n}.npz", hu=hu, mask=mk, index=chosen)

        for i in chosen:
            rows.append([n, int(i), int(liver_px[i]), int(tumor_px[i])])
        print(f"patient {n}: {len(chosen)} slices "
              f"({len(pick_evenly(tumor_idx, N_TUMOR))} tumor), "
              f"HU range {hu.min()}..{hu.max()}")

    with open(dst / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient", "slice", "liver_px", "tumor_px"])
        w.writerows(rows)
    print("done ->", dst)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])