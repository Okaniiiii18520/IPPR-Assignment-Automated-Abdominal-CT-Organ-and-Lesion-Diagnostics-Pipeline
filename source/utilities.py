import re
import numpy as np
from pathlib import Path
#=========================================================
#Phase 1: CT Preprocessing & Spatial/Frequency Enhancement
#=========================================================
#Slices are stored as patient_<N>.npz with:
#hu (n, 512, 512) int16  raw Hounsfield Units
#mask (n, 512, 512) uint8  0 = background, 1 = liver, 2 = tumor
#index (n,) original axial slice numbers

#===================================================
#Task 1: Image enhancement & Intensity normalization
#===================================================
#Soft-tissue/liver window (level, width) in HU.
#[-125, 225] HU
LIVER_WINDOW = (50, 350) 
#[-160, 240] HU (used for comparison in the report)
WIDE_WINDOW = (40, 400)
#Return sorted patient ids found in data_dir
def list_patients(data_dir):
    ids = [int(re.search(r"patient_(\d+)\.npz", p.name).group(1))
           for p in Path(data_dir).glob("patient_*.npz")]
    return sorted(ids)
#Load one patient. Returns (hu, mask, index) as arrays
def load_patient(data_dir, pid):
    d = np.load(Path(data_dir) / f"patient_{pid}.npz")
    # The extraction script stored slices rotated 180 deg from the standard radiological
    # view (spine at top). Rotate both image and mask back: anterior up, liver on image left.
    hu = np.rot90(d["hu"], 2, axes=(1, 2)).astype(np.float32)
    mask = np.rot90(d["mask"], 2, axes=(1, 2))
    return np.ascontiguousarray(hu), np.ascontiguousarray(mask), d["index"]
#Yield (pid, slice_no, hu_slice, mask_slice) for every slice of the given patients.
def iter_slices(data_dir, pids):
    for pid in pids:
        hu, mask, idx = load_patient(data_dir, pid)
        for k in range(len(hu)):
            yield pid, int(idx[k]), hu[k], mask[k]
#Clip HU to [level - width/2, level + width/2]. Returns clipped float array and (lo, hi).
def window_hu(hu, level = LIVER_WINDOW[0], width = LIVER_WINDOW[1]):
    lo, hi = level - width / 2.0, level + width / 2.0
    return np.clip(hu, lo, hi), (lo, hi)
#Linear map of arr from [lo, hi] to uint8 [0, 255]:  I' = 255 * (I - lo) / (hi - lo).
def normalize_0_255(arr, lo=None, hi=None):
    lo = float(arr.min()) if lo is None else lo
    hi = float(arr.max()) if hi is None else hi
    out = 255.0 * (arr - lo) / max(hi - lo, 1e-9)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
#Convenience: raw HU slice -> windowed, 0-255 uint8 grayscale image.
def hu_to_gray(hu, level = LIVER_WINDOW[0], width = LIVER_WINDOW[1]):
    clipped, (lo, hi) = window_hu(hu, level, width)
    return normalize_0_255(clipped, lo, hi)
#Patient-level 70/30 split (no slice from a test patient appears in training)
def split_patients(pids, test_frac = 0.30, seed = 42):
    rng = np.random.default_rng(seed)
    pids = np.array(sorted(pids))
    rng.shuffle(pids)
    n_test = int(round(test_frac * len(pids)))
    return sorted(pids[n_test:].tolist()), sorted(pids[:n_test].tolist())