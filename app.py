"""
phase1_demo.py - run Phase 1 on one slice and save report figures + a metrics table.

Usage: python phase1_demo.py <data/slices dir> <patient id> <slice position> <out dir>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "source"))
import enhancement as e
import utilities as io

def main(data_dir, pid, pos, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    hu, mask, idx = io.load_patient(data_dir, int(pid))
    hu, mask = hu[int(pos)], mask[int(pos)]

    # ---- 1. windowing + normalization -------------------------------------
    raw_gray = io.normalize_0_255(np.clip(hu, -1024, 1024), -1024, 1024)   # "no windowing"
    liver_win = io.hu_to_gray(hu, *io.LIVER_WINDOW)
    wide_win = io.hu_to_gray(hu, *io.WIDE_WINDOW)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for a, im, t in zip(ax, [raw_gray, wide_win, liver_win],
                        ["Full HU range (-1024..1024)", "Window L40/W400", "Window L50/W350"]):
        a.imshow(im, cmap="gray", vmin=0, vmax=255); a.set_title(t); a.axis("off")
    fig.tight_layout(); fig.savefig(out / "01_windowing.png", dpi=120); plt.close(fig)

    # ---- 2. denoising comparison (with synthetic noise for a PSNR reference) ----
    clean = liver_win
    noisy = e.add_gaussian_noise(clean, sigma=12)
    filt = {
        "Noisy": noisy,
        "Gaussian": e.gaussian_smooth(noisy, 1.5),
        "Median": e.median_smooth(noisy, 5),
        "Bilateral": e.bilateral_smooth(noisy),
        "Anisotropic": e.anisotropic_diffusion(noisy),
    }
    liver, tumor = mask >= 1, mask == 2
    parenchyma = mask == 1
    ring = e.surrounding_ring(liver) & e.body_mask(hu)

    rows = []
    for name, im in filt.items():
        rows.append((name, e.psnr(clean, im),
                     e.cnr(im, liver, ring), e.cnr(im, parenchyma, tumor)))
    fig, ax = plt.subplots(1, 5, figsize=(22, 5))
    for a, (n, im) in zip(ax, filt.items()):
        a.imshow(im, cmap="gray", vmin=0, vmax=255); a.set_title(n); a.axis("off")
    fig.tight_layout(); fig.savefig(out / "02_denoising.png", dpi=110); plt.close(fig)

    # ---- 3. histogram enhancement ----------------------------------------
    smooth = filt["Bilateral"]
    he, cl = e.histogram_equalize(smooth), e.clahe_equalize(smooth, 2.0, (8, 8))
    for name, im in [("Bilateral+HE", he), ("Bilateral+CLAHE", cl)]:
        rows.append((name, e.psnr(clean, im),
                     e.cnr(im, liver, ring), e.cnr(im, parenchyma, tumor)))
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    for j, (im, t) in enumerate([(smooth, "Bilateral"), (he, "+ HE"), (cl, "+ CLAHE")]):
        ax[0, j].imshow(im, cmap="gray", vmin=0, vmax=255); ax[0, j].set_title(t); ax[0, j].axis("off")
        ax[1, j].hist(im.ravel(), 256, range=(0, 255)); ax[1, j].set_title("histogram")
    fig.tight_layout(); fig.savefig(out / "03_hist.png", dpi=110); plt.close(fig)

    # ---- 4. frequency domain ---------------------------------------------
    blp, Hb = e.butterworth_lowpass(noisy, d0=70, order=2)
    homo, Hh = e.homomorphic(liver_win)
    for name, im in [("Butterworth LP", blp), ("Homomorphic", homo)]:
        rows.append((name, e.psnr(clean, im),
                     e.cnr(im, liver, ring), e.cnr(im, parenchyma, tumor)))
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    ax[0, 0].imshow(e.log_magnitude_spectrum(clean), cmap="magma"); ax[0, 0].set_title("log|F| clean")
    ax[0, 1].imshow(e.log_magnitude_spectrum(noisy), cmap="magma"); ax[0, 1].set_title("log|F| noisy")
    ax[0, 2].imshow(Hb, cmap="gray"); ax[0, 2].set_title("BLPF transfer fn (D0=70, n=2)")
    ax[1, 0].imshow(blp, cmap="gray"); ax[1, 0].set_title("Butterworth LP of noisy")
    ax[1, 1].imshow(homo, cmap="gray"); ax[1, 1].set_title("Homomorphic")
    ax[1, 2].imshow(Hh, cmap="gray"); ax[1, 2].set_title("Homomorphic transfer fn")
    for a in ax.ravel(): a.axis("off")
    fig.tight_layout(); fig.savefig(out / "04_frequency.png", dpi=110); plt.close(fig)

    # ---- metrics table ----------------------------------------------------
    print(f"patient {pid} slice {idx[int(pos)]}: liver px={int(liver.sum())} tumor px={int(tumor.sum())}")
    print(f"{'method':18s} {'PSNR(dB)':>9s} {'CNR liver/ring':>15s} {'CNR liver/tumor':>16s}")
    for n, p, c1, c2 in rows:
        print(f"{n:18s} {p:9.2f} {c1:15.2f} {c2:16.2f}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(*sys.argv[1:])
