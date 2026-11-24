import cv2
import numpy as np
from scipy import ndimage as ndi
#=========================================================
#Phase 1: CT Preprocessing & Spatial/Frequency Enhancement
#=========================================================
#All image functions take / return 2-D uint8 (0-255) arrays unless noted.

#================================
#Task 2: Spatial Domain Filtering
#================================
#---------------------------------------------------------------------------------
#Section 1: Denoising filters (Gaussian, Median, Bilateral, Anisotropic Diffusion)
#---------------------------------------------------------------------------------
#Standard Gaussian smoothing (blurs edges as well as noise)
def gaussian_smooth(img, sigma = 1.5):
    return cv2.GaussianBlur(img, (0, 0), sigma)
#Median filter (good against impulse noise, moderately edge preserving)
def median_smooth(img, ksize = 5):
    return cv2.medianBlur(img, ksize)
#Edge-preserving bilateral filter (weights = spatial Gaussian * intensity Gaussian)
def bilateral_smooth(img, d = 9, sigma_color = 40, sigma_space = 5):
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)
#Perona-Malik anisotropic diffusion
#dI/dt = div( c(|grad I|) grad I )
#Option 1: c = exp(-(|grad I|/kappa)^2)
#Option 2: c = 1/(1+(|grad I|/kappa)^2)
#The "gamma" value is the explicit time step (0 < gamma <= 0.25 for stability)
#The "kappa" value is the contrast parameter (larger = more smoothing)
#Replace gamma with a smaller value to keep the explicit scheme stable 
def anisotropic_diffusion(img, n_iter = 15, kappa = 20.0, gamma = 0.15, option = 2):
    I = img.astype(np.float32)
    for _ in range(n_iter):
        #Differences to the four neighbours
        P = np.pad(I, 1, mode="edge")
        dN = P[:-2, 1:-1] - I
        dS = P[2:, 1:-1] - I
        dE = P[1:-1, 2:] - I
        dW = P[1:-1, :-2] - I
        if option == 1:
            c = lambda d: np.exp(-(d / kappa) ** 2)
        else:
            c = lambda d: 1.0 / (1.0 + (d / kappa) ** 2)
        I = I + gamma * (c(dN) * dN + c(dS) * dS + c(dE) * dE + c(dW) * dW)
    return np.clip(np.rint(I), 0, 255).astype(np.uint8)
#-----------------------------------------------
#Section 2: Histogram-based contrast enhancement
#-----------------------------------------------
#Global histogram equalization
def histogram_equalize(img):
    return cv2.equalizeHist(img)
#Contrast Limited Adaptive Histogram Equalization (CLAHE)
def clahe_equalize(img, clip_limit = 2.0, tile = (8, 8)):
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile).apply(img)

#====================================
#Task 3: Frequency Domain Enhancement
#====================================
#------------------------------------------------
#Section 1: Fourier Transform & Frequency Filters
#------------------------------------------------
#Centered Two-Dimensional Discrete Fourier Transform (2D-DFT) of a float image
def dft2(img):
    return np.fft.fftshift(np.fft.fft2(img.astype(np.float64)))
#log(1 + |F(u,v)|) of the centered DFT, for visualization
def log_magnitude_spectrum(img):
    return np.log1p(np.abs(dft2(img)))
#Builds a distance grid D(u,v) = sqrt((u-M/2)^2 + (v-N/2)^2) for an MxN image
def _dist_grid(shape):
    rows, cols = shape
    u = np.arange(rows) - rows / 2
    v = np.arange(cols) - cols / 2
    V, U = np.meshgrid(v, u)
    return np.sqrt(U ** 2 + V ** 2)
#Butterworth lowpass: H(u,v) = 1 / (1 + (D(u,v)/D0)^(2n))
#d0 is the cutoff radius in frequency pixels, order n controls roll-off sharpness
def butterworth_lowpass(img, d0=60, order=2):
    D = _dist_grid(img.shape)
    H = 1.0 / (1.0 + (D / d0) ** (2 * order))
    out = np.real(np.fft.ifft2(np.fft.ifftshift(dft2(img) * H)))
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), H
#Homomorphic filter: ln(I) = ln(illumination) + ln(reflectance); a high-emphasis
#Gaussian filter H = (gh-gl)(1-exp(-c D^2/D0^2)) + gl attenuates slow illumination
#(gl<1) and boosts detail (gh>1); the result is exponentiated back
def homomorphic(img, d0 = 30, gamma_l = 0.6, gamma_h = 1.5, c = 1.0):
    f = np.log1p(img.astype(np.float64))
    D = _dist_grid(img.shape)
    H = (gamma_h - gamma_l) * (1 - np.exp(-c * (D ** 2) / (d0 ** 2))) + gamma_l
    g = np.real(np.fft.ifft2(np.fft.ifftshift(dft2(f) * H)))
    out = np.expm1(g)
    out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)
    return np.rint(out).astype(np.uint8), H

#==================
#Task 4: Validation
#==================
#Quality metrics
#Peak Signal-to-Noise Ratio (PSNR)
#PSNR = 10 log10(MAX^2 / MSE) in dB
def psnr(ref, test, max_val=255.0):
    mse = np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(max_val ** 2 / mse)
#Synthetic CT-like noise so PSNR has a clean reference to compare against.
def add_gaussian_noise(img, sigma=10.0, seed=0):
    rng = np.random.default_rng(seed)
    noisy = img.astype(np.float64) + rng.normal(0, sigma, img.shape)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)
#Contrast-to-Noise Ratio (CNR)
#Contrast-to-noise ratio between two regions:
#CNR = |mu_A - mu_B| / sqrt(sigma_A^2 + sigma_B^2)
#mask_a / mask_b are boolean arrays. Returns nan if either region is empty
def cnr(img, mask_a, mask_b):
    a, b = img[mask_a].astype(np.float64), img[mask_b].astype(np.float64)
    if a.size < 10 or b.size < 10:
        return float("nan")
    denom = np.sqrt(a.var() + b.var())
    return float(abs(a.mean() - b.mean()) / denom) if denom > 0 else float("nan")
#Soft-tissue ring just outside the organ (excluding a small gap): the 'surrounding' ROI
def surrounding_ring(organ_mask, width=15, gap=3):
    se = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    m = organ_mask.astype(np.uint8)
    outer = cv2.dilate(m, se(gap + width))
    inner = cv2.dilate(m, se(gap))
    return (outer > 0) & (inner == 0)
#Rough body mask (largest component of HU > thresh, holes filled) to exclude air
def body_mask(hu, thresh = -300):
    b = hu > thresh
    lab, n = ndi.label(b)
    if n == 0:
        return b
    sizes = ndi.sum(b, lab, range(1, n + 1))
    return ndi.binary_fill_holes(lab == (1 + int(np.argmax(sizes))))