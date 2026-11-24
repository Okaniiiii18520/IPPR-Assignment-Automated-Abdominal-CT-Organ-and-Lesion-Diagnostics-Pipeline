import numpy as np
from scipy import stats
from skimage.feature import graycomatrix
from skimage.measure import label, regionprops
#=========================================================
#Phase 3: Feature Extraction & Pattern Recognition
#=========================================================
#Every ROI is a boolean mask (True = pixel inside the region of interest, one connected region)
#Texture features use the windowed 0-255 image, first-order statistics use the raw HU slice

#==========================
#Task 1: Feature Extraction
#==========================
#------------------------------------------
#Section 1: Morphological/Shape Descriptors
#------------------------------------------
#Names of the shape features, in the order returned by shape_features
SHAPE_NAMES = ["area", "perimeter_to_area", "circularity", "compactness", "eccentricity", "solidity"]
#Largest connected component of a mask (the ROI is expected to be one region)
def largest_region(mask):
    lab = label(mask)
    if lab.max() == 0:
        return None
    sizes = np.bincount(lab.ravel())[1:]
    return regionprops((lab == 1 + int(np.argmax(sizes))).astype(np.uint8))[0]
#Shape descriptors of the ROI
#Area = number of pixels, Perimeter-to-Area = P / A (large for thin, ragged shapes)
#Circularity = 4 pi A / P^2 (1 for a circle), Compactness = P^2 / (4 pi A) (1 for a circle, larger = less compact)
#Eccentricity = how elongated the best-fit ellipse is (0 = circle, 1 = line)
#Solidity = A / convex hull area (low for notched or spiky shapes)
def shape_features(mask):
    r = largest_region(mask)
    if r is None or r.area < 3:
        return np.zeros(len(SHAPE_NAMES))
    area, per = float(r.area), max(float(r.perimeter), 1e-6)
    circ = min(4.0 * np.pi * area / per ** 2, 1.0)
    return np.array([area, per / area, circ, 1.0 / max(circ, 1e-6), r.eccentricity, r.solidity])
#---------------------------------------------------------
#Section 2: Grey-Level Co-occurrence Matrix (GLCM) Texture
#---------------------------------------------------------
#Names of the GLCM features, in the order returned by glcm_features
GLCM_NAMES = ["glcm_energy", "glcm_contrast", "glcm_homogeneity", "glcm_correlation", "glcm_entropy"]
#Angles (degrees) used for the co-occurrence matrices
GLCM_ANGLES = (0, 45, 90, 135)
#Number of gray levels after quantisation (256 levels would give a sparse matrix on a small ROI)
GLCM_LEVELS = 32
#Co-occurrence matrices of the ROI for each angle, normalised to probabilities, shape (L, L, n_angles)
#Pixels are quantised to levels 1..L inside the mask; level 0 marks "outside the ROI" and is removed afterwards,
#so only pixel pairs that are both inside the ROI are counted
def roi_glcm(gray, mask, distance = 1, levels = GLCM_LEVELS, angles = GLCM_ANGLES):
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop, m = gray[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    q = (crop.astype(np.int32) * levels // 256) + 1
    q[~m] = 0
    g = graycomatrix(q.astype(np.uint8), [distance], [np.deg2rad(a) for a in angles],
                     levels = levels + 1, symmetric = True, normed = False)[1:, 1:, 0, :].astype(np.float64)
    total = g.sum(axis = (0, 1), keepdims = True)
    return g / np.maximum(total, 1e-12)
#Haralick-style features of one normalised matrix P(i, j)
#Energy = sum P^2 (uniformity), Contrast = sum (i-j)^2 P, Homogeneity = sum P / (1 + (i-j)^2)
#Correlation = sum (i-mu_i)(j-mu_j) P / (sigma_i sigma_j), Entropy = -sum P log2 P
def glcm_from_matrix(P):
    L = P.shape[0]
    i, j = np.meshgrid(np.arange(L), np.arange(L), indexing = "ij")
    energy = np.sum(P ** 2)
    contrast = np.sum((i - j) ** 2 * P)
    homogeneity = np.sum(P / (1.0 + (i - j) ** 2))
    mu_i, mu_j = np.sum(i * P), np.sum(j * P)
    sd_i, sd_j = np.sqrt(np.sum((i - mu_i) ** 2 * P)), np.sqrt(np.sum((j - mu_j) ** 2 * P))
    correlation = np.sum((i - mu_i) * (j - mu_j) * P) / (sd_i * sd_j) if sd_i * sd_j > 1e-9 else 1.0
    nz = P[P > 0]
    entropy = -np.sum(nz * np.log2(nz))
    return np.array([energy, contrast, homogeneity, correlation, entropy])
#GLCM features for every angle, shape (n_angles, 5). Used for the per-angle analysis table in the report
def glcm_features_per_angle(gray, mask):
    P = roi_glcm(gray, mask)
    return np.array([glcm_from_matrix(P[:, :, a]) for a in range(P.shape[2])])
#GLCM features averaged over the four angles (direction-independent, 5 values)
def glcm_features(gray, mask):
    return glcm_features_per_angle(gray, mask).mean(axis = 0)
#---------------------------------------------
#Section 3: First-Order Attenuation Statistics
#---------------------------------------------
#Names of the first-order features, in the order returned by first_order_features
FIRST_ORDER_NAMES = ["mean_hu", "variance_hu", "skewness_hu", "kurtosis_hu"]
#Mean, variance, skewness (asymmetry) and excess kurtosis (tail weight) of the HU values inside the ROI
def first_order_features(hu, mask):
    v = hu[mask].astype(np.float64)
    if v.size < 4 or v.std() < 1e-9:
        return np.array([v.mean() if v.size else 0.0, 0.0, 0.0, 0.0])
    return np.array([v.mean(), v.var(), stats.skew(v), stats.kurtosis(v)])
#-------------------------------------------
#Section 4: Feature vector x = [f1, ..., fd]
#-------------------------------------------
#Names of all features, in the order of the vector returned by extract_features
FEATURE_NAMES = SHAPE_NAMES + GLCM_NAMES + FIRST_ORDER_NAMES
#Full feature vector of one ROI: shape (6) + GLCM (5) + first-order (4) = 15 features
#hu = raw HU slice, gray = windowed 0-255 slice, mask = boolean ROI mask
def extract_features(hu, gray, mask):
    return np.concatenate([shape_features(mask), glcm_features(gray, mask), first_order_features(hu, mask)])
