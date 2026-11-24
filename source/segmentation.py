import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_multiotsu, sobel
from skimage.segmentation import watershed
from skimage.measure import regionprops, label
import enhancement as enh
import utilities as util
#======================================================
#Phase 2: Morphological Processing & Organ Segmentation
#======================================================
#Input images are 2-D uint8 (0-255) windowed slices (see utilities.hu_to_gray)
#All masks are boolean arrays; True = pixel belongs to the organ / lesion

#================================
#Task 1: Morphological Operations
#================================
#-------------------------------------------
#Section 1: Custom Structuring Elements (SE)
#-------------------------------------------
#Disk SE of the given radius (isotropic: grows/shrinks objects equally in all directions)
def disk_se(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
#Elliptical SE with different horizontal (rx) and vertical (ry) radii
#Used to cut thin bridges that are mostly horizontal or vertical (e.g. liver-stomach contact)
def ellipse_se(rx, ry):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rx + 1, 2 * ry + 1))
#Cross-shaped SE (4-connected): removes less than a disk, preserves thin diagonal parts
def cross_se(radius):
    return cv2.getStructuringElement(cv2.MORPH_CROSS, (2 * radius + 1, 2 * radius + 1))
#Line SE of the given length and angle in degrees (0 = horizontal, 90 = vertical)
#Opening with a line SE removes structures that are thinner than the line in that direction
def line_se(length, angle = 0):
    size = length if length % 2 == 1 else length + 1
    se = np.zeros((size, size), np.uint8)
    c = size // 2
    dx, dy = np.cos(np.deg2rad(angle)) * c, -np.sin(np.deg2rad(angle)) * c
    cv2.line(se, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1, 1)
    return se
#-----------------------------------------
#Section 2: Basic Morphological Operations
#-----------------------------------------
#Erosion: a pixel stays only if the whole SE fits inside the mask (shrinks, removes thin links)
def erode(mask, se, iterations = 1):
    return cv2.erode(mask.astype(np.uint8), se, iterations = iterations).astype(bool)
#Dilation: a pixel is set if the SE touches the mask (grows, bridges small gaps)
def dilate(mask, se, iterations = 1):
    return cv2.dilate(mask.astype(np.uint8), se, iterations = iterations).astype(bool)
#Opening (erosion then dilation): disconnects touching organs and removes small specks
def opening(mask, se):
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, se).astype(bool)
#Closing (dilation then erosion): fills small holes and gaps while keeping the outline
def closing(mask, se):
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, se).astype(bool)
#Fill every internal void of the mask (e.g. a dark tumour left as a hole inside the liver mask)
def fill_holes(mask):
    return ndi.binary_fill_holes(mask)
#Remove connected components smaller than min_area pixels
def remove_small_components(mask, min_area):
    lab, n = ndi.label(mask)
    if n == 0:
        return mask.astype(bool)
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    keep = np.isin(lab, 1 + np.where(sizes >= min_area)[0])
    return keep

#========================================
#Task 2: Segmentation Strategy Comparison
#========================================
#--------------------------------------------------------
#Section 1: Shared helpers (spatial prior, preprocessing)
#--------------------------------------------------------
#Expected liver centroid as a fraction of the body bounding box (x from the image left, y from the top)
#Measured as the mean ground-truth liver centroid of the training patients (std about 0.07 in x and y)
LIVER_PRIOR = (0.27, 0.43)
#Plausible median gray level (liver window L50/W350) of liver tissue: rejects fat/air (too dark) and
#contrast-filled bowel or bone (too bright). Non-contrast liver (about 55 HU) maps to about 131
LIVER_GRAY_RANGE = (105, 200)
#Tolerance (std) of the spatial prior; wider than the measured spread to cope with unusual anatomy
PRIOR_STD = (0.12, 0.12)
#Bounding box (y0, y1, x0, x1) of the body mask
def body_bbox(body):
    ys, xs = np.where(body)
    return ys.min(), ys.max(), xs.min(), xs.max()
#Gaussian weight in (0, 1] of how close a point (cy, cx) lies to the expected liver centroid
def prior_weight(cy, cx, bbox):
    y0, y1, x0, x1 = bbox
    fx, fy = (cx - x0) / max(x1 - x0, 1), (cy - y0) / max(y1 - y0, 1)
    return float(np.exp(-0.5 * (((fx - LIVER_PRIOR[0]) / PRIOR_STD[0]) ** 2 + ((fy - LIVER_PRIOR[1]) / PRIOR_STD[1]) ** 2)))
#Expected liver centroid in pixel coordinates (row, col) for this body
def prior_seed(bbox):
    y0, y1, x0, x1 = bbox
    return int(y0 + LIVER_PRIOR[1] * (y1 - y0)), int(x0 + LIVER_PRIOR[0] * (x1 - x0))
#Pick the connected component that is large AND close to the expected liver position
#score = area * spatial prior weight. Returns the chosen component (empty mask if none) and its score
def select_liver_component(mask, bbox):
    lab, n = ndi.label(mask)
    best, best_score = np.zeros(mask.shape, bool), 0.0
    for i in range(1, n + 1):
        comp = lab == i
        cy, cx = ndi.center_of_mass(comp)
        score = comp.sum() * prior_weight(cy, cx, bbox)
        if score > best_score:
            best, best_score = comp, score
    return best, best_score
#Edge-preserving smoothing applied before segmentation (bilateral, chosen in Phase 1)
def smooth_for_segmentation(gray):
    return enh.bilateral_smooth(gray)
#-----------------------------------------------
#Section 2: Strategy A - Multi-Otsu Thresholding
#-----------------------------------------------
#Multi-Otsu thresholds from the histogram of body pixels only (air outside the body would dominate otherwise)
#Otsu picks the thresholds that maximise the between-class variance of the classes
def multiotsu_thresholds(gray, body, classes = 4):
    return threshold_multiotsu(gray[body], classes = classes, nbins = 256)
#Strategy A: Multi-Otsu classes -> morphological cleaning -> spatially selected component
#Every class except the darkest (fat/air) is tried as the "liver class"; the class whose cleaned component
#has the best score (area x spatial prior) wins. Returns the organ mask and the thresholds used
def strategy_a_organ(gray, body, classes = 4, open_r = 5, close_r = 9):
    bbox = body_bbox(body)
    th = multiotsu_thresholds(gray, body, classes)
    labels = np.digitize(gray, th)
    best, best_score = np.zeros(gray.shape, bool), 0.0
    for k in range(1, classes):
        cand = (labels == k) & body
        #opening disconnects the liver from touching organs (stomach, muscle, vessels)
        cand = opening(cand, disk_se(open_r))
        comp, score = select_liver_component(cand, bbox)
        #skip classes whose component does not have a liver-like median intensity
        if comp.any() and not (LIVER_GRAY_RANGE[0] <= np.median(gray[comp]) <= LIVER_GRAY_RANGE[1]):
            continue
        if score > best_score:
            best, best_score = comp, score
    if best.any():
        #closing + hole filling recover the boundary and the dark lesions inside the liver
        best = fill_holes(closing(best, disk_se(close_r)))
        best = opening(best, disk_se(3))
        best, _ = select_liver_component(best, bbox)
    return best, th
#---------------------------------------------------
#Section 3: Strategy B - Marker-Controlled Watershed
#---------------------------------------------------
#Strategy B: watershed on the gradient magnitude, with markers placed from the spatial prior
#Sure-foreground marker = pixels connected to the prior seed whose intensity is close to the liver intensity
#Sure-background marker = outside the body, much darker/brighter than the liver, or far from the seed region
def strategy_b_organ(gray, body, tol = 28, fg_erode = 8, bg_dilate = 15):
    bbox = body_bbox(body)
    sy, sx = prior_seed(bbox)
    #robust liver intensity estimate: median of the body pixels in a window around the prior seed
    win = np.zeros(gray.shape, bool)
    win[max(sy - 15, 0):sy + 16, max(sx - 15, 0):sx + 16] = True
    sample = gray[win & body]
    #only liver-like intensities vote, so fat/air/bone inside the window do not bias the estimate
    sample = sample[(sample >= LIVER_GRAY_RANGE[0]) & (sample <= LIVER_GRAY_RANGE[1])]
    if sample.size < 50:
        return np.zeros(gray.shape, bool), 0.0
    ref = np.median(sample)
    band = body & (np.abs(gray.astype(np.int16) - ref) <= tol)
    band = opening(band, disk_se(3))
    #keep the band component that is large and close to the expected position
    coarse, _ = select_liver_component(band, bbox)
    if not coarse.any():
        return coarse, ref
    #markers: eroded coarse region is certainly liver, far-away / out-of-band pixels are certainly not
    fg = erode(coarse, disk_se(fg_erode))
    if not fg.any():
        fg = coarse
    bg = ~dilate(coarse, disk_se(bg_dilate)) | ~body
    markers = np.zeros(gray.shape, np.int32)
    markers[bg] = 1
    markers[fg] = 2
    grad = sobel(gray.astype(np.float64) / 255.0)
    ws = watershed(grad, markers) == 2
    ws = fill_holes(closing(ws, disk_se(7)))
    ws = opening(ws, disk_se(3))
    ws, _ = select_liver_component(ws, bbox)
    return ws, ref
#-----------------------------------------------
#Section 4: Lesion segmentation inside the organ
#-----------------------------------------------
#Hypo-dense lesion detection: pixels clearly darker than the typical liver intensity, inside the eroded organ
#k = number of robust standard deviations below the liver median, min_drop = minimum drop in gray levels
#floor = darkest allowed lesion gray level (fat in fissures and the gallbladder are darker than any tumour)
#Candidates must also be compact (solidity) so that thin dark vessels are not reported as lesions
def segment_lesion(gray, organ, k = 2.0, min_drop = 10, min_area = 60, edge_r = 8, floor = 95, min_solidity = 0.8):
    if organ.sum() < 500:
        return np.zeros(gray.shape, bool)
    smooth = cv2.GaussianBlur(gray, (0, 0), 1.5)
    #erode the organ so the darker boundary pixels (partial volume, fat) are not taken as lesion
    inner = erode(organ, disk_se(edge_r))
    if inner.sum() < 200:
        return np.zeros(gray.shape, bool)
    vals = smooth[inner].astype(np.float64)
    med = np.median(vals)
    sigma = 1.4826 * np.median(np.abs(vals - med))
    thr = med - max(k * sigma, min_drop)
    cand = inner & (smooth < thr) & (smooth > floor)
    cand = opening(cand, disk_se(2))
    cand = closing(cand, disk_se(3))
    cand = remove_small_components(cand, min_area)
    #shape filter: keep compact blobs only
    out = np.zeros(gray.shape, bool)
    lab = label(cand)
    for r in regionprops(lab):
        if r.solidity >= min_solidity:
            out[lab == r.label] = True
    return out
#------------------------------
#Section 5: Full slice pipeline
#------------------------------
#Run the whole Phase 1 -> Phase 2 chain on one raw HU slice
#strategy is "A" (Multi-Otsu) or "B" (marker-controlled watershed). Returns a dict of masks and images
def segment_slice(hu, strategy = "A"):
    gray = smooth_for_segmentation(util.hu_to_gray(hu))
    body = enh.body_mask(hu)
    if strategy == "A":
        organ, _ = strategy_a_organ(gray, body)
    else:
        organ, _ = strategy_b_organ(gray, body)
    lesion = segment_lesion(gray, organ)
    return {"gray": gray, "body": body, "organ": organ, "lesion": lesion}
