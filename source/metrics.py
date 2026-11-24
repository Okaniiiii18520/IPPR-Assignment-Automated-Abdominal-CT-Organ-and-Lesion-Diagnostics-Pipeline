from asyncio import Task
import numpy as np
#======================================================
#Phase 2: Morphological Processing & Organ Segmentation
#======================================================

#=======================
#Task 3: Mask Validation
#=======================
#All masks are boolean / 0-1 arrays of the same shape (True = organ or lesion pixel)
#--------------------------------------------
#Section 1: Dice Similarity Coefficient (DSC)
#--------------------------------------------
#DSC = 2|A ∩ B| / (|A| + |B|), where A = predicted mask and B = ground-truth mask
#Returns empty_value when both masks are empty (nothing to find and nothing found)
def dice_score(pred, gt, empty_value = 1.0):
    pred, gt = pred.astype(bool), gt.astype(bool)
    total = pred.sum() + gt.sum()
    if total == 0:
        return empty_value
    return float(2.0 * np.logical_and(pred, gt).sum() / total)
#------------------------------------------------------
#Section 2: Jaccard Index/Intersection over Union (IoU)
#------------------------------------------------------
#IoU = |A ∩ B|/|A ∪ B|  (related to Dice by IoU = DSC / (2 - DSC))
def iou_score(pred, gt, empty_value = 1.0):
    pred, gt = pred.astype(bool), gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return empty_value
    return float(np.logical_and(pred, gt).sum() / union)
#Mean and standard deviation of a list of scores, ignoring NaN values
def summarize_scores(scores):
    s = np.asarray(scores, dtype = np.float64)
    s = s[~np.isnan(s)]
    if s.size == 0:
        return float("nan"), float("nan")
    return float(s.mean()), float(s.std())

#=================================================
#Phase 3: Feature Extraction & Pattern Recognition
#=================================================
#=============================
#Task 3: Diagnostic Evaluation
#=============================
#Counts of true/false positives and negatives (positive class = 1, e.g. Tumor)
def confusion_counts(y_true, y_pred, positive = 1):
    y_true, y_pred = np.asarray(y_true) == positive, np.asarray(y_pred) == positive
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    return tp, fp, tn, fn
#2x2 confusion matrix laid out as [[TN, FP], [FN, TP]] (rows = true class, cols = predicted class)
def confusion_matrix_2x2(y_true, y_pred, positive = 1):
    tp, fp, tn, fn = confusion_counts(y_true, y_pred, positive)
    return np.array([[tn, fp], [fn, tp]])
#Classification metrics derived from the confusion matrix
#Accuracy = (TP + TN) / all, Sensitivity (Recall) = TP / (TP + FN), Specificity = TN / (TN + FP)
#Precision = TP / (TP + FP), F1 = 2 * Precision * Recall / (Precision + Recall)
def classification_metrics(y_true, y_pred, positive = 1):
    tp, fp, tn, fn = confusion_counts(y_true, y_pred, positive)
    safe = lambda a, b: a / b if b > 0 else 0.0
    sensitivity = safe(tp, tp + fn)
    precision = safe(tp, tp + fp)
    return {
        "accuracy": safe(tp + tn, tp + tn + fp + fn),
        "sensitivity": sensitivity,
        "specificity": safe(tn, tn + fp),
        "precision": precision,
        "f1": safe(2 * precision * sensitivity, precision + sensitivity),
    }
