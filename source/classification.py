import numpy as np
from scipy import ndimage as ndi
from scipy.stats import f_oneway
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
import utilities as util
import features as feat
import segmentation as seg
import metrics as met
#=================================================
#Phase 3: Feature Extraction & Pattern Recognition
#=================================================
#Classify a soft-tissue ROI as Healthy Tissue (class 0) or Tumor (class 1)

#==============================
#Task 2: Pattern Classification
#==============================
#Classical pattern classifiers used only, no pre-trained deep learning models
#Building the ROI dataset with features and labels, training classifiers and evaluating on the held-out test patients
#----------------------------------------
#Section 1: Tumor and Healthy-tissue ROIs
#----------------------------------------
#Three kinds of ROI are collected from every slice that shows enough liver:
#  "tumor"        connected components of the ground-truth tumor mask                      -> class 1
#  "control"      a copy of each tumor's exact shape and size, moved to tumor-free liver   -> class 0
#  "fp_candidate" dark blobs that the Phase 2 lesion detector proposes but are not tumor   -> class 0
#Controls make area and shape identical for both classes, so the classifier cannot win on size alone
#Candidates are the realistic confusers (vessels, fissures, noise) that a detector would have to reject
MIN_ROI_AREA = 30
#Minimum number of liver pixels for a slice to be used
MIN_LIVER_PX = 3000
#Connected components of a mask larger than min_area pixels, each returned as its own boolean mask
def split_components(mask, min_area = MIN_ROI_AREA):
    lab, n = ndi.label(mask)
    return [lab == i for i in range(1, n + 1) if (lab == i).sum() >= min_area]
#Place a copy of the tumor shape at a random position that lies completely inside the allowed region
#Returns None if the shape does not fit anywhere after the given number of attempts
def shape_matched_control(tumor_roi, allowed, rng, tries = 300):
    ys, xs = np.where(tumor_roi)
    shape = tumor_roi[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = shape.shape
    cand = np.argwhere(allowed)
    if len(cand) == 0:
        return None
    for _ in range(tries):
        cy, cx = cand[rng.integers(len(cand))]
        y0, x0 = cy - h // 2, cx - w // 2
        if y0 < 0 or x0 < 0 or y0 + h > allowed.shape[0] or x0 + w > allowed.shape[1]:
            continue
        if np.all(allowed[y0:y0 + h, x0:x0 + w][shape]):
            out = np.zeros(allowed.shape, bool)
            out[y0:y0 + h, x0:x0 + w] = shape
            return out
    return None
#False-positive candidates: relaxed Phase 2 lesion detector run inside the true liver, keeping blobs that barely overlap the ground-truth tumor (less than 10 % of their area)
def false_positive_candidates(gray, liver, tumor):
    cand = seg.segment_lesion(gray, liver, k = 1.5, min_drop = 8, min_area = MIN_ROI_AREA, min_solidity = 0.0)
    near_tumor = seg.dilate(tumor, seg.disk_se(3))
    return [c for c in split_components(cand) if (c & near_tumor).sum() < 0.10 * c.sum()]
#All ROIs of one slice: list of (mask, class label, kind)
def slice_rois(gray, mask, rng):
    liver, tumor = mask >= 1, mask == 2
    rois = []
    healthy = seg.erode(liver & ~seg.dilate(tumor, seg.disk_se(10)), seg.disk_se(5))
    for t in split_components(tumor):
        rois.append((t, 1, "tumor"))
        c = shape_matched_control(t, healthy, rng)
        if c is not None:
            rois.append((c, 0, "control"))
            healthy = healthy & ~seg.dilate(c, seg.disk_se(3))
    for c in false_positive_candidates(gray, liver, tumor):
        rois.append((c, 0, "fp_candidate"))
    return rois
#----------------------------------------
#Section 2: Dataset for a set of patients
#----------------------------------------
#Feature matrix X (n_rois, 15), labels y (0 = healthy, 1 = tumor), patient id of every ROI (needed for grouped
#cross-validation, so that no patient is split across training and validation) and a list of (patient, slice, kind)
def build_roi_dataset(data_dir, pids, seed = 0):
    rng = np.random.default_rng(seed)
    X, y, groups, meta = [], [], [], []
    for pid, sl, hu, m in util.iter_slices(data_dir, pids):
        if (m >= 1).sum() < MIN_LIVER_PX:
            continue
        gray = util.hu_to_gray(hu)
        for roi, label, kind in slice_rois(gray, m, rng):
            X.append(feat.extract_features(hu, gray, roi))
            y.append(label)
            groups.append(pid)
            meta.append((pid, sl, kind))
    return np.array(X), np.array(y), np.array(groups), meta

#==============================
#Task 2: Pattern Classification
#==============================
#----------------------
#Section 1: Classifiers
#----------------------
#Candidate classical classifiers and the hyper-parameters searched for each one
#Every classifier is a Pipeline: z-score normalisation (fitted on the training data only) -> classifier
#class_weight = "balanced" compensates for the unequal number of healthy and tumor ROIs
CLASSIFIERS = {
    "svm": (SVC(kernel = "rbf", class_weight = "balanced"), {"clf__C": [0.5, 1, 4, 16], "clf__gamma": ["scale", 0.01, 0.05]}),
    "knn": (KNeighborsClassifier(), {"clf__n_neighbors": [3, 5, 9, 15], "clf__weights": ["uniform", "distance"]}),
    "rf": (RandomForestClassifier(n_estimators = 200, class_weight = "balanced", random_state = 0), {"clf__max_depth": [3, 5, None], "clf__min_samples_leaf": [1, 3]}),
}
#Pipeline (scaler + classifier) for the given classifier name
def make_pipeline(name):
    return Pipeline([("scaler", StandardScaler()), ("clf", CLASSIFIERS[name][0])])
#------------------------------------------------------
#Section 2: 5-fold cross-validation on the training set
#------------------------------------------------------
#Stratified 5-fold split that keeps all ROIs of a patient in the same fold (no patient leakage between folds)
def grouped_folds(n_splits = 5, seed = 0):
    return StratifiedGroupKFold(n_splits = n_splits, shuffle = True, random_state = seed)
#Choose the hyper-parameters of one classifier with 5-fold grouped cross-validation (score = F1 of the tumor class)
def tune_classifier(name, X, y, groups, seed = 0):
    search = GridSearchCV(make_pipeline(name), CLASSIFIERS[name][1], scoring = "f1",
                          cv = grouped_folds(5, seed).split(X, y, groups), refit = True)
    search.fit(X, y)
    return search.best_estimator_, search.best_params_
#Accuracy, sensitivity, specificity, precision and F1 of a fitted model on each CV fold (hyper-parameters fixed)
#Returns the list of per-fold metric dictionaries
def cross_validate_model(model, X, y, groups, seed = 0):
    out = []
    for tr, va in grouped_folds(5, seed).split(X, y, groups):
        model.fit(X[tr], y[tr])
        out.append(met.classification_metrics(y[va], model.predict(X[va])))
    return out
#Mean and standard deviation of every metric over the folds
def summarize_folds(fold_metrics):
    return {k: (float(np.mean([f[k] for f in fold_metrics])), float(np.std([f[k] for f in fold_metrics]))) for k in fold_metrics[0]}
#---------------------------------------------------------
#Section 3: Final evaluation on the held-out test patients
#---------------------------------------------------------
#Predict the test set and return the confusion matrix [[TN, FP], [FN, TP]] and the classification metrics
def evaluate_model(model, X, y):
    pred = model.predict(X)
    return met.confusion_matrix_2x2(y, pred), met.classification_metrics(y, pred), pred
#Complete experiment: patient-level 70/30 split -> ROI datasets -> for each classifier: tune and cross-validate
#on the 70 % training patients, then evaluate once on the 30 % test patients
def run_classification(data_dir, names = ("svm", "knn", "rf"), seed = 42):
    train_p, test_p = util.split_patients(util.list_patients(data_dir), 0.30, seed)
    Xtr, ytr, gtr, mtr = build_roi_dataset(data_dir, train_p, seed)
    Xte, yte, gte, mte = build_roi_dataset(data_dir, test_p, seed)
    results = {}
    for name in names:
        model, best = tune_classifier(name, Xtr, ytr, gtr, seed)
        folds = cross_validate_model(model, Xtr, ytr, gtr, seed)
        model.fit(Xtr, ytr)
        cm, m, pred = evaluate_model(model, Xte, yte)
        results[name] = {"best_params": best, "cv": summarize_folds(folds), "test_confusion": cm, "test_metrics": m, "test_pred": pred, "model": model}
    data = {"train": (Xtr, ytr, gtr, mtr), "test": (Xte, yte, gte, mte), "train_patients": train_p, "test_patients": test_p}
    return results, data

#=============================
#Task 3: Diagnostic Evaluation
#=============================
#Feature analysis tables (for the report)
#Per-class mean +- std of every feature and a one-way ANOVA F-test between healthy and tumor ROIs
#Returns a list of rows (name, mean_healthy, std_healthy, mean_tumor, std_tumor, F, p)
def feature_table(X, y, names = feat.FEATURE_NAMES):
    rows = []
    for j, n in enumerate(names):
        a, b = X[y == 0, j], X[y == 1, j]
        F, p = f_oneway(a, b) if a.std() + b.std() > 0 else (0.0, 1.0)
        rows.append((n, a.mean(), a.std(), b.mean(), b.std(), float(F), float(p)))
    return rows
