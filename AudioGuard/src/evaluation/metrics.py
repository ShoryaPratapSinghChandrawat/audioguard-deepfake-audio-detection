"""Spoof-positive metrics with dev-only operating threshold and explicit EER method.

EER is the mean of FAR and FRR at the finite empirical ROC point minimizing
absolute FAR-FRR (first tie). It is not an interpolated challenge-toolkit EER.
Eval EER/AUC describe rankings; they NEVER replace the frozen dev threshold.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix


def validate(y, scores):
    y, scores = np.asarray(y), np.asarray(scores)
    if y.ndim != 1 or scores.shape != y.shape or set(y.tolist()) != {0, 1}:
        raise ValueError("Expected equally sized one-dimensional arrays containing both labels 0 and 1")
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite scores")
    return y, scores


def empirical_eer(y, scores):
    y, scores = validate(y, scores)
    far, tpr, thresholds = roc_curve(y, scores, pos_label=1, drop_intermediate=False)
    valid = np.flatnonzero(np.isfinite(thresholds))
    index = valid[np.argmin(np.abs(far[valid] - (1 - tpr[valid])))]
    return {"eer": float((far[index] + 1 - tpr[index]) / 2),
            "threshold": float(thresholds[index]), "far": float(far[index]),
            "frr": float(1-tpr[index]), "method": "nearest finite empirical ROC; mean FAR/FRR"}


def metrics(y, scores, threshold):
    y, scores = validate(y, scores)
    if not np.isfinite(threshold):
        raise ValueError("Non-finite threshold")
    tn, fp, fn, tp = confusion_matrix(y, scores >= threshold, labels=[0, 1]).ravel()
    return {"n": int(len(y)), "bonafide": int(tn+fp), "spoof": int(tp+fn),
            "threshold": float(threshold), "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp), "far": float(fp/(tn+fp)),
            "frr": float(fn/(tp+fn)), "accuracy": float((tn+tp)/len(y)),
            "roc_auc": float(roc_auc_score(y, scores)),
            "diagnostic_eer": empirical_eer(y, scores)["eer"]}


def attack_metrics(rows, scores, threshold, training_attacks):
    result = {}
    for attack in sorted({r["attack"] for r in rows if r["label"] == 1}):
        mask = np.array([r["attack"] == attack for r in rows])
        result[attack] = {"n": int(mask.sum()), "frr": float(np.mean(scores[mask] < threshold)),
                          "seen_in_train": attack in training_attacks}
    return result
