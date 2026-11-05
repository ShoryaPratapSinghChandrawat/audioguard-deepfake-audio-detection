"""Persist richer eval diagnostics without using eval to select any parameter.

Per-attack EER pairs that attack with the SAME eval bonafide pool; spoof-only
ROC is undefined. Protocol membership and synthesis-system reuse are separate claims.
"""
import csv
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve
from src.data.protocol import parse_protocol
from src.evaluation.metrics import metrics, empirical_eer


def summary(rows, scores, threshold, top):
    y = np.array([r["label"] for r in rows])
    prediction = scores >= threshold
    overall = {**metrics(y, scores, threshold),
               "precision": float(precision_score(y, prediction, zero_division=0)),
               "recall": float(recall_score(y, prediction, zero_division=0)),
               "f1": float(f1_score(y, prediction, zero_division=0))}
    overall["confusion_matrix"] = [[overall["tn"], overall["fp"]], [overall["fn"], overall["tp"]]]
    far, tpr, thresholds = roc_curve(y, scores, pos_label=1, drop_intermediate=False)
    # Infinity is the all-negative ROC sentinel, never a valid JSON number.
    roc = {"fpr": far.tolist(), "tpr": tpr.tolist(),
           "thresholds": [float(v) if np.isfinite(v) else None for v in thresholds]}
    attacks = {}
    for attack in sorted({r["attack"] for r in rows if r["label"]}):
        mask = np.array([r["label"] == 0 or r["attack"] == attack for r in rows])
        attacks[attack] = {"metrics": metrics(y[mask], scores[mask], threshold),
                           "eer": empirical_eer(y[mask], scores[mask])["eer"],
                           "eer_scope": "this attack plus all eval bonafide; diagnostic only"}
    speakers = {}
    for speaker in sorted({r["speaker"] for r in rows}):
        mask = np.array([r["speaker"] == speaker for r in rows])
        n0, n1 = int(np.sum(mask & (y == 0))), int(np.sum(mask & (y == 1)))
        fp, fn = int(np.sum(mask & (y == 0) & prediction)), int(np.sum(mask & (y == 1) & ~prediction))
        speakers[speaker] = {"n": int(mask.sum()), "bonafide": n0, "spoof": n1,
                             "errors": fp+fn, "error_rate": float(np.mean(prediction[mask] != y[mask])),
                             "far": fp/n0 if n0 else None, "frr": fn/n1 if n1 else None}
    errors = {}
    for name, label, descending in (("false_positives", 0, True), ("false_negatives", 1, False)):
        idx = [i for i in range(len(rows)) if y[i] == label and prediction[i] != bool(label)]
        idx.sort(key=lambda i: float(scores[i]), reverse=descending)
        errors[name] = [{**rows[i], "score": float(scores[i]), "threshold": threshold,
                         "margin": float(scores[i]-threshold)} for i in idx[:top]]
    return {"metrics": overall, "roc": roc, "per_attack": attacks,
            "per_speaker": speakers, "top_errors": errors}


def attack_claims(dataset_cfg, evidence_path=None):
    # Full protocols, never a debug subset, establish membership only.
    protocols = {p: parse_protocol(dataset_cfg["protocols"][p], p) for p in ("train", "dev", "eval")}
    ids = {p: {r["attack"] for r in rows if r["label"]} for p, rows in protocols.items()}
    evidence = {}
    if evidence_path:
        with Path(evidence_path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if not row.get("source", "").strip():
                    raise ValueError("Reuse evidence needs a source reference")
                evidence[row["attack_id"]] = row
    result = {}
    requested_unseen = {f"A{i:02d}" for i in range(7, 16)}
    requested_reuse = {"A16": "A04", "A19": "A06"}
    for attack in sorted(ids["eval"] | requested_unseen | set(requested_reuse)):
        present = attack in ids["eval"]
        item = {"present_in_eval_protocol": present, "present_in_train_protocol": attack in ids["train"],
                "unseen_id_verified": present and attack not in ids["train"],
                "requested_unseen_group": attack in requested_unseen}
        if attack in requested_reuse:
            parent = requested_reuse[attack]
            item["claimed_training_reuse"] = parent
            item["parent_present_in_train_protocol"] = parent in ids["train"]
            row = evidence.get(attack)
            supported = bool(present and parent in ids["train"] and row and row["reuses_training_attack"] == parent)
            item["reuse_status"] = "supported_by_user_supplied_metadata" if supported else "UNVERIFIED: protocols encode IDs, not synthesis methods"
            item["reuse_source"] = row["source"] if supported else None
        result[attack] = item
    return {"protocol_membership": {p: sorted(v) for p, v in ids.items()}, "annotations": result,
            "warning": "Unseen ID does not prove unseen synthesis family. Reuse cannot be established from trial protocols alone."}
