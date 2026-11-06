"""Report expanded metrics from saved DEV/EVAL per-clip score CSVs.

The threshold is read from training.json and never selected from EVAL.
Continuous scores are kept in their original orientation for all metrics.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.experiment import config
from src.common import digest, write_json
from src.evaluation.metrics import empirical_eer


LABEL_NAMES = {0: "bonafide", 1: "spoof"}
RATE_DEFINITIONS = {
    "TP": "spoof correctly classified as spoof",
    "FN": "spoof classified as bona fide",
    "FP": "bona fide classified as spoof",
    "TN": "bona fide correctly classified as bona fide",
    "spoof_false_accept_rate": "FN / (TP + FN), per the requested definition",
    "bonafide_false_reject_rate": "FP / (TN + FP), per the requested definition",
    "conventional_fpr": "FP / (FP + TN), denominator is all bona fide trials",
    "conventional_fnr": "FN / (FN + TP), denominator is all spoof trials",
}


def calculate_metrics(labels, scores, threshold):
    """Calculate thresholded class metrics and threshold-independent ranking metrics."""
    y = np.asarray(labels, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or values.shape != y.shape or set(y.tolist()) != {0, 1}:
        raise ValueError("Expected equally sized one-dimensional labels containing both 0 and 1")
    if not np.isfinite(values).all() or not np.isfinite(threshold):
        raise ValueError("Scores and threshold must be finite")

    predictions = (values >= threshold).astype(np.int64)
    matrix = confusion_matrix(y, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in matrix.ravel())
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predictions, labels=[0, 1], zero_division=0
    )
    eer = empirical_eer(y, values)

    def rate(numerator, denominator):
        return float(numerator / denominator) if denominator else None

    return {
        "n": int(len(y)),
        "bonafide": int(np.sum(y == 0)),
        "spoof": int(np.sum(y == 1)),
        "threshold": float(threshold),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision_by_class": {LABEL_NAMES[label]: float(precision[label]) for label in (0, 1)},
        "recall_by_class": {LABEL_NAMES[label]: float(recall[label]) for label in (0, 1)},
        "f1_by_class": {LABEL_NAMES[label]: float(f1[label]) for label in (0, 1)},
        "support_by_class": {LABEL_NAMES[label]: int(support[label]) for label in (0, 1)},
        "f1_spoof": float(f1[1]),
        "macro_f1": float(np.mean(f1)),
        "balanced_accuracy": float(np.mean(recall)),
        "spoof_false_accept_rate": rate(fn, tp + fn),
        "bonafide_false_reject_rate": rate(fp, tn + fp),
        "conventional_fpr": rate(fp, fp + tn),
        "conventional_fnr": rate(fn, fn + tp),
        "roc_auc_original_scores": float(roc_auc_score(y, values)),
        "roc_auc_negated_scores_diagnostic": float(roc_auc_score(y, -values)),
        "pooled_eer": float(eer["eer"]),
        "pooled_eer_method": eer["method"],
        "rate_definitions": RATE_DEFINITIONS,
    }


def read_scores(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No score rows in {path}")
    for row in rows:
        row["label"] = int(row["label"])
        row["score"] = float(row["score"])
    return rows


def write_predictions(rows, threshold, path):
    fields = ["partition", "speaker", "utterance", "attack", "label", "score",
              "prediction", "prediction_name", "threshold", "audio_sha256"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            predicted = int(float(row["score"]) >= threshold)
            writer.writerow({
                **{field: row.get(field, "") for field in fields if field in row},
                "prediction": predicted,
                "prediction_name": LABEL_NAMES[predicted],
                "threshold": threshold,
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research.yaml")
    parser.add_argument("--model", choices=("svm_mfcc",), default="svm_mfcc")
    args = parser.parse_args()
    cfg = config(args.config)
    name = args.model
    if name not in cfg["models"]:
        raise ValueError(f"{name} is not enabled in {args.config}")

    out = Path(cfg["output"]) / name
    meta = json.loads((out / "training.json").read_text(encoding="utf-8"))
    if cfg != meta["config"]:
        raise ValueError("Config changed after train. Use the frozen config or a fresh experiment.")
    threshold = float(meta["threshold_selection"]["threshold"])
    dev_rows = read_scores(out / "dev_scores.csv")
    eval_rows = read_scores(out / "eval_clean_scores.csv")
    if any(row["partition"] != "dev" for row in dev_rows):
        raise ValueError("dev_scores.csv contains a non-DEV row")
    if any(row["partition"] != "eval" for row in eval_rows):
        raise ValueError("eval_clean_scores.csv contains a non-EVAL row")

    selected = empirical_eer([row["label"] for row in dev_rows], [row["score"] for row in dev_rows])
    if selected["threshold"] != threshold:
        raise ValueError("Saved DEV scores do not reproduce the frozen threshold")

    dev_predictions_path = out / "dev_predictions.csv"
    eval_predictions_path = out / "eval_clean_predictions.csv"
    metrics_path = out / "extra_metrics.json"
    polarity_path = out / "score_polarity.json"
    scaler_path = out / "scaler.joblib"
    for path in (dev_predictions_path, eval_predictions_path, metrics_path, polarity_path, scaler_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    dev_metrics = calculate_metrics(
        [row["label"] for row in dev_rows], [row["score"] for row in dev_rows], threshold
    )
    eval_metrics = calculate_metrics(
        [row["label"] for row in eval_rows], [row["score"] for row in eval_rows], threshold
    )
    write_predictions(dev_rows, threshold, dev_predictions_path)
    write_predictions(eval_rows, threshold, eval_predictions_path)

    attack_ids = sorted({row["attack"] for row in eval_rows if row["label"] == 1})
    per_attack = {}
    for attack in attack_ids:
        subset = [row for row in eval_rows if row["label"] == 0 or row["attack"] == attack]
        per_attack[attack] = {
            "scope": "this spoof attack plus the same shared EVAL bona fide pool",
            "eer_label": "DIAGNOSTIC; not a deployment threshold",
            "metrics": calculate_metrics(
                [row["label"] for row in subset], [row["score"] for row in subset], threshold
            ),
        }

    model = joblib.load(out / "model.joblib")
    classes = [int(value) for value in model[-1].classes_]
    joblib.dump(model[0], scaler_path)
    polarity = {
        "svc_classes_ordering": classes,
        "expected_classes_order": [0, 1],
        "classes_order_matches_expected": classes == [0, 1],
        "binary_decision_function_positive_side_class": classes[-1],
        "higher_score_means_spoof_by_classes_order": classes[-1] == 1,
        "audioGuard_metric_convention": "score >= threshold predicts spoof; ROC-AUC uses spoof label 1",
        "raw_scores_reversed": False,
        "dev_auc_original": dev_metrics["roc_auc_original_scores"],
        "dev_auc_negated_diagnostic": dev_metrics["roc_auc_negated_scores_diagnostic"],
        "eval_auc_original": eval_metrics["roc_auc_original_scores"],
        "eval_auc_negated_diagnostic": eval_metrics["roc_auc_negated_scores_diagnostic"],
        "interpretation": (
            "Original score orientation favors spoof-positive ranking."
            if dev_metrics["roc_auc_original_scores"] >= dev_metrics["roc_auc_negated_scores_diagnostic"]
            else "Polarity discrepancy: negated DEV scores rank better; original scores were retained."
        ),
    }
    result = {
        "model": name,
        "threshold_source": "complete clean DEV only; frozen before EVAL",
        "threshold": threshold,
        "dev": dev_metrics,
        "eval": eval_metrics,
        "per_attack_eval": per_attack,
        "score_polarity": polarity,
        "prediction_files": [str(dev_predictions_path), str(eval_predictions_path)],
        "scaler_artifact": str(scaler_path),
        "scaler_sha256": digest(scaler_path),
    }
    write_json(metrics_path, result)
    write_json(polarity_path, polarity)
    print(json.dumps({
        "saved": str(metrics_path),
        "threshold": threshold,
        "dev": {key: dev_metrics[key] for key in ("n", "balanced_accuracy", "macro_f1", "roc_auc_original_scores", "roc_auc_negated_scores_diagnostic", "pooled_eer")},
        "eval": {key: eval_metrics[key] for key in ("n", "balanced_accuracy", "macro_f1", "roc_auc_original_scores", "roc_auc_negated_scores_diagnostic", "pooled_eer")},
        "score_polarity": polarity,
        "attack_ids": attack_ids,
    }, indent=2))


if __name__ == "__main__":
    main()
