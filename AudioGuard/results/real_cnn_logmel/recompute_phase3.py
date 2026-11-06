"""Independently validate and recompute Phase 3 scores from saved CSVs only."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/real_cnn_logmel"
CNN_OUT = OUT / "cnn_logmel"
SVM_OUT = ROOT / "results/real_svm/phase2_original/svm_mfcc"
REQUIRED = {"partition", "speaker", "utterance", "attack", "label", "score", "audio_sha256"}
EXPECTED_N = {"dev": 24844, "eval": 71237}
EXPECTED_ATTACKS = {"dev": {f"A{i:02d}" for i in range(1, 7)},
                    "eval": {f"A{i:02d}" for i in range(7, 20)}}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(path, partition):
    result = {}
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 5:
                raise ValueError(f"Malformed protocol row {path}:{number}: {len(fields)} columns")
            speaker, utterance, placeholder, attack, label = fields
            if placeholder != "-" or label not in ("bonafide", "spoof"):
                raise ValueError(f"Unsupported protocol row {path}:{number}: {fields}")
            if utterance in result:
                raise ValueError(f"Duplicate protocol utterance {utterance} in {partition}")
            if (label == "bonafide" and attack != "-") or (label == "spoof" and not attack.startswith("A")):
                raise ValueError(f"Protocol attack/label mismatch at {path}:{number}")
            result[utterance] = {"speaker": speaker, "attack": attack,
                                 "label": 0 if label == "bonafide" else 1,
                                 "partition": partition}
    return result


def validate_score_csv(path, partition, protocol):
    errors = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        if not REQUIRED.issubset(columns):
            raise ValueError(f"{path} missing columns: {sorted(REQUIRED-columns)}")
        rows = list(reader)
    seen = set()
    labels, scores = [], []
    for index, row in enumerate(rows, 2):
        if any(row.get(key) is None or row.get(key) == "" for key in REQUIRED):
            errors.append(f"row {index}: missing value")
            continue
        utterance = row["utterance"]
        if utterance in seen:
            errors.append(f"row {index}: duplicate utterance {utterance}")
        seen.add(utterance)
        if row["partition"] != partition:
            errors.append(f"row {index}: partition={row['partition']}")
        try:
            label = int(row["label"])
            score = float(row["score"])
        except (TypeError, ValueError):
            errors.append(f"row {index}: invalid label/score")
            continue
        if label not in (0, 1):
            errors.append(f"row {index}: invalid label {label}")
        if not math.isfinite(score):
            errors.append(f"row {index}: non-finite score")
        expected = protocol.get(utterance)
        if expected is None:
            errors.append(f"row {index}: unknown utterance {utterance}")
        else:
            for key in ("speaker", "attack"):
                if row[key] != expected[key]:
                    errors.append(f"row {index}: {key} mismatch for {utterance}")
            if label != expected["label"]:
                errors.append(f"row {index}: label mismatch for {utterance}")
        labels.append(label)
        scores.append(score)
    if len(rows) != EXPECTED_N[partition]:
        errors.append(f"row_count={len(rows)} expected={EXPECTED_N[partition]}")
    missing = sorted(set(protocol) - seen)
    extra = sorted(seen - set(protocol))
    if missing:
        errors.append(f"missing_protocol_utterances={len(missing)}")
    if extra:
        errors.append(f"extra_utterances={len(extra)}")
    labels_set = set(labels)
    attacks = {r["attack"] for r in rows if r["label"] == "1"}
    if labels_set != {0, 1}:
        errors.append(f"labels_present={sorted(labels_set)}")
    if attacks != EXPECTED_ATTACKS[partition]:
        errors.append(f"spoof_attack_ids={sorted(attacks)} expected={sorted(EXPECTED_ATTACKS[partition])}")
    report = {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
              "sha256": sha256(path), "row_count": len(rows), "unique_utterances": len(seen),
              "expected_trials": EXPECTED_N[partition], "target_0_bona_fide": labels.count(0),
              "target_1_spoof": labels.count(1), "spoof_attack_ids": sorted(attacks),
              "missing_protocol_utterances": len(missing), "extra_utterances": len(extra),
              "validation_errors": errors}
    if errors:
        raise ValueError(json.dumps(report, indent=2))
    return rows, np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float64), report


def eer_values(y, scores):
    fpr, tpr, thresholds = roc_curve(y, scores, pos_label=1, drop_intermediate=False)
    fnr = 1.0 - tpr
    finite = np.flatnonzero(np.isfinite(thresholds))
    nearest = int(finite[np.argmin(np.abs(fpr[finite] - fnr[finite]))])
    difference = fpr - fnr
    interpolated = None
    for i in range(len(difference) - 1):
        if difference[i] == 0:
            interpolated = float((fpr[i] + fnr[i]) / 2)
            break
        if difference[i] * difference[i + 1] < 0:
            weight = -difference[i] / (difference[i + 1] - difference[i])
            interp_fpr = fpr[i] + weight * (fpr[i + 1] - fpr[i])
            interp_fnr = fnr[i] + weight * (fnr[i + 1] - fnr[i])
            interpolated = float((interp_fpr + interp_fnr) / 2)
            break
    if interpolated is None:
        interpolated = float((fpr[nearest] + fnr[nearest]) / 2)
    return {"nearest_point_eer": float((fpr[nearest] + fnr[nearest]) / 2),
            "nearest_point_fpr": float(fpr[nearest]), "nearest_point_fnr": float(fnr[nearest]),
            "nearest_point_threshold": float(thresholds[nearest]),
            "interpolated_eer": interpolated,
            "interpolation_method": "linear interpolation between adjacent ROC points where FPR-FNR changes sign"}


def metrics(rows, y, scores, threshold):
    prediction = (scores >= threshold).astype(np.int64)
    cm = confusion_matrix(y, prediction, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    precision, recall, f1, support = precision_recall_fscore_support(
        y, prediction, labels=[0, 1], zero_division=0)
    attacks = sorted({r["attack"] for r in rows if int(r["label"]) == 1})
    per_attack = {}
    for attack in attacks:
        selected = np.asarray([int(r["label"]) == 0 or r["attack"] == attack for r in rows])
        spoof_selected = np.asarray([r["attack"] == attack and int(r["label"]) == 1 for r in rows])
        attack_y, attack_scores = y[selected], scores[selected]
        per_attack[attack] = {
            "trial_count_including_shared_bona_fide_pool": int(selected.sum()),
            "attack_spoof_trials": int(spoof_selected.sum()),
            "shared_bona_fide_trials": int(np.sum(y == 0)),
            **eer_values(attack_y, attack_scores),
            "miss_rate_at_frozen_threshold": float(np.mean(scores[spoof_selected] < threshold)),
            "misses_at_frozen_threshold": int(np.sum(scores[spoof_selected] < threshold)),
        }
    return {
        "n": int(len(y)), "bona_fide_trials": int(np.sum(y == 0)), "spoof_trials": int(np.sum(y == 1)),
        "threshold_full_precision": float(threshold), "confusion_matrix_rows_true_0_1_cols_pred_0_1": cm.tolist(),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "fpr_fp_over_all_bona_fide": float(fp / (tn + fp)),
        "fnr_fn_over_all_spoof": float(fn / (fn + tp)),
        "accuracy": float(accuracy_score(y, prediction)),
        "precision_bona_fide": float(precision[0]), "recall_bona_fide": float(recall[0]),
        "f1_bona_fide": float(f1[0]), "support_bona_fide": int(support[0]),
        "precision_spoof": float(precision[1]), "recall_spoof": float(recall[1]),
        "spoof_f1": float(f1[1]), "support_spoof": int(support[1]),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "roc_auc_raw_score_target1_spoof": float(roc_auc_score(y, scores)),
        "roc_auc_negated_score": float(roc_auc_score(y, -scores)),
        **eer_values(y, scores), "per_attack": per_attack,
    }


def always_spoof(y):
    prediction = np.ones_like(y)
    precision, recall, f1, _ = precision_recall_fscore_support(y, prediction, labels=[0, 1], zero_division=0)
    return {"accuracy": float(accuracy_score(y, prediction)), "spoof_f1": float(f1[1]),
            "macro_f1": float(f1_score(y, prediction, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            "spoof_precision": float(precision[1]), "spoof_recall": float(recall[1]),
            "bona_fide_recall": float(recall[0]), "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist()}


def saved_compare(recomputed, saved, label):
    keys = ("roc_auc", "diagnostic_eer", "accuracy", "far", "frr", "precision", "recall", "f1", "confusion_matrix")
    comparisons = {}
    for key in keys:
        actual = recomputed["roc_auc_raw_score_target1_spoof"] if key == "roc_auc" else recomputed.get(key)
        if key == "diagnostic_eer":
            actual = recomputed["nearest_point_eer"]
        expected = saved.get(key)
        if key == "f1":
            actual = recomputed["spoof_f1"]
        if key == "precision":
            actual = recomputed["precision_spoof"]
        if key == "recall":
            actual = recomputed["recall_spoof"]
        if key == "far":
            actual = recomputed["fpr_fp_over_all_bona_fide"]
        if key == "frr":
            actual = recomputed["fnr_fn_over_all_spoof"]
        if key == "confusion_matrix":
            actual = recomputed["confusion_matrix_rows_true_0_1_cols_pred_0_1"]
        if isinstance(actual, list) or isinstance(expected, list):
            equal = actual == expected
            delta = None
        else:
            delta = None if actual is None or expected is None else float(actual) - float(expected)
            equal = delta is not None and abs(delta) <= 1e-12
        comparisons[key] = {"saved": expected, "recomputed": actual, "difference": delta, "matches_within_1e-12": equal}
    return {"label": label, "comparisons": comparisons,
            "all_match": all(x["matches_within_1e-12"] for x in comparisons.values())}


def main():
    config_path = ROOT / "configs/baseline.local.yaml"
    baseline = json.loads("{}")
    import yaml
    with config_path.open(encoding="utf-8") as handle:
        baseline = yaml.safe_load(handle)["dataset"]
    protocols = {p: load_protocol(Path(baseline["protocols"][p]), p) for p in ("dev", "eval")}
    # Validate all CNN and copied Phase 2 CSVs before any metric is computed.
    cnn_scores = {}
    svm_scores = {}
    validations = {}
    for part, cnn_path, svm_path in (
        ("dev", CNN_OUT / "dev_scores.csv", SVM_OUT / "dev_scores.csv"),
        ("eval", CNN_OUT / "eval_clean_scores.csv", SVM_OUT / "eval_clean_scores.csv"),
    ):
        cnn_scores[part] = validate_score_csv(cnn_path, part, protocols[part])
        svm_scores[part] = validate_score_csv(svm_path, part, protocols[part])
        validations[part] = {"cnn": cnn_scores[part][3], "svm": svm_scores[part][3]}
    validation = {"status": "PASS", "method": "protocol-keyed row-by-row validation before metric calculation",
                  "checks": ["row and unique utterance counts", "exact protocol utterance set", "speaker/attack/label agreement",
                             "partition names", "required columns", "missing values", "finite scores", "attack ID set"],
                  **validations}
    assert validation["status"] == "PASS"

    cnn_meta = json.loads((CNN_OUT / "training.json").read_text(encoding="utf-8"))
    svm_meta = json.loads((SVM_OUT / "training.json").read_text(encoding="utf-8"))
    cnn_threshold = float(cnn_meta["threshold_selection"]["threshold"])
    svm_threshold = float(svm_meta["threshold_selection"]["threshold"])
    cnn_metrics = {p: metrics(*cnn_scores[p][:3], cnn_threshold) for p in ("dev", "eval")}
    svm_metrics = {p: metrics(*svm_scores[p][:3], svm_threshold) for p in ("dev", "eval")}
    saved_cnn_dev = cnn_meta["dev"]["metrics"]
    saved_cnn_eval = json.loads((CNN_OUT / "eval_clean.json").read_text(encoding="utf-8"))["metrics"]
    saved_svm_dev = svm_meta["dev"]["metrics"]
    saved_svm_eval = json.loads((SVM_OUT / "eval_clean.json").read_text(encoding="utf-8"))["metrics"]
    comparisons = {
        "cnn_dev_vs_saved_training_json": saved_compare(cnn_metrics["dev"], saved_cnn_dev, "CNN DEV"),
        "cnn_eval_vs_saved_clean_eval_json": saved_compare(cnn_metrics["eval"], saved_cnn_eval, "CNN EVAL"),
        "svm_dev_vs_saved_phase2_training_json": saved_compare(svm_metrics["dev"], saved_svm_dev, "SVM DEV"),
        "svm_eval_vs_saved_phase2_evaluator_json": saved_compare(svm_metrics["eval"], saved_svm_eval, "SVM EVAL"),
    }
    always = always_spoof(cnn_scores["eval"][1])
    comparison_tables = {}
    for part in ("dev", "eval"):
        comparison_tables[part] = {
            "CNN": {k: cnn_metrics[part][k] for k in ("roc_auc_raw_score_target1_spoof", "nearest_point_eer", "interpolated_eer", "spoof_f1", "macro_f1", "balanced_accuracy", "accuracy", "recall_spoof", "confusion_matrix_rows_true_0_1_cols_pred_0_1", "fpr_fp_over_all_bona_fide", "fnr_fn_over_all_spoof")},
            "SVM": {k: svm_metrics[part][k] for k in ("roc_auc_raw_score_target1_spoof", "nearest_point_eer", "interpolated_eer", "spoof_f1", "macro_f1", "balanced_accuracy", "accuracy", "recall_spoof", "confusion_matrix_rows_true_0_1_cols_pred_0_1", "fpr_fp_over_all_bona_fide", "fnr_fn_over_all_spoof")},
        }
    comparison_tables["eval"]["always_spoof_baseline"] = always
    pooled_deltas = {}
    for key, label in (("nearest_point_eer", "pooled_eer"), ("roc_auc_raw_score_target1_spoof", "roc_auc"),
                       ("balanced_accuracy", "balanced_accuracy"), ("recall_spoof", "spoof_recall")):
        delta = float(cnn_metrics["eval"][key] - svm_metrics["eval"][key])
        pooled_deltas[label] = {"cnn_minus_svm": delta, "direction": "improvement" if (delta < 0 if label == "pooled_eer" else delta > 0) else "regression"}
    attack_comparison = {}
    for part in ("dev", "eval"):
        attack_comparison[part] = {attack: {model: {"eer_nearest": scores[part]["per_attack"][attack]["nearest_point_eer"],
                                                        "eer_interpolated": scores[part]["per_attack"][attack]["interpolated_eer"],
                                                        "miss_rate": scores[part]["per_attack"][attack]["miss_rate_at_frozen_threshold"],
                                                        "attack_trials": scores[part]["per_attack"][attack]["attack_spoof_trials"],
                                                        "shared_bona_fide_trials": scores[part]["per_attack"][attack]["shared_bona_fide_trials"]}
                                              for model, scores in (("CNN", cnn_metrics), ("SVM", svm_metrics))}
                                        for attack in cnn_metrics[part]["per_attack"]}
    result = {
        "validation": validation,
        "threshold_full_precision": {"cnn_dev_selected": repr(cnn_threshold), "svm_phase2_dev_selected": repr(svm_threshold)},
        "cnn": {p: cnn_metrics[p] for p in ("dev", "eval")},
        "svm_recomputed_from_saved_csvs": {p: svm_metrics[p] for p in ("dev", "eval")},
        "saved_metric_comparisons": comparisons,
        "eer_method_note": "nearest finite ROC point is the evaluator convention; interpolated EER linearly crosses adjacent ROC points where FPR-FNR changes sign",
        "cnn_vs_svm_tables": comparison_tables,
        "eval_pooled_deltas_cnn_minus_svm": pooled_deltas,
        "per_attack_cnn_vs_svm": attack_comparison,
        "input_score_hashes": {"cnn_dev": cnn_scores['dev'][3]['sha256'], "cnn_eval": cnn_scores['eval'][3]['sha256'],
                               "svm_dev": svm_scores['dev'][3]['sha256'], "svm_eval": svm_scores['eval'][3]['sha256']},
    }
    target = OUT / "phase3_recomputed_metrics.json"
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(json.dumps({"validation": validation["status"], "thresholds": result["threshold_full_precision"],
                      "cnn_dev": {k: cnn_metrics['dev'][k] for k in ('roc_auc_raw_score_target1_spoof','nearest_point_eer','interpolated_eer','confusion_matrix_rows_true_0_1_cols_pred_0_1')},
                      "cnn_eval": {k: cnn_metrics['eval'][k] for k in ('roc_auc_raw_score_target1_spoof','nearest_point_eer','interpolated_eer','accuracy','spoof_f1','macro_f1','balanced_accuracy','recall_spoof','confusion_matrix_rows_true_0_1_cols_pred_0_1')},
                      "saved_metric_comparison_all_match": {k:v['all_match'] for k,v in comparisons.items()},
                      "always_spoof_eval": always, "cnn_vs_svm_eval_deltas": pooled_deltas,
                      "output": str(target)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
