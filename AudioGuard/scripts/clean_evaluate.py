"""Score only clean EVAL audio using the frozen DEV threshold.

This is a clean-only alternative to experiment.py evaluate, which also runs
trim, noise, and MP3 robustness conditions.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.audioguard import save_scores
from scripts.experiment import config, load_model, predict_rows
from src.common import write_json
from src.data.protocol import load_dataset
from src.evaluation.analysis import attack_claims, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research.yaml")
    parser.add_argument("--model", choices=("svm_mfcc", "cnn_logmel"), default="svm_mfcc")
    args = parser.parse_args()

    cfg = config(args.config)
    name = args.model
    if name not in cfg["models"]:
        raise ValueError(f"{name} is not enabled in {args.config}")
    out = Path(cfg["output"]) / name
    meta = json.loads((out / "training.json").read_text(encoding="utf-8"))
    if cfg != meta["config"]:
        raise ValueError("Config changed after train. Use the frozen config or a fresh experiment.")

    result_path = out / "eval_clean.json"
    scores_path = out / "eval_clean_scores.csv"
    attack_path = out / "attack_verification.json"
    if result_path.exists() or scores_path.exists() or attack_path.exists():
        raise FileExistsError("Refusing to overwrite clean EVAL evidence")

    # The DEV operating point is read and frozen before EVAL waveforms are scored.
    threshold = float(meta["threshold_selection"]["threshold"])
    data, audit = load_dataset(cfg)
    if audit["protocol_sha256"] != meta["dataset_audit"]["protocol_sha256"]:
        raise ValueError("Protocols changed after training")

    model, scaler, device = load_model(out, meta)
    if name == "svm_mfcc":
        svc = model[-1]
        classes = [int(value) for value in svc.classes_]
        polarity = {
            "score_type": "SVM decision function",
            "classes": classes,
            "expected_classes_order": [0, 1],
            "classes_order_matches_expected": classes == [0, 1],
            "higher_score_means_spoof": classes[-1] == 1,
            "raw_scores_reversed": False,
        }
    else:
        classes = [0, 1]
        polarity = {
            "score_type": "raw CNN logit",
            "target_encoding": {"0": "bona fide", "1": "spoof"},
            "higher_score_means_spoof": True,
            "raw_scores_reversed": False,
        }
    scores = predict_rows(data["eval"], cfg, name, model, scaler, device)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(data["eval"]),) or not np.isfinite(scores).all():
        raise ValueError("Expected one finite continuous score per EVAL utterance")

    save_scores(scores_path, data["eval"], scores)
    annotation = attack_claims(cfg["dataset"], cfg["attack_reuse_evidence"])
    write_json(attack_path, annotation)
    result = {
        "condition": "clean",
        "scope": cfg["scope"],
        "name": name,
        "partition": "eval",
        "status": "completed",
        "dataset_audit": audit,
        "threshold": threshold,
        "threshold_source": "complete clean DEV only; frozen before EVAL scoring",
        "artifact_sha256": meta["artifact_sha256"],
        "attack_verification": annotation,
        "score_polarity": polarity,
    }
    result.update(summary(data["eval"], scores, threshold, cfg["analysis"]["top_errors"]))
    write_json(result_path, result)
    print(json.dumps({
        "status": result["status"],
        "partition": "eval",
        "n": len(data["eval"]),
        "threshold_source": result["threshold_source"],
        "threshold": threshold,
        "score_polarity": polarity,
        "scores_csv": str(scores_path),
        "result_json": str(result_path),
    }, indent=2))


if __name__ == "__main__":
    main()
