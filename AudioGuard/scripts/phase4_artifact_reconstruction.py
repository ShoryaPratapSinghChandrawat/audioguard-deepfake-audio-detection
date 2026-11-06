"""Verify saved CNN and SVM inference against 20 saved clean EVAL scores.

The script reads source audio and saved artifacts directly. It bypasses both
feature caches and does not write to protected Phase 3 result directories.
"""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.experiment import load_model
from src.data.protocol import load_dataset
from src.features.acoustic import extract, pooled
from src.features.spectrum import read_audio
from src.models.cnn import tensor_input


OUT = ROOT / "results/phase4a"
CNN_OUT = ROOT / "results/real_cnn_logmel/cnn_logmel"
SVM_REFERENCE = ROOT / "results/real_svm/phase2_original/svm_mfcc"
SVM_ARTIFACTS = OUT / "artifact_inputs/svm_mfcc"
ORIGINAL_SVM = Path(r"C:\Games\projects\audioguarddeepfake\audioguard-deepfake-audio-detection-main\audioguard-deepfake-audio-detection-main\AudioGuard\results\real_svm\svm_mfcc")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_map(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["utterance"]: float(row["score"]) for row in csv.DictReader(stream)}


def select_trials(rows):
    bona_fide = next(row for row in rows if row["label"] == 0)
    selected = [bona_fide]
    seen = set()
    for row in rows:
        if row["label"] == 1 and row["attack"] not in seen:
            selected.append(row)
            seen.add(row["attack"])
    more_bona_fide = [row for row in rows if row["label"] == 0 and row["utterance"] != bona_fide["utterance"]]
    selected.extend(more_bona_fide[: max(0, 20 - len(selected))])
    if len(selected) != 20:
        raise ValueError(f"Selected {len(selected)} EVAL trials; expected 20")
    return selected


def main():
    cnn_meta = json.loads((CNN_OUT / "training.json").read_text(encoding="utf-8"))
    svm_meta = json.loads((SVM_REFERENCE / "training.json").read_text(encoding="utf-8"))
    cnn_cfg = cnn_meta["config"]
    svm_cfg = svm_meta["config"]
    cnn_data, _ = load_dataset(cnn_cfg)
    svm_data, _ = load_dataset(svm_cfg)
    cnn_rows, svm_rows = cnn_data["eval"], svm_data["eval"]
    if [r["utterance"] for r in cnn_rows] != [r["utterance"] for r in svm_rows]:
        raise ValueError("CNN and SVM EVAL protocol order differs")
    rows = select_trials(cnn_rows)
    utterances = [row["utterance"] for row in rows]

    cnn, scaler, device = load_model(CNN_OUT, cnn_meta)
    svm, _, _ = load_model(SVM_ARTIFACTS, svm_meta)
    if device != "cuda":
        raise RuntimeError(f"Expected saved CNN to reconstruct on CUDA, got {device}")
    cnn_saved = score_map(CNN_OUT / "eval_clean_scores.csv")
    svm_saved = score_map(SVM_REFERENCE / "eval_clean_scores.csv")
    cnn_actual_by_utterance, svm_actual = {}, []
    cnn_frontend, svm_frontend = cnn_cfg["frontend"], svm_cfg["frontend"]
    batch_size = int(cnn_cfg["cnn"]["batch_size"])
    eval_index = {row["utterance"]: index for index, row in enumerate(cnn_rows)}
    selected_batches = sorted({(eval_index[row["utterance"]] // batch_size) * batch_size for row in rows})
    with torch.no_grad():
        # Preserve the original clean evaluator's batch shape to avoid small
        # cuDNN numeric changes from singleton-vs-batched convolution kernels.
        for start in selected_batches:
            batch_rows = cnn_rows[start:start + batch_size]
            batch_features = []
            for batch_row in batch_rows:
                audio = read_audio(batch_row["path"], cnn_frontend["sample_rate"])
                feature = extract(audio, cnn_frontend, "logmel")
                batch_features.append(tensor_input(feature, scaler, cnn_cfg["cnn"]["frames"]))
            batch_tensor = torch.stack(batch_features).to(device)
            batch_scores = cnn(batch_tensor).cpu().tolist()
            for batch_row, score in zip(batch_rows, batch_scores):
                cnn_actual_by_utterance[batch_row["utterance"]] = float(score)
        for row in rows:
            svm_audio = read_audio(row["path"], svm_frontend["sample_rate"])
            svm_feature = pooled(extract(svm_audio, svm_frontend, "mfcc"))[None, :]
            svm_actual.append(float(svm.decision_function(svm_feature)[0]))

    cnn_expected = [cnn_saved[u] for u in utterances]
    svm_expected = [svm_saved[u] for u in utterances]
    cnn_actual = [cnn_actual_by_utterance[u] for u in utterances]
    result = {
        "status": "passed",
        "n_clips": len(rows),
        "protocol_selection": "7 bona fide clips plus one spoof clip per EVAL attack A07-A19",
        "feature_caches_used": False,
        "cnn": {
            "checkpoint": str((CNN_OUT / cnn_meta["artifact"]).relative_to(ROOT)),
            "checkpoint_sha256": sha256(CNN_OUT / cnn_meta["artifact"]),
            "checkpoint_bytes": (CNN_OUT / cnn_meta["artifact"]).stat().st_size,
            "normalization_statistics": str((CNN_OUT / "scaler.npz").relative_to(ROOT)),
            "normalization_sha256": sha256(CNN_OUT / "scaler.npz"),
            "normalization_bytes": (CNN_OUT / "scaler.npz").stat().st_size,
            "architecture_definition": "src/models/cnn.py",
            "frontend_configuration": cnn_meta["config"]["features_config"],
            "frontend": cnn_meta["config"]["frontend"],
            "resolved_config": "results/real_cnn_logmel/resolved_config_resume.json",
            "score_type": "raw CNN logit",
            "device": str(device),
            "threshold": repr(cnn_meta["threshold_selection"]["threshold"]),
            "max_absolute_difference": float(np.max(np.abs(np.asarray(cnn_actual) - cnn_expected))),
        },
        "svm": {
            "model": str((SVM_ARTIFACTS / svm_meta["artifact"]).relative_to(ROOT)),
            "model_sha256": sha256(SVM_ARTIFACTS / svm_meta["artifact"]),
            "model_bytes": (SVM_ARTIFACTS / svm_meta["artifact"]).stat().st_size,
            "scaler": str((SVM_ARTIFACTS / "scaler.joblib").relative_to(ROOT)),
            "scaler_sha256": sha256(SVM_ARTIFACTS / "scaler.joblib"),
            "scaler_bytes": (SVM_ARTIFACTS / "scaler.joblib").stat().st_size,
            "source_original_workspace": str(ORIGINAL_SVM),
            "training_config": svm_meta["config"],
            "score_type": "SVM decision function",
            "threshold": repr(svm_meta["threshold_selection"]["threshold"]),
            "max_absolute_difference": float(np.max(np.abs(np.asarray(svm_actual) - svm_expected))),
        },
        "clips": [
            {"utterance": row["utterance"], "speaker": row["speaker"],
             "label": int(row["label"]), "attack": row["attack"],
             "cnn_saved": cnn_saved[row["utterance"]], "cnn_reproduced": cnn_actual[i],
             "svm_saved": svm_saved[row["utterance"]], "svm_reproduced": svm_actual[i]}
            for i, row in enumerate(rows)
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "artifact_reconstruction.json"
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "clips"}, indent=2))


if __name__ == "__main__":
    main()
