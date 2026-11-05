"""Score one local mono clip using the frozen model and clean-dev threshold.

This is an offline research demo, not a claim of reliable real-world detection.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.common import digest
from src.features.spectrum import read_audio, extract
from src.models.logistic import score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--run", default="results/baseline")
    args = parser.parse_args()
    run = Path(args.run)
    meta = json.loads((run / "training.json").read_text(encoding="utf-8"))
    if digest(run / "model.npz") != meta["model_sha256"]:
        raise ValueError("Model checksum mismatch")
    cfg = meta["config"]["features"]
    x = extract(read_audio(args.audio, cfg["sample_rate"]), cfg)[None, :]
    with np.load(run / "model.npz", allow_pickle=False) as model:
        value = float(score(x, model)[0])
    threshold = meta["threshold_selection"]["threshold"]
    print(json.dumps({"score": value, "dev_threshold": threshold,
                      "predicted_label": int(value >= threshold), "higher_score": "spoof"}, indent=2))


if __name__ == "__main__":
    main()
