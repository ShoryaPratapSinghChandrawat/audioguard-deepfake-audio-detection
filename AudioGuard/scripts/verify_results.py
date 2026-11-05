"""Recompute persisted metrics from CSV scores to detect stale or inconsistent results.

This checks provenance consistency without running feature extraction again. It is
not a substitute for rerunning training from the original audio and protocols.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.common import write_json
from src.evaluation.metrics import metrics, empirical_eer


def read_scores(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return np.array([int(r["label"]) for r in rows]), np.array([float(r["score"]) for r in rows])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="results/baseline")
    args = parser.parse_args()
    root = Path(args.run)
    training = json.loads((root / "training.json").read_text(encoding="utf-8"))
    y, s = read_scores(root / "dev_scores.csv")
    chosen = empirical_eer(y, s)
    if chosen["threshold"] != training["threshold_selection"]["threshold"]:
        raise ValueError("Persisted threshold does not match dev score selection")
    checked = []
    for path in sorted(root.glob("*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        if "condition" not in obj or obj["status"] != "completed":
            continue
        y, s = read_scores(root / (obj["condition"] + "_scores.csv"))
        if obj["metrics"] != metrics(y, s, chosen["threshold"]):
            raise ValueError(f"Metric mismatch: {path}")
        checked.append(obj["condition"])
    write_json(root / "verification.json", {"status": "passed", "checked_conditions": checked,
                                            "threshold_matches_dev_csv": True})
    print("Verified:", checked)


if __name__ == "__main__":
    main()
