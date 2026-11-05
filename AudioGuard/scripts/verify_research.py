"""Recompute ablation metrics/analyses from CSV and verify frozen-dev decisions.

Also validates restored checkpoints and constructs Gradio without launching a
public service. The checks apply to synthetic or official-data result files.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.common import write_json
from src.evaluation.analysis import summary
from src.evaluation.metrics import empirical_eer
from scripts.experiment import config, load_model


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["label"] = int(row["label"])
    return rows, np.array([float(r["score"]) for r in rows])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/research.yaml")
    args = parser.parse_args()
    cfg = config(args.config)
    checked = []
    for name in cfg["models"]:
        out = Path(cfg["output"]) / name
        meta = json.loads((out / "training.json").read_text(encoding="utf-8"))
        rows, scores = read(out / "dev_scores.csv")
        selection = empirical_eer([r["label"] for r in rows], scores)
        assert selection["threshold"] == meta["threshold_selection"]["threshold"]
        load_model(out, meta)
        for path in out.glob("eval_*.json"):
            obj = json.loads(path.read_text(encoding="utf-8"))
            if obj["status"] != "completed":
                raise ValueError(f"Failed condition: {path}")
            assert obj["threshold"] == selection["threshold"]
            rows, scores = read(out / (path.stem + "_scores.csv"))
            derived = summary(rows, scores, selection["threshold"], cfg["analysis"]["top_errors"])
            for key in ("metrics", "roc", "per_attack", "per_speaker"):
                assert derived[key] == obj[key], (path, key)
            for key in ("false_positives", "false_negatives"):
                assert [r["utterance"] for r in derived["top_errors"][key]] == [r["utterance"] for r in obj["top_errors"][key]]
            checked.append(str(path))
    from demo.app import build
    build(Path(cfg["output"]) / cfg["models"][0]).close()
    write_json(Path(cfg["output"]) / "verification.json", {"status": "passed", "checked_conditions": checked,
               "dev_thresholds_verified": True, "checkpoints_load": True, "gradio_builds": True})
    print("Verified metrics, plot arrays, thresholds, checkpoints and demo for:", cfg["models"])


if __name__ == "__main__":
    main()
