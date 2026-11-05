"""Command-line experiment lifecycle; writes raw scores before derived reports.

Run from the AudioGuard workspace. Train never extracts eval audio; evaluation
loads a frozen artifact and cannot change model parameters or operating threshold.
"""
import argparse
import csv
import json
import subprocess
import sys
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.common import load_config, seed_all, write_json, provenance, digest
from src.data.protocol import inspect, load_dataset, counts
from src.features.spectrum import matrix
from src.models.logistic import fit, score
from src.evaluation.metrics import empirical_eer, metrics, attack_metrics
from src.robustness.transforms import noise, mp3, ffmpeg_executable


def save_scores(path, rows, scores):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        fields = ["partition", "speaker", "utterance", "attack", "label", "score", "audio_sha256"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, value in zip(rows, scores):
            writer.writerow({**{k: row[k] for k in fields[:5]}, "score": float(value),
                             "audio_sha256": digest(row["path"])})


def train(cfg, out):
    if (out / "model.npz").exists() or (out / "training.json").exists():
        raise FileExistsError("Training output already exists; choose a fresh output in YAML")
    data, audit = load_dataset(cfg)
    seed_all(cfg["seed"])
    xtrain = matrix(data["train"], cfg["features"])
    ytrain = np.array([r["label"] for r in data["train"]])
    model, history, device = fit(xtrain, ytrain, cfg["model"], cfg["seed"])
    xdev = matrix(data["dev"], cfg["features"])
    ydev = np.array([r["label"] for r in data["dev"]])
    dev_scores = score(xdev, model)
    selection = empirical_eer(ydev, dev_scores)
    np.savez(out / "model.npz", **model)
    save_scores(out / "train_scores.csv", data["train"], score(xtrain, model))
    save_scores(out / "dev_scores.csv", data["dev"], dev_scores)
    metadata = {**provenance(cfg), "status": "completed", "dataset_audit": audit,
                "device": device, "history": history, "threshold_selection": {"partition": "dev", **selection},
                "model_sha256": digest(out / "model.npz"),
                "training_attacks": sorted({r["attack"] for r in data["train"] if r["label"]}),
                "dev_metrics": metrics(ydev, dev_scores, selection["threshold"])}
    write_json(out / "training.json", metadata)
    print(json.dumps(metadata["dev_metrics"], indent=2))


def frozen(cfg, out):
    meta = json.loads((out / "training.json").read_text(encoding="utf-8"))
    for key in ("dataset", "features", "model", "seed"):
        if cfg[key] != meta["config"][key]:
            raise ValueError(f"Frozen experiment config changed: {key}; create a new run instead")
    if digest(out / "model.npz") != meta["model_sha256"]:
        raise ValueError("Model checksum mismatch")
    data, audit = load_dataset(cfg)
    if audit["protocol_sha256"] != meta["dataset_audit"]["protocol_sha256"]:
        raise ValueError("Protocols changed since training")
    with np.load(out / "model.npz", allow_pickle=False) as archive:
        model = {name: archive[name] for name in archive.files}
    return meta, data, audit, model


def evaluate(cfg, out, robustness=False):
    meta, data, audit, model = frozen(cfg, out)
    rows = data["eval"]
    labels = np.array([r["label"] for r in rows])
    threshold = meta["threshold_selection"]["threshold"]
    conditions = [("clean", None)]
    if robustness:
        conditions = []
        for snr in cfg["robustness"]["snr_db"]:
            conditions.append((f"noise_{snr}dB", lambda a, r, snr=snr: noise(a, snr, cfg["robustness"]["seed"], r["utterance"])))
        for bitrate in cfg["robustness"]["mp3_bitrates_kbps"]:
            conditions.append((f"mp3_{bitrate}kbps", lambda a, r, bitrate=bitrate: mp3(a, cfg["features"]["sample_rate"], bitrate)))
    failures = []
    for name, transform in conditions:
        result = {"condition": name, "partition": "eval", "threshold_source": "clean dev",
                  "model_sha256": meta["model_sha256"], "threshold": threshold,
                  "dataset_audit": audit, "provenance": provenance(cfg)}
        try:
            scores = score(matrix(rows, cfg["features"], transform), model)
            save_scores(out / f"{name}_scores.csv", rows, scores)
            result.update(status="completed", metrics=metrics(labels, scores, threshold),
                          per_attack=attack_metrics(rows, scores, threshold, meta["training_attacks"]))
            if name.startswith("mp3"):
                result["ffmpeg_version"] = subprocess.run([ffmpeg_executable(), "-version"], capture_output=True, text=True, check=True).stdout
        except Exception as exc:
            result.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
            failures.append(name)
        write_json(out / f"{name}.json", result)
        print(name, result["status"])
    if failures:
        raise RuntimeError(f"Failed conditions (recorded, not hidden): {failures}")


def report(out):
    # All performance numbers are loaded from script-written JSON, never literals.
    lines = ["# AudioGuard experiment report", "",
             "Scope: " + str(out) + ". Paths containing smoke refer to SYNTHETIC SOFTWARE TESTS, NOT ASVspoof.", "Higher score = spoof; bonafide=0, spoof=1.",
             "Operating threshold is selected on clean dev only. EER is empirical, not official toolkit EER.",
             "No min-tDCF is claimed: ASV scores/costs and official implementation are not provided.", ""]
    training = out / "training.json"
    if not training.exists():
        lines.append("BLOCKED: no completed training artifact. No dataset performance can be reported.")
    else:
        meta = json.loads(training.read_text(encoding="utf-8"))
        lines += ["## Data scope", "```json", json.dumps({k: meta["dataset_audit"][k] for k in
                  ("full_counts", "used_counts", "subset_requested", "subset_selection", "checks")}, indent=2), "```",
                  "## Dev operating-point selection", "```json", json.dumps(meta["threshold_selection"], indent=2), "```"]
        for path in sorted(out.glob("*.json")):
            obj = json.loads(path.read_text(encoding="utf-8"))
            if "condition" in obj:
                lines += [f"## {obj['condition']}", "```json", json.dumps({k: obj[k] for k in
                          ("status", "metrics", "per_attack", "error") if k in obj}, indent=2), "```"]
        if not (out / "clean.json").exists():
            lines.append("Eval has not been run. Generalization remains unanswered.")
    path = Path("reports") / (out.name + ".md")
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "audit", "train", "evaluate", "robustness", "report"])
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    try:
        if args.command == "inspect":
            write_json(out / "inspection.json", inspect(cfg["dataset"]["root"]))
        elif args.command == "audit":
            _, audit = load_dataset(cfg)
            write_json(out / "audit.json", audit)
        elif args.command == "train":
            train(cfg, out)
        elif args.command == "report":
            report(out)
        else:
            evaluate(cfg, out, args.command == "robustness")
    except Exception as exc:
        # Append attempts so a subsequent success cannot erase an earlier failure.
        with (out / "failures.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"command": args.command, "error": repr(exc),
                                     "traceback": traceback.format_exc(), "provenance": provenance(cfg)}) + "\n")
        raise


if __name__ == "__main__":
    main()
