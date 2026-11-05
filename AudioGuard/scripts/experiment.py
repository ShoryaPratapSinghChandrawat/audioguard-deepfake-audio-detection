"""Run the SVM/CNN ablations and shortcut controls with frozen dev selection.

Suite trains ALL requested models before it accesses eval audio. Every experiment
has a fresh output directory, raw scores, complete config, sample counts and failures.
"""
import argparse
import csv
import json
import os
import sys
import traceback
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from src.common import load_config, seed_all, write_json, digest, provenance
from src.data.protocol import load_dataset, counts
from src.features.acoustic import cached, extract, pooled
from src.features.spectrum import read_audio
from src.evaluation.metrics import empirical_eer
from src.evaluation.analysis import summary, attack_claims
from src.robustness.shortcuts import nuisance, trim
from src.robustness.transforms import noise, mp3, ffmpeg_executable
from scripts.audioguard import save_scores

NAMES = ("svm_mfcc", "cnn_logmel", "cnn_mfcc", "shortcut")


def config(path):
    cfg = load_config(path)
    cfg["dataset"] = load_config(cfg["dataset_config"])["dataset"]
    cfg["frontend"] = load_config(cfg["features_config"])
    return cfg


def vector(row, cfg, name, transform=None):
    frontend = cfg["frontend"]
    if name == "shortcut" or transform is not None:
        audio = read_audio(row["path"], frontend["sample_rate"])
        if transform:
            audio = transform(audio, row)
        if name == "shortcut":
            return nuisance(audio, frontend["sample_rate"], cfg["shortcut"]["silence_dbfs"])
        return pooled(extract(audio, frontend, "mfcc"))
    return pooled(cached(row["path"], frontend, "mfcc"))


def train(cfg, name):
    out = Path(cfg["output"]) / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "training.json").exists() or (out / "model.pt").exists() or (out / "model.joblib").exists():
        raise FileExistsError(f"Use a fresh output directory: {out}")
    data, audit = load_dataset(cfg)  # Existing leakage checks and inspection, unchanged.
    seed_all(cfg["seed"])
    y = np.array([r["label"] for r in data["train"]])
    fit_info, device = {}, "cpu"
    if name.startswith("cnn_"):
        import torch
        from src.models import cnn
        from src.models.logistic import select_device
        kind = name.removeprefix("cnn_")
        device = select_device(cfg["cnn"]["device"])
        scaler = cnn.train_scaler(data["train"], cfg["frontend"], kind)
        sanity = cnn.overfit(data["train"], cfg["frontend"], kind, scaler, cfg["cnn"], cfg["seed"], device)
        write_json(out / "sanity.json", {**sanity, "scope": cfg["scope"], "config": cfg,
                                       "counts": counts(data["train"])})
        if sanity["status"] != "passed":
            raise RuntimeError("Train-only overfit sanity check failed. Recorded sanity.json; normal fitting is blocked.")
        model, fit_info = cnn.fit(data["train"], data["dev"], cfg["frontend"], kind, scaler, cfg["cnn"], cfg["seed"], device)
        torch.save(model.cpu().state_dict(), out / "model.pt")
        model.to(device)
        np.savez(out / "scaler.npz", **scaler)
        scores = cnn.predict(model, data["dev"], cfg["frontend"], kind, scaler, cfg["cnn"], device)
        artifact = "model.pt"
    else:
        options = cfg["svm"] if name == "svm_mfcc" else cfg["shortcut"]
        estimator = SVC(**options, probability=False, random_state=cfg["seed"]) if name == "svm_mfcc" else LogisticRegression(
            C=options["C"], max_iter=options["max_iter"], class_weight=options["class_weight"], random_state=cfg["seed"])
        model = make_pipeline(StandardScaler(), estimator)
        x = np.stack([vector(row, cfg, name) for row in data["train"]])
        model.fit(x, y)
        joblib.dump(model, out / "model.joblib")
        scores = model.decision_function(np.stack([vector(row, cfg, name) for row in data["dev"]]))
        artifact = "model.joblib"
    selection = empirical_eer([r["label"] for r in data["dev"]], scores)
    save_scores(out / "dev_scores.csv", data["dev"], scores)
    # Hash train waveforms even though raw train predictions are not needed.
    write_json(out / "train_manifest.json", [{**row, "audio_sha256": digest(row["path"])} for row in data["train"]])
    meta = {**provenance(cfg), "name": name, "scope": cfg["scope"], "status": "completed", "dataset_audit": audit,
            "artifact": artifact, "artifact_sha256": digest(out / artifact), "device": device, "fit": fit_info,
            "threshold_selection": {"partition": "dev", **selection},
            "dev": summary(data["dev"], scores, selection["threshold"], cfg["analysis"]["top_errors"])}
    if name.startswith("cnn_"):
        meta["scaler_sha256"] = digest(out / "scaler.npz")
    write_json(out / "training.json", meta)
    return out


def load_model(out, meta):
    cfg, name = meta["config"], meta["name"]
    if digest(out / meta["artifact"]) != meta["artifact_sha256"]:
        raise ValueError("Model checksum mismatch")
    if name.startswith("cnn_"):
        import torch
        from src.models.cnn import SmallCNN
        from src.models.logistic import select_device
        if digest(out / "scaler.npz") != meta["scaler_sha256"]:
            raise ValueError("Train scaler checksum mismatch")
        with np.load(out / "scaler.npz", allow_pickle=False) as source:
            scaler = {k: source[k] for k in source.files}
        device = select_device(cfg["cnn"]["device"])
        model = SmallCNN(cfg["cnn"])
        model.load_state_dict(torch.load(out / meta["artifact"], map_location="cpu", weights_only=True))
        return model.to(device).eval(), scaler, device
    # Never load untrusted downloaded pickle/joblib files. Checksum is provenance, not a sandbox.
    return joblib.load(out / meta["artifact"]), None, "cpu"


def predict_rows(rows, cfg, name, model, scaler, device, transform=None):
    if not name.startswith("cnn_"):
        return model.decision_function(np.stack([vector(row, cfg, name, transform) for row in rows]))
    from src.models import cnn
    import torch
    if transform is None:
        return cnn.predict(model, rows, cfg["frontend"], name.removeprefix("cnn_"), scaler, cfg["cnn"], device)
    values = []
    with torch.no_grad():
        for row in rows:
            audio = transform(read_audio(row["path"], cfg["frontend"]["sample_rate"]), row)
            x = extract(audio, cfg["frontend"], name.removeprefix("cnn_"))
            tensor = cnn.tensor_input(x, scaler, cfg["cnn"]["frames"])[None].to(device)
            values.append(float(torch.sigmoid(model(tensor)).cpu()[0]))
    return np.array(values)


def evaluate(cfg, name):
    out = Path(cfg["output"]) / name
    meta = json.loads((out / "training.json").read_text(encoding="utf-8"))
    if cfg != meta["config"]:
        raise ValueError("Config changed after train. Use the frozen config or a fresh experiment.")
    data, audit = load_dataset(cfg)
    if audit["protocol_sha256"] != meta["dataset_audit"]["protocol_sha256"]:
        raise ValueError("Protocols changed after training")
    model, scaler, device = load_model(out, meta)
    annotation = attack_claims(cfg["dataset"], cfg["attack_reuse_evidence"])
    write_json(out / "attack_verification.json", annotation)
    threshold = meta["threshold_selection"]["threshold"]
    analysis = cfg["analysis"]
    conditions = [("clean", None), ("trimmed", lambda a, r: trim(a, analysis["trim_dbfs"]))]
    for snr in analysis["noise_snr_db"]:
        conditions.append((f"noise_{snr}dB", lambda a, r, snr=snr: noise(a, snr, analysis["seed"], r["utterance"])))
    for bitrate in analysis["mp3_bitrates_kbps"]:
        conditions.append((f"mp3_{bitrate}kbps", lambda a, r, bitrate=bitrate: mp3(a, cfg["frontend"]["sample_rate"], bitrate)))
    failed = []
    for condition, transform in conditions:
        path = out / ("eval_" + condition + ".json")
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite condition evidence: {path}")
        result = {"condition": condition, "scope": cfg["scope"], "name": name, "partition": "eval",
                  "dataset_audit": audit, "threshold": threshold, "threshold_source": "clean dev only",
                  "artifact_sha256": meta["artifact_sha256"], "attack_verification": annotation}
        try:
            scores = predict_rows(data["eval"], cfg, name, model, scaler, device, transform)
            save_scores(out / ("eval_" + condition + "_scores.csv"), data["eval"], scores)
            result.update(status="completed", **summary(data["eval"], scores, threshold, analysis["top_errors"]))
            if condition.startswith("mp3"):
                import subprocess
                result["ffmpeg_version"] = subprocess.run([ffmpeg_executable(), "-version"], capture_output=True, text=True, check=True).stdout
        except Exception as exc:
            result.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
            failed.append(condition)
        write_json(path, result)
    if failed:
        raise RuntimeError(f"Failed conditions saved: {failed}")


def report(cfg, name):
    from src.evaluation.plots import draw
    out = Path(cfg["output"]) / name
    draw(out)
    meta = json.loads((out / "training.json").read_text(encoding="utf-8"))
    lines = ["# AudioGuard: " + name, "", "Scope: " + meta["scope"],
             "All values below come from saved JSON; no benchmark claims for synthetic fixtures.",
             "Higher score means spoof. Operating threshold is chosen on dev, never eval.",
             "## Frozen training and data scope", "```json",
             json.dumps({k: meta[k] for k in ("dataset_audit", "threshold_selection", "fit")}, indent=2), "```"]
    for path in sorted(out.glob("eval_*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        lines += ["## " + result["condition"], "```json", json.dumps({k: result[k] for k in
                  ("status", "metrics", "per_attack", "per_speaker", "top_errors", "error") if k in result}, indent=2), "```"]
    lines += ["## Attack membership and reuse caveat", "```json", (out / "attack_verification.json").read_text(encoding="utf-8"), "```",
              "Plots are in this run's figures/ folder. Checkpoint, feature configuration and trial scope are frozen."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "evaluate", "report", "suite"])
    parser.add_argument("--config", default="configs/research.yaml")
    parser.add_argument("--model", choices=NAMES)
    args = parser.parse_args()
    cfg = config(args.config)
    names = [args.model] if args.model else cfg["models"]
    root = Path(cfg["output"])
    root.mkdir(parents=True, exist_ok=True)
    try:
        if args.command in ("suite", "train"):
            for name in names:
                train(cfg, name)
        if args.command in ("suite", "evaluate"):
            for name in names:
                evaluate(cfg, name)
        if args.command in ("suite", "report"):
            for name in names:
                report(cfg, name)
    except Exception as exc:
        with (root / "failures.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"command": args.command, "models": names, "config": cfg,
                                     "error": repr(exc), "traceback": traceback.format_exc()}) + "\n")
        raise


if __name__ == "__main__":
    main()
