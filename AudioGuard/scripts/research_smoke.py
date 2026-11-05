"""End-to-end synthetic ablation smoke; archive failures, never claim speech results.

Balanced train-only tone fixtures exercise the mandatory memorization check.
Passing demonstrates software wiring, not genuine deepfake generalization.
"""
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
from src.common import load_config, write_json


def main():
    cfg = load_config("configs/research_smoke.yaml")
    ds = load_config(cfg["dataset_config"])["dataset"]
    root, output = Path(ds["root"]), Path(cfg["output"])
    if output.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        shutil.move(str(output), str(output.parent / ("archive_" + stamp)))
    fixture = cfg["fixture"]
    rng = np.random.default_rng(cfg["seed"])
    for part in ("train", "dev", "eval"):
        folder = root / part
        folder.mkdir(parents=True, exist_ok=True)
        lines = []
        for i in range(fixture["counts"][part]):
            label = i % 2
            name = f"synthetic_{part}_{i:04d}"
            attack = "-" if not label else ("A07" if part == "eval" else "A01")
            lines.append(f"speaker_{part}_{i//8} {name} - {attack} {'spoof' if label else 'bonafide'}")
            t = np.arange(fixture["duration_samples"]) / fixture["sample_rate"]
            hz = fixture["spoof_hz"] if label else fixture["bonafide_hz"]
            hz += rng.uniform(-fixture["frequency_jitter_hz"], fixture["frequency_jitter_hz"])
            waveform = fixture["amplitude"] * np.sin(2*np.pi*hz*t) + rng.normal(0, fixture["noise_std"], len(t))
            sf.write(folder / (name + ".wav"), waveform, fixture["sample_rate"])
        (root / (part + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    commands = [[sys.executable, "-m", "pytest", "-q"],
                [sys.executable, "scripts/experiment.py", "suite", "--config", "configs/research_smoke.yaml"],
                [sys.executable, "scripts/verify_research.py", "--config", "configs/research_smoke.yaml"]]
    records = []
    env = dict(os.environ, GRADIO_ANALYTICS_ENABLED="False")
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True, env=env)
        records.append({"command": command, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        print(" ".join(command[1:]), result.returncode, flush=True)
    status = {"scope": cfg["scope"], "fixture_counts": fixture["counts"], "records": records}
    write_json("results/research_smoke/status.json", status)
    with Path("results/research_smoke/history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(status) + "\n")
    write_json("docs/research_validation.json", {"scope": status["scope"], "fixture_counts": status["fixture_counts"],
               "records": [{k: v for k, v in item.items() if k != "stdout"} | {"stdout_tail": item["stdout"][-1500:]} for item in records]})
    if any(item["exit_code"] for item in records):
        raise SystemExit("Research smoke failed. Inspect saved status/history and condition failures.")


if __name__ == "__main__":
    main()
