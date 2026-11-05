"""Build tiny synthetic fixtures and exercise the pipeline without dataset claims.

Only software behavior is tested. The artificial tones are not speech, and their
classification metrics MUST NOT be interpreted as deepfake detection evidence.
"""
import json
from datetime import datetime, timezone
import shutil
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
from src.common import load_config, write_json


def main():
    cfg = load_config("configs/smoke.yaml")
    root = Path(cfg["dataset"]["root"])
    output = Path(cfg["output"])
    # Archive previous attempts, including failures, instead of erasing them.
    if output.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        shutil.move(str(output), str(output.parent / ("archived_" + stamp)))
    for p, partition in enumerate(("train", "dev", "eval")):
        folder = root / partition
        folder.mkdir(parents=True, exist_ok=True)
        lines = []
        for i in range(8):
            label = i % 2
            name = f"synthetic_{partition}_{i}"
            attack = "-" if label == 0 else ("A01" if partition != "eval" else "A99")
            lines.append(f"speaker_{p}_{i} {name} - {attack} {'spoof' if label else 'bonafide'}")
            t = np.arange(8000) / cfg["features"]["sample_rate"]
            waveform = 0.2 * np.sin(2 * np.pi * (200 + 900 * label + i * 3) * t)
            sf.write(folder / f"{name}.wav", waveform, cfg["features"]["sample_rate"])
        (root / f"{partition}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    stages = []
    for stage in ("inspect", "audit", "train", "evaluate", "robustness", "report"):
        command = [sys.executable, "scripts/audioguard.py", stage, "--config", "configs/smoke.yaml"]
        run = subprocess.run(command, capture_output=True, text=True)
        stages.append({"stage": stage, "exit_code": run.returncode,
                       "stdout": run.stdout, "stderr": run.stderr})
        print(stage, run.returncode)
    verify = subprocess.run([sys.executable, "scripts/verify_results.py", "--run", str(output)], capture_output=True, text=True)
    stages.append({"stage": "score_verification", "exit_code": verify.returncode,
                   "stdout": verify.stdout, "stderr": verify.stderr})
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True)
    stages.append({"stage": "tests", "exit_code": tests.returncode, "stdout": tests.stdout, "stderr": tests.stderr})
    status = {"scope": "SYNTHETIC SOFTWARE TEST ONLY, NOT ASVSPOOF", "stages": stages}
    write_json("results/smoke/status.json", status)
    with Path("results/smoke/history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(status) + "\n")
    failed = [s["stage"] for s in stages if s["exit_code"]]
    if failed:
        print("Failed stages (see status.json):", failed)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
