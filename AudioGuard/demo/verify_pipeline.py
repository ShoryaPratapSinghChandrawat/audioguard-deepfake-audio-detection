"""Parity and simulated-microphone checks against saved clean EVAL scores."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo.inference import FrozenCNN, load_waveform
from src.common import load_config
from src.data.protocol import parse_protocol
from src.robustness.transforms import ffmpeg_executable


def _ffmpeg_run(ffmpeg, command):
    result = subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-y", *command],
                            capture_output=True, text=True, timeout=60, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:] or "FFmpeg conversion failed")


def verify(run_directory, dataset_config):
    run = Path(run_directory)
    if not run.is_absolute():
        run = ROOT / run
    cfg_path = Path(dataset_config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    dataset = load_config(cfg_path)["dataset"]
    eval_rows = parse_protocol(dataset["protocols"]["eval"], "eval")
    row_map = {row["utterance"]: row for row in eval_rows}
    audio_root = Path(dataset["audio_roots"]["eval"])
    saved_csv = run / "eval_clean_scores.csv"
    with saved_csv.open(newline="", encoding="utf-8") as handle:
        saved_rows = list(csv.DictReader(handle))
    bona_fide = [row for row in saved_rows if int(row["label"]) == 0]
    spoof = [row for row in saved_rows if int(row["label"]) == 1]
    parity = bona_fide[:10] + spoof[:10]
    if len(parity) < 20:
        raise ValueError("The saved clean EVAL score CSV must include at least 10 trials per class")
    model = FrozenCNN.from_run(run)
    parity_differences = []
    for score_row in parity:
        trial = row_map[score_row["utterance"]]
        path = audio_root / f"{trial['utterance']}.flac"
        waveform, _ = load_waveform(path)
        logit, _ = model.infer(waveform)
        saved_logit = float(score_row["score"])
        parity_differences.append({
            "utterance": trial["utterance"],
            "label": trial["label"],
            "saved_logit": saved_logit,
            "demo_logit": logit,
            "absolute_difference": abs(logit - saved_logit),
        })

    ffmpeg = ffmpeg_executable()
    version = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True,
                             timeout=10, check=True).stdout.splitlines()[0]
    encoders = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True,
                              text=True, timeout=10, check=True).stdout
    available = {"mp3": "libmp3lame" in encoders, "opus": "libopus" in encoders}
    simulation_rows = (bona_fide[:3] + spoof[:2])
    simulations = []
    with tempfile.TemporaryDirectory(prefix="audioguard-simulated-mic-") as temporary:
        temp = Path(temporary)
        for index, score_row in enumerate(simulation_rows):
            trial = row_map[score_row["utterance"]]
            original_path = audio_root / f"{trial['utterance']}.flac"
            original_waveform, _ = load_waveform(original_path)
            reference_logit, _ = model.infer(original_waveform)
            wav48 = temp / f"clip_{index}_48k.wav"
            _ffmpeg_run(ffmpeg, ["-i", str(original_path), "-map", "0:a:0", "-ac", "1",
                                 "-ar", "48000", "-c:a", "pcm_f32le", "-f", "wav", str(wav48)])
            formats = [("wav_48k", wav48)]
            if available["mp3"]:
                mp3_path = temp / f"clip_{index}_48k.mp3"
                _ffmpeg_run(ffmpeg, ["-i", str(wav48), "-map", "0:a:0", "-ac", "1",
                                     "-ar", "48000", "-c:a", "libmp3lame", "-b:a", "128k",
                                     "-f", "mp3", str(mp3_path)])
                formats.append(("mp3_48k", mp3_path))
            if available["opus"]:
                opus_path = temp / f"clip_{index}_48k.webm"
                _ffmpeg_run(ffmpeg, ["-i", str(wav48), "-map", "0:a:0", "-ac", "1",
                                     "-ar", "48000", "-c:a", "libopus", "-b:a", "64k",
                                     "-f", "webm", str(opus_path)])
                formats.append(("opus_webm_48k", opus_path))
            for label, path in formats:
                converted, _ = load_waveform(path)
                actual, _ = model.infer(converted)
                simulations.append({
                    "utterance": trial["utterance"],
                    "label": trial["label"],
                    "input_path": label,
                    "reference_logit": reference_logit,
                    "simulated_input_logit": actual,
                    "absolute_difference": abs(actual - reference_logit),
                })
    by_format = {}
    for item in simulations:
        by_format.setdefault(item["input_path"], []).append(item["absolute_difference"])
    return {
        "scope": "simulated microphone-input pipeline tests, not physical microphone recordings",
        "parity_trials": len(parity_differences),
        "parity_class_counts": {"bona_fide": sum(x["label"] == 0 for x in parity_differences),
                                "spoof": sum(x["label"] == 1 for x in parity_differences)},
        "parity_max_absolute_logit_difference": max(x["absolute_difference"] for x in parity_differences),
        "parity_differences": parity_differences,
        "ffmpeg_version": version,
        "available_encoders": available,
        "simulated_clip_count": len(simulation_rows),
        "simulated_input_differences": simulations,
        "simulated_difference_summary": {
            key: {"count": len(values), "max_absolute_logit_difference": max(values),
                  "mean_absolute_logit_difference": float(np.mean(values))}
            for key, values in by_format.items()
        },
        "temporary_audio_removed": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--dataset-config", default="configs/baseline.local.yaml")
    args = parser.parse_args()
    print(json.dumps(verify(args.run, args.dataset_config), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
