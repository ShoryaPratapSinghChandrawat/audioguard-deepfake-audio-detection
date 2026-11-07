"""Reproduce Phase 5.1 CNN batch-shape and sample-rate sensitivity diagnostics."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly, stft

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo.inference import FrozenCNN
from scripts.experiment import load_model
from src.common import load_config
from src.data.protocol import parse_protocol
from src.features.acoustic import extract
from src.features.spectrum import read_audio
from src.robustness.transforms import ffmpeg_executable


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/real_cnn_logmel/cnn_logmel"
DATASET_CONFIG = ROOT / "configs/baseline.local.yaml"
OUT = ROOT / "results/phase5_diagnostics/summary.md"
PROTECTED = (
    "results/real_cnn_logmel/cnn_logmel/model.pt",
    "results/real_cnn_logmel/cnn_logmel/scaler.npz",
    "results/real_cnn_logmel/cnn_logmel/eval_clean.json",
    "results/real_cnn_logmel/cnn_logmel/training.json",
    "results/real_cnn_logmel/phase3_recomputed_metrics.json",
    "results/phase4b/optimized_results/phase4b_metrics.json",
    "MANIFEST.sha256.json",
)


def sha256_snapshot():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in PROTECTED}


def label(logit, threshold):
    return "spoof" if logit >= threshold else "bona fide"


def waveform_for(row, rows_by_id, audio_root):
    protocol_row = rows_by_id[row["utterance"]]
    path = audio_root / f"{protocol_row['utterance']}.flac"
    return read_audio(path, 16000), path


def cnn_input(audio, frontend, scaler, frames):
    from src.models.cnn import tensor_input

    feature = extract(audio, frontend, "logmel")
    return tensor_input(feature, scaler, frames).squeeze(0)


def run_batch_diagnostic(saved_rows, protocol_rows, audio_root, meta, model, scaler, device):
    selected_bf = [r for r in saved_rows if int(r["label"]) == 0][:10]
    selected_spoof = [r for r in saved_rows if int(r["label"]) == 1][:10]
    selected = selected_bf + selected_spoof
    if len(selected) != 20:
        raise ValueError("Need 10 saved bona fide and 10 saved spoof EVAL rows")
    selected_ids = {r["utterance"] for r in selected}
    padding = [r for r in saved_rows if r["utterance"] not in selected_ids][:12]
    if len(padding) != 12:
        raise ValueError("Need 12 distinct EVAL rows to form the batch of 32")
    rows_by_id = {r["utterance"]: r for r in protocol_rows}
    ordered = selected + padding
    tensors, audios = [], {}
    frontend = meta["config"]["frontend"]
    for row in ordered:
        audio, _ = waveform_for(row, rows_by_id, audio_root)
        audios[row["utterance"]] = audio
        tensors.append(cnn_input(audio, frontend, scaler, meta["config"]["cnn"]["frames"]))
    batch = torch.stack(tensors).unsqueeze(1).to(device)
    with torch.no_grad():
        batch_logits = model(batch).detach().cpu().numpy()
        singles = []
        for tensor in tensors[:20]:
            value = model(tensor[None, None].to(device)).detach().cpu().item()
            singles.append(float(value))
    output = []
    for i, row in enumerate(selected):
        saved = float(row["score"])
        output.append({
            "utterance": row["utterance"],
            "label": "spoof" if int(row["label"]) == 1 else "bona fide",
            "saved": saved,
            "batch32": float(batch_logits[i]),
            "single": singles[i],
            "batch32_abs_diff": abs(float(batch_logits[i]) - saved),
            "single_abs_diff": abs(singles[i] - saved),
        })
    return output, ordered


def ffmpeg_convert(ffmpeg, source, destination, output_rate):
    command = [
        ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(source),
        "-map", "0:a:0", "-vn", "-sn", "-dn", "-ac", "1", "-ar", str(output_rate),
        "-c:a", "pcm_f32le", "-f", "wav", str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=90, check=False)
    if result.returncode or not destination.is_file():
        raise RuntimeError(result.stderr[-2000:] or f"FFmpeg failed for {source}")


def ffmpeg_roundtrip(ffmpeg, source_audio, directory, stem, high_rate):
    source_path = directory / f"{stem}_16k.wav"
    high_path = directory / f"{stem}_{high_rate}k.wav"
    out_path = directory / f"{stem}_{high_rate}k_to_16k.wav"
    sf.write(source_path, source_audio, 16000, subtype="FLOAT")
    ffmpeg_convert(ffmpeg, source_path, high_path, high_rate)
    ffmpeg_convert(ffmpeg, high_path, out_path, 16000)
    return read_audio(out_path, 16000)


def spectral_values(audio, frontend):
    win, hop, nfft = frontend["win_length"], frontend["hop_length"], frontend["n_fft"]
    frequencies, _, spectrum = stft(
        np.asarray(audio, dtype=np.float64), fs=frontend["sample_rate"], window="hann",
        nperseg=win, noverlap=win-hop, nfft=nfft, boundary=None, padded=False,
    )
    power = np.abs(spectrum) ** 2
    band_means = []
    for low, high in ((6000, 7000), (7000, 8000)):
        mask = (frequencies >= low) & (frequencies < high)
        band_means.append(float(power[mask].mean()))
    logmel = extract(audio, frontend, "logmel")
    return {
        "6_7kHz_mean_power": band_means[0],
        "7_8kHz_mean_power": band_means[1],
        "first_mel_time_mean": float(logmel[0].mean()),
        "last_mel_time_mean": float(logmel[-1].mean()),
    }


def main():
    before = sha256_snapshot()
    meta = json.loads((RUN / "training.json").read_text(encoding="utf-8"))
    eval_artifact = json.loads((RUN / "eval_clean.json").read_text(encoding="utf-8"))
    threshold = float(eval_artifact["threshold"])
    if threshold != float(meta["threshold_selection"]["threshold"]):
        raise ValueError("Frozen eval and training thresholds disagree")
    model, scaler, device = load_model(RUN, meta)
    dataset = load_config(DATASET_CONFIG)["dataset"]
    protocol_rows = parse_protocol(dataset["protocols"]["eval"], "eval")
    audio_root = Path(dataset["audio_roots"]["eval"])
    with (RUN / "eval_clean_scores.csv").open(newline="", encoding="utf-8") as handle:
        saved_rows = list(csv.DictReader(handle))

    batch_rows, ordered32 = run_batch_diagnostic(
        saved_rows, protocol_rows, audio_root, meta, model, scaler, device,
    )
    max_batch32 = max(r["batch32_abs_diff"] for r in batch_rows)
    max_single = max(r["single_abs_diff"] for r in batch_rows)
    nearest_saved_margin = min(abs(float(r["score"]) - threshold) for r in saved_rows
                               if r["utterance"] in {item["utterance"] for item in batch_rows})

    # The requested five are the three bona fide and two spoof clips from the
    # prior Phase 5 simulated-microphone report.
    selected_ids = ["LA_E_1002903", "LA_E_1003128", "LA_E_1004363", "LA_E_1000147", "LA_E_1000273"]
    saved_by_id = {r["utterance"]: r for r in saved_rows}
    protocol_by_id = {r["utterance"]: r for r in protocol_rows}
    cnn = FrozenCNN.from_run(RUN)
    ffmpeg = ffmpeg_executable()
    transformations = []
    spectral_rows = []
    with tempfile.TemporaryDirectory(prefix="audioguard-phase5-diagnostics-") as temporary:
        temp = Path(temporary)
        for index, utterance in enumerate(selected_ids):
            score_row = saved_by_id[utterance]
            original, source_path = waveform_for(score_row, protocol_by_id, audio_root)
            reference, _ = cnn.infer(original)
            controls = {"ffmpeg_16k_rewrite": original}

            # (a) Re-decode the original file through FFmpeg and write float PCM
            # at the original 16 kHz rate (no sample-rate conversion).
            rewrite_path = temp / f"{index}_ffmpeg_16k.wav"
            ffmpeg_convert(ffmpeg, source_path, rewrite_path, 16000)
            controls["ffmpeg_16k_rewrite"] = read_audio(rewrite_path, 16000)

            # (b), (d), and (e) use the same 16 kHz float PCM source and real FFmpeg
            # conversion to the intermediate rate and back to 16 kHz.
            for rate, key in ((48000, "ffmpeg_16_48_16"), (32000, "ffmpeg_16_32_16"),
                              (8000, "ffmpeg_16_8_16")):
                controls[key] = ffmpeg_roundtrip(ffmpeg, original, temp, f"{index}_{rate}", rate)

            # (c) SciPy polyphase resampling provides an independent software path.
            controls["scipy_poly_16_48_16"] = resample_poly(resample_poly(original, 3, 1), 1, 3).astype(np.float32)

            for name, waveform in controls.items():
                logit, _ = cnn.infer(waveform)
                transformations.append({
                    "utterance": utterance,
                    "path": name,
                    "reference_logit": float(reference),
                    "input_logit": float(logit),
                    "signed_difference": float(logit - reference),
                    "reference_label": label(reference, threshold),
                    "input_label": label(logit, threshold),
                    "flipped": label(reference, threshold) != label(logit, threshold),
                })
            converted48 = controls["ffmpeg_16_48_16"]
            spectral_rows.append({
                "utterance": utterance,
                "original": spectral_values(original, meta["config"]["frontend"]),
                "ffmpeg_16_48_16": spectral_values(converted48, meta["config"]["frontend"]),
            })

    after = sha256_snapshot()
    if before != after:
        raise RuntimeError("A protected artifact hash changed during diagnostics")

    lines = [
        "# Phase 5.1 diagnostics",
        "",
        "Generated by `scripts/phase5_diagnostics.py` from saved Phase 3 artifacts and official EVAL audio.",
        "No model, threshold, or demo behavior was changed. All differences use raw CNN logits.",
        "",
        f"Inference device: `{device}`. Frozen threshold from saved `eval_clean.json`: `{threshold:.17g}`.",
        f"Batch construction: 20 selected parity trials plus 12 distinct EVAL trials; batch size `{len(ordered32)}`.",
        f"Max absolute difference from saved logits, batch-32: `{max_batch32:.17g}`.",
        f"Max absolute difference from saved logits, one-at-a-time: `{max_single:.17g}`.",
        f"Nearest selected saved logit to threshold: `{nearest_saved_margin:.17g}`; selected clips within the one-at-a-time maximum difference of threshold: "
        f"`{sum(abs(float(r['score']) - threshold) <= max_single for r in saved_rows if r['utterance'] in {item['utterance'] for item in batch_rows})}`.",
        "",
        "## Batch-shape comparison",
        "",
        "| EVAL utterance | Target | Saved logit | Batch-32 logit | Single logit | Batch-32 abs diff | Single abs diff |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in batch_rows:
        lines.append(
            f"| {r['utterance']} | {r['label']} | {r['saved']:.17g} | {r['batch32']:.17g} | "
            f"{r['single']:.17g} | {r['batch32_abs_diff']:.17g} | {r['single_abs_diff']:.17g} |"
        )
    lines += [
        "",
        "## Signed simulated-input results",
        "",
        "Labels use the frozen threshold above (`logit >= threshold` means spoof).",
        "",
        "| EVAL utterance | Input path | Reference logit | Input logit | Signed difference | Reference label | Input label | Flipped |",
        "|---|---|---:|---:|---:|---|---|---:|",
    ]
    for r in transformations:
        lines.append(
            f"| {r['utterance']} | {r['path']} | {r['reference_logit']:.17g} | {r['input_logit']:.17g} | "
            f"{r['signed_difference']:.17g} | {r['reference_label']} | {r['input_label']} | {r['flipped']} |"
        )
    lines += [
        "",
        "## Spectral measurements",
        "",
        "Mean one-sided STFT power for frequency bands; first/last Mel values are means of the raw log-Mel time series.",
        "",
        "| EVAL utterance | Waveform | 6–7 kHz mean power | 7–8 kHz mean power | First Mel-bin mean | Last Mel-bin mean |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in spectral_rows:
        for wave_key, wave_label in (("original", "original"), ("ffmpeg_16_48_16", "FFmpeg 16→48→16")):
            v = row[wave_key]
            lines.append(
                f"| {row['utterance']} | {wave_label} | {v['6_7kHz_mean_power']:.17g} | "
                f"{v['7_8kHz_mean_power']:.17g} | {v['first_mel_time_mean']:.17g} | {v['last_mel_time_mean']:.17g} |"
            )
    lines += ["", "## Protected artifact hashes", "", "| Artifact | Before SHA-256 | After SHA-256 | Match |", "|---|---|---|---:|"]
    for name in PROTECTED:
        lines.append(f"| `{name}` | `{before[name]}` | `{after[name]}` | {before[name] == after[name]} |")
    lines += ["", "Temporary resampling audio was removed by the temporary-directory context manager.", ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"Wrote {OUT}")
    print(f"device={device} batch32_max_abs_diff={max_batch32:.17g} single_max_abs_diff={max_single:.17g}")
    print(f"simulated rows={len(transformations)} spectral clips={len(spectral_rows)} protected_hashes_match=True")


if __name__ == "__main__":
    main()
