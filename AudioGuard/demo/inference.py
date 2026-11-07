"""Local-file adapter and frozen AudioGuard CNN/SVM inference helpers."""
from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import joblib

from scripts.experiment import load_model
from src.features.acoustic import extract, pooled
from src.features.spectrum import read_audio
from src.models.cnn import tensor_input
from src.robustness.transforms import ffmpeg_executable

LOGGER = logging.getLogger("audioguard.demo")
SUPPORTED_SUFFIXES = {".wav", ".flac", ".mp3", ".webm", ".opus", ".ogg"}
SAMPLE_RATE = 16000
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_DURATION_SECONDS = 30.0
SHORT_AUDIO_WARNING_SECONDS = 1.0
SILENCE_RMS_FLOOR = 1e-8
DECODE_HEADROOM_SECONDS = 0.05


class DemoInputError(ValueError):
    """Safe, user-visible audio validation error."""


def load_waveform(path, *, temp_root=None, max_bytes=MAX_UPLOAD_BYTES,
                  max_duration=MAX_DURATION_SECONDS):
    """Return mono 16 kHz float32 audio and a short-clip warning, if needed.

    A 16 kHz mono file is decoded with the training reader unchanged. Other
    supported layouts use FFmpeg to decode, downmix, and genuinely resample.
    No amplitude normalization is performed.
    """
    if path is None or str(path) == "":
        raise DemoInputError("Choose an audio file or record audio before analysis.")
    source = Path(path)
    if not source.is_file():
        raise DemoInputError("Audio file is empty or unavailable.")
    if source.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise DemoInputError("Unsupported audio type. Use WAV, FLAC, MP3, WebM, or Opus.")
    size = source.stat().st_size
    if size == 0:
        raise DemoInputError("Audio file is empty.")
    if size > max_bytes:
        raise DemoInputError("Audio file exceeds the 25 MiB upload limit.")

    info = None
    try:
        info = sf.info(str(source))
    except (RuntimeError, sf.SoundFileError, OSError):
        # MP3/WebM/Opus support varies with libsndfile; FFmpeg handles these.
        pass
    if info is not None and info.samplerate > 0:
        source_duration = info.frames / info.samplerate
        if source_duration > max_duration:
            raise DemoInputError("Audio duration exceeds the 30-second limit.")
        if info.samplerate == SAMPLE_RATE and info.channels == 1:
            try:
                audio = read_audio(source, SAMPLE_RATE)
            except Exception as exc:
                LOGGER.exception("Training-path audio decode failed for %s", source)
                raise DemoInputError("Audio could not be decoded. Check the file and try again.") from exc
            return _validate_waveform(audio, max_duration)

    try:
        ffmpeg = ffmpeg_executable()
        with tempfile.TemporaryDirectory(prefix="audioguard-decode-", dir=temp_root) as work:
            decoded_path = Path(work) / "decoded.wav"
            command = [
                ffmpeg, "-nostdin", "-v", "error", "-i", str(source),
                "-map", "0:a:0", "-vn", "-sn", "-dn",
                "-t", str(max_duration + DECODE_HEADROOM_SECONDS),
                "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_f32le",
                "-f", "wav", str(decoded_path),
            ]
            completed = subprocess.run(command, capture_output=True, text=True,
                                       timeout=60, check=False)
            if completed.returncode != 0 or not decoded_path.exists():
                LOGGER.error("FFmpeg decode failed for %s: %s", source,
                             completed.stderr[-2000:])
                raise DemoInputError("Audio could not be decoded. Check the file and try again.")
            audio = read_audio(decoded_path, SAMPLE_RATE)
    except DemoInputError:
        raise
    except (subprocess.SubprocessError, RuntimeError, OSError, ValueError) as exc:
        LOGGER.exception("Audio conversion failed for %s", source)
        raise DemoInputError("Audio could not be decoded. Check the file and try again.") from exc
    return _validate_waveform(audio, max_duration)


def _validate_waveform(audio, max_duration):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise DemoInputError("Audio is empty or invalid.")
    if len(audio) > int(max_duration * SAMPLE_RATE):
        raise DemoInputError("Audio duration exceeds the 30-second limit.")
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms <= SILENCE_RMS_FLOOR:
        raise DemoInputError("Audio is silent; no prediction is available.")
    warning = None
    if len(audio) / SAMPLE_RATE < SHORT_AUDIO_WARNING_SECONDS:
        warning = "Audio is shorter than 1 second; this prediction may be unreliable."
    return audio, warning


def cleanup_upload(path, temp_root):
    """Delete only a file contained in this app process's Gradio temp folder."""
    if path is None or temp_root is None:
        return False
    root = Path(temp_root).resolve()
    candidate = Path(path).resolve()
    if candidate == root or not candidate.is_relative_to(root):
        return False
    if candidate.is_file():
        candidate.unlink()
        return True
    return False


@dataclass
class FrozenCNN:
    model: object
    scaler: dict
    device: str
    config: dict
    threshold: float

    @classmethod
    def from_run(cls, run_directory):
        run = Path(run_directory)
        meta = json.loads((run / "training.json").read_text(encoding="utf-8"))
        if meta.get("name") != "cnn_logmel":
            raise ValueError("The demo requires the saved cnn_logmel Phase 3 run.")
        metrics_path = run.parent / "phase3_recomputed_metrics.json"
        try:
            training_threshold_value = meta["threshold_selection"]["threshold"]
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics_threshold_value = metrics["cnn"]["eval"]["threshold_full_precision"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "Saved TRAIN threshold or tracked Phase 3 EVAL threshold is unavailable."
            ) from exc
        if (isinstance(training_threshold_value, bool) or
                not isinstance(training_threshold_value, (int, float)) or
                isinstance(metrics_threshold_value, bool) or
                not isinstance(metrics_threshold_value, (int, float))):
            raise ValueError("Saved TRAIN and Phase 3 EVAL thresholds must be numeric.")
        threshold = float(training_threshold_value)
        metrics_threshold = float(metrics_threshold_value)
        if not math.isfinite(threshold) or not math.isfinite(metrics_threshold):
            raise ValueError("Saved TRAIN and Phase 3 EVAL thresholds must be finite.")
        if threshold != metrics_threshold:
            raise ValueError("Saved training and Phase 3 recomputed thresholds disagree.")
        model, scaler, device = load_model(run, meta)
        return cls(model, scaler, device, meta["config"], threshold)

    def infer(self, audio):
        frontend = self.config["frontend"]
        logmel = extract(audio, frontend, "logmel")
        inputs = tensor_input(logmel, self.scaler, self.config["cnn"]["frames"])
        with torch.no_grad():
            logit = float(self.model(inputs[None].to(self.device)).cpu().item())
        return logit, logmel


@dataclass
class FrozenSVM:
    model: object
    config: dict
    threshold: float

    @classmethod
    def from_artifacts(cls, metadata_directory, artifact_directory):
        metadata_directory = Path(metadata_directory)
        artifact_directory = Path(artifact_directory)
        model_path = artifact_directory / "model.joblib"
        scaler_path = artifact_directory / "scaler.joblib"
        if not model_path.is_file() or not scaler_path.is_file():
            return None
        meta = json.loads((metadata_directory / "training.json").read_text(encoding="utf-8"))
        model, _, _ = load_model(artifact_directory, meta)
        sidecar_scaler = joblib.load(scaler_path)
        embedded_scaler = model.named_steps["standardscaler"]
        if not (np.array_equal(sidecar_scaler.mean_, embedded_scaler.mean_) and
                np.array_equal(sidecar_scaler.scale_, embedded_scaler.scale_)):
            raise ValueError("SVM sidecar scaler differs from the frozen pipeline scaler")
        return cls(model, meta["config"], float(meta["threshold_selection"]["threshold"]))

    def infer(self, audio):
        frontend = self.config["frontend"]
        features = pooled(extract(audio, frontend, "mfcc"))[None, :]
        return float(self.model.decision_function(features)[0])
