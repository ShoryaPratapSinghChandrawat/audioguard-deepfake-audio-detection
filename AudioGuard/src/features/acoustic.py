"""Auditable log-Mel and MFCC frontends with content-addressed disk caching.

HTK-spaced triangular mel filters are unit-peak (not Slaney normalized). STFT
power is converted using log(x + eps), not a dB floor. MFCC uses orthonormal
DCT-II; regression deltas and repeated deltas retain temporal change.
"""
import hashlib
import json
import os
import tempfile
from pathlib import Path
import numpy as np
from scipy.fft import dct
from scipy.ndimage import correlate1d
from scipy.signal import stft
from src.common import digest
from src.features.spectrum import read_audio

FRONTEND_VERSION = "htk-unitpeak-log-plus-eps-dct2-v1"


def mel_filters(cfg):
    sr, nfft = cfg["sample_rate"], cfg["n_fft"]
    low, high = cfg["f_min"], cfg["f_max"]
    if not 0 <= low < high <= sr / 2:
        raise ValueError("Mel frequency limits must be inside Nyquist")
    to_mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)
    edges = 700.0 * (10 ** (np.linspace(to_mel(low), to_mel(high), cfg["n_mels"] + 2) / 2595.0) - 1)
    frequencies = np.fft.rfftfreq(nfft, 1 / sr)
    left = (frequencies[None, :] - edges[:-2, None]) / (edges[1:-1] - edges[:-2])[:, None]
    right = (edges[2:, None] - frequencies[None, :]) / (edges[2:] - edges[1:-1])[:, None]
    filters = np.maximum(0, np.minimum(left, right))
    if np.any(filters.sum(axis=1) == 0):
        raise ValueError("Empty mel band: reduce n_mels or increase n_fft")
    return filters


def delta(values, width):
    if width < 3 or width % 2 != 1:
        raise ValueError("delta_width must be odd and at least three")
    offsets = np.arange(-(width // 2), width // 2 + 1, dtype=float)
    # Replicated boundaries also work for utterances shorter than the delta window.
    return correlate1d(values, offsets / np.sum(offsets ** 2), axis=1, mode="nearest")


def extract(audio, cfg, kind):
    if kind not in ("logmel", "mfcc"):
        raise ValueError(f"Unknown CNN/frontend input: {kind}")
    win, hop, nfft = cfg["win_length"], cfg["hop_length"], cfg["n_fft"]
    if not 0 < hop <= win <= nfft or cfg["eps"] <= 0:
        raise ValueError("Invalid STFT or log-epsilon settings")
    if not 0 < cfg["n_mfcc"] <= cfg["n_mels"]:
        raise ValueError("n_mfcc must fit within mel dimension")
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Expected nonempty, finite mono waveform")
    if len(audio) < win:
        audio = np.pad(audio, (0, win - len(audio)))
    _, _, spectrum = stft(audio, fs=cfg["sample_rate"], window="hann", nperseg=win,
                           noverlap=win-hop, nfft=nfft, boundary=None, padded=False)
    logmel = np.log(mel_filters(cfg) @ (np.abs(spectrum) ** 2) + cfg["eps"])
    if kind == "logmel":
        return logmel.astype(np.float32)
    mfcc = dct(logmel, type=2, norm="ortho", axis=0)[:cfg["n_mfcc"]]
    first = delta(mfcc, cfg["delta_width"])
    return np.concatenate([mfcc, first, delta(first, cfg["delta_width"])], axis=0).astype(np.float32)


def cached(path, cfg, kind):
    identity = {"audio_sha256": digest(path), "config": {k: v for k, v in cfg.items() if k != "cache_dir"},
                "kind": kind, "implementation": FRONTEND_VERSION}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    folder = Path(cfg["cache_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (key + ".npy")
    expected = cfg["n_mels"] if kind == "logmel" else cfg["n_mfcc"] * 3
    if target.exists():
        value = np.load(target, allow_pickle=False)
        if value.ndim != 2 or value.shape[0] != expected or not np.isfinite(value).all():
            raise ValueError(f"Corrupt feature cache: {target}; delete it and rerun")
        return value
    value = extract(read_audio(path, cfg["sample_rate"]), cfg, kind)
    # Atomic replace prevents partial .npy reads across DataLoader processes.
    with tempfile.NamedTemporaryFile(dir=folder, suffix=".npy", delete=False) as handle:
        temporary = Path(handle.name)
        np.save(handle, value, allow_pickle=False)
    try:
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return value


def pooled(values):
    return np.concatenate([values.mean(axis=1), values.std(axis=1)]).astype(np.float64)
