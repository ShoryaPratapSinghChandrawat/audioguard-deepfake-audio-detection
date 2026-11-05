"""Fixed log-spectral band mean/std features: a transparent, inexpensive baseline.

These are linear-frequency bands, NOT LFCC or mel features. Whole-utterance
pooling avoids cropping and padding biases, but discards temporal detail.
"""
import numpy as np
import soundfile as sf
from scipy.signal import stft


def read_audio(path, expected_rate):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != expected_rate:
        raise ValueError(f"{path}: sample rate {rate}, expected {expected_rate}; no silent resampling")
    if audio.shape[1] != 1:
        raise ValueError(f"{path}: expected mono, found {audio.shape[1]} channels")
    audio = audio[:, 0]
    if len(audio) == 0 or not np.isfinite(audio).all():
        raise ValueError(f"{path}: empty or non-finite waveform")
    return audio


def extract(audio, cfg):
    frame, hop, nfft = cfg["frame_length"], cfg["hop_length"], cfg["fft_size"]
    if not 0 < hop <= frame <= nfft or not 0 < cfg["bands"] <= nfft // 2 + 1:
        raise ValueError("Invalid STFT/band configuration")
    # Pad only very short clips so scipy never changes the configured window.
    if len(audio) < frame:
        audio = np.pad(audio, (0, frame - len(audio)))
    _, _, spectrum = stft(audio, fs=cfg["sample_rate"], window="hann", nperseg=frame,
                           noverlap=frame-hop, nfft=nfft, boundary=None, padded=False)
    power = np.abs(spectrum) ** 2
    bands = np.array_split(power, cfg["bands"], axis=0)
    log_bands = np.stack([np.log(np.maximum(b.mean(axis=0), cfg["log_floor"])) for b in bands])
    return np.concatenate([log_bands.mean(axis=1), log_bands.std(axis=1)]).astype(np.float64)


def matrix(rows, cfg, transform=None):
    features = []
    for row in rows:
        audio = read_audio(row["path"], cfg["sample_rate"])
        if transform is not None:
            audio = transform(audio, row)
        features.append(extract(audio, cfg))
    return np.stack(features)
