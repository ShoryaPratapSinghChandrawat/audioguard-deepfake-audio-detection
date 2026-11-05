"""Reproducible additive white noise and real MP3 round-trip via external FFmpeg.

MP3 is lossy compression, not mislabeled quantization. Missing codecs fail the
condition explicitly. No robustness condition is used to retune the threshold.
"""
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf


def noise(audio, snr_db, seed, utterance):
    # Per-trial seeds make noise independent of iteration order and other conditions.
    key = int.from_bytes(hashlib.sha256(utterance.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed ^ key)
    power = np.mean(audio.astype(np.float64) ** 2)
    if power == 0:
        raise ValueError(f"{utterance}: SNR undefined for silence")
    draw = rng.normal(size=len(audio))
    draw *= np.sqrt(power / (10 ** (snr_db / 10)) / np.mean(draw ** 2))
    # Do not clip: clipping would confound additive-noise with nonlinear distortion.
    return (audio + draw).astype(np.float32)


def ffmpeg_executable():
    executable = shutil.which("ffmpeg")
    if executable is not None:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("No FFmpeg: install pinned requirements or provide ffmpeg on PATH") from exc


def mp3(audio, sample_rate, bitrate):
    executable = ffmpeg_executable()
    with tempfile.TemporaryDirectory() as tmp:
        wav, encoded, decoded = [Path(tmp) / n for n in ("input.wav", "coded.mp3", "decoded.wav")]
        sf.write(wav, audio, sample_rate, subtype="FLOAT")
        # A list (not shell=True) works with spaces and on Windows.
        for args in (["-i", str(wav), "-c:a", "libmp3lame", "-b:a", f"{bitrate}k", str(encoded)],
                     ["-i", str(encoded), "-ar", str(sample_rate), "-ac", "1", str(decoded)]):
            subprocess.run([executable, "-v", "error", "-y", *args], check=True, capture_output=True)
        result, rate = sf.read(decoded, dtype="float32")
        if rate != sample_rate:
            raise ValueError("Unexpected MP3 decode rate")
        return result
