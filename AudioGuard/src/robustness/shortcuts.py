"""Fixed waveform nuisance features and amplitude-based edge-silence trimming.

Absolute dBFS thresholds are predeclared, not selected using eval errors. These
simple checks reveal possible shortcuts but do not establish causal attribution.
"""
import numpy as np


def activity(audio, dbfs):
    return np.flatnonzero(np.abs(audio) > 10 ** (dbfs / 20))


def nuisance(audio, sample_rate, dbfs):
    active = activity(audio, dbfs)
    leading = int(active[0]) if len(active) else len(audio)
    return np.array([len(audio) / sample_rate, leading / sample_rate,
                     np.sqrt(np.mean(audio.astype(np.float64) ** 2))])


def trim(audio, dbfs):
    active = activity(audio, dbfs)
    # Preserve a single silent sample rather than returning an invalid empty clip.
    return audio[active[0]:active[-1]+1] if len(active) else audio[:1]
