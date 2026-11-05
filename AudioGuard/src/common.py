"""Shared configuration, provenance and deterministic I/O for rerunnable experiments.

Paths are explicit and JSON rejects NaN so failures cannot look like valid metrics.
"""
import hashlib
import json
import platform
import random
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import numpy as np
import yaml


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    return cfg


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def provenance(cfg):
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "soundfile", "PyYAML", "imageio-ffmpeg", "torch", "matplotlib", "joblib", "gradio"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"
    return {"config": cfg, "python": platform.python_version(),
            "platform": platform.platform(), "packages": packages,
            "label_convention": {"bonafide": 0, "spoof": 1},
            "score_convention": "higher score means more likely spoof"}
