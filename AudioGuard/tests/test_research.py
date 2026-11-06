"""Research-component invariants tested with synthetic signals, never benchmark claims.

Tests lock down cache identity, spoof-positive orientation, train-only sanity data,
checkpoint restoration, and the distinction between attack IDs and synthesis reuse.
"""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
import torch
from src.common import load_config
from src.features.acoustic import extract, cached, pooled, delta, mel_filters
from src.models.cnn import SmallCNN, tensor_input, balanced_sanity_rows
from src.evaluation.analysis import summary, attack_claims
from src.robustness.shortcuts import nuisance, trim


def feature_cfg(tmp_path):
    cfg = load_config("configs/features.yaml")
    cfg["cache_dir"] = str(tmp_path / "cache")
    return cfg


def test_frontend_shapes_formula_and_finite(tmp_path):
    cfg = feature_cfg(tmp_path)
    silent = np.zeros(16000)
    mel = extract(silent, cfg, "logmel")
    mfcc = extract(silent, cfg, "mfcc")
    assert mel.shape[0] == cfg["n_mels"]
    assert mfcc.shape[0] == cfg["n_mfcc"] * 3
    assert pooled(mfcc).shape == (cfg["n_mfcc"] * 6,)
    assert np.allclose(mel, np.log(cfg["eps"]))
    assert np.isfinite(mfcc).all()
    assert np.allclose(delta(np.ones((20, 2)), 9), 0)
    assert np.all(mel_filters(cfg).sum(axis=1) > 0)


def test_cache_hits_and_invalidates(tmp_path):
    cfg = feature_cfg(tmp_path)
    audio = tmp_path / "clip.wav"
    sf.write(audio, np.zeros(16000), 16000)
    a = cached(audio, cfg, "logmel")
    assert np.array_equal(a, cached(audio, cfg, "logmel"))
    assert len(list((tmp_path / "cache").glob("*.npy"))) == 1
    changed = dict(cfg, eps=1e-5)
    cached(audio, changed, "logmel")
    sf.write(audio, np.ones(16000) * .1, 16000)
    cached(audio, cfg, "logmel")
    assert len(list((tmp_path / "cache").glob("*.npy"))) == 3


def test_cnn_both_inputs_and_logits():
    cfg = load_config("configs/research.yaml")["cnn"]
    torch.set_num_threads(1)
    net = SmallCNN(cfg).eval()
    with torch.no_grad():
        for dimension in (64, 60):
            assert net(torch.zeros(2, 1, dimension, 64)).shape == (2,)
    scaler = {"mean": np.zeros(60), "std": np.ones(60)}
    assert tensor_input(np.ones((60, 2)), scaler, 64).shape == (1, 60, 64)


def test_cnn_predict_returns_raw_logits(monkeypatch):
    from src.models import cnn
    cfg = load_config("configs/research.yaml")["cnn"]
    torch.set_num_threads(1)
    model = SmallCNN(cfg).eval()
    batch = torch.ones(2, 1, 64, 64)
    monkeypatch.setattr(cnn, "make_loader", lambda *a, **k: [(batch, torch.zeros(2))])
    with torch.no_grad():
        expected = model(batch).numpy()
    actual = cnn.predict(model, [{}, {}], {}, "logmel", {}, cfg, "cpu")
    assert np.allclose(actual, expected)
    assert not np.allclose(actual, torch.sigmoid(torch.from_numpy(expected)).numpy())


def test_sanity_unique_and_train_only():
    rows = [{"partition": "train", "label": i % 2, "utterance": str(i)} for i in range(80)]
    selected = balanced_sanity_rows(rows, 64, 1)
    assert len({r["utterance"] for r in selected}) == 64
    assert sum(r["label"] for r in selected) == 32
    with pytest.raises(ValueError):
        balanced_sanity_rows(rows[:10], 64, 1)


def test_analysis_errors_and_per_attack_eer():
    rows = [{"speaker": "speaker", "utterance": str(i), "path": "test.wav", "label": y, "attack": a}
            for i, (y, a) in enumerate([(0, "-"), (0, "-"), (1, "A07"), (1, "A08")])]
    result = summary(rows, np.array([.8, .1, .9, .2]), .5, 2)
    assert result["metrics"]["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["metrics"]["precision"] == .5
    assert result["per_attack"]["A07"]["metrics"]["n"] == 3  # two bonafide plus one attack
    assert result["top_errors"]["false_positives"][0]["utterance"] == "0"
    assert result["top_errors"]["false_negatives"][0]["utterance"] == "3"
    assert result["per_speaker"]["speaker"]["error_rate"] == .5


def test_silence_shortcuts_and_all_silence():
    audio = np.r_[np.zeros(100), np.ones(200)*.1, np.zeros(100)]
    values = nuisance(audio, 16000, -40)
    assert values[0] == len(audio)/16000 and values[1] == 100/16000
    assert len(trim(audio, -40)) == 200
    assert len(trim(np.zeros(20), -40)) == 1


def test_attack_reuse_not_claimed_from_protocols(tmp_path):
    paths = {}
    for part, attack in (("train", "A04"), ("dev", "A04"), ("eval", "A16")):
        path = tmp_path / (part + ".txt")
        path.write_text(f"s u0 - - bonafide\ns u1 - {attack} spoof\n")
        paths[part] = str(path)
    result = attack_claims({"protocols": paths})
    item = result["annotations"]["A16"]
    assert item["unseen_id_verified"]
    assert item["parent_present_in_train_protocol"]
    assert item["reuse_status"].startswith("UNVERIFIED")
    assert not result["annotations"]["A07"]["unseen_id_verified"]


def test_sanity_rejects_dev_trials():
    rows = [{"partition": "dev", "label": i % 2, "utterance": str(i)} for i in range(64)]
    with pytest.raises(ValueError, match="train-partition"):
        balanced_sanity_rows(rows, 64, 1)


def test_early_stop_restores_best_dev_checkpoint(monkeypatch):
    from src.models import cnn
    cfg = copy.deepcopy(load_config("configs/research.yaml")["cnn"])
    cfg.update(epochs=10, patience=2, cpu_threads=1)
    losses, training_calls = iter([.7, .3, .4, .6]), []
    monkeypatch.setattr(cnn, "make_loader", lambda *a, **k: [])

    def fake_epoch(model, loader, loss_fn, device, optimizer=None, return_scores=False):
        if optimizer is not None:
            training_calls.append(True)
            with torch.no_grad():
                next(model.parameters()).fill_(len(training_calls))
            return {"loss": .5, "accuracy": .5, "n": 2}
        result = {"loss": next(losses), "accuracy": .5, "n": 2}
        return (result, (np.array([0, 1]), np.array([-.4, .6]))) if return_scores else result

    monkeypatch.setattr(cnn, "epoch", fake_epoch)
    rows = [{"label": 0}, {"label": 1}]
    model, info = cnn.fit(rows, rows, {}, "logmel", {}, cfg, 1, "cpu")
    assert info["best_epoch"] == 2
    assert len(info["history"]) == 4
    assert all("dev_eer" in row for row in info["history"])
    assert info["pos_weight"] == 1
    assert torch.all(next(model.parameters()) == 2)


def test_loaded_models_reject_tampering(tmp_path):
    from scripts.experiment import load_model
    path = tmp_path / "model.joblib"
    path.write_bytes(b"untrusted model content")
    meta = {"name": "svm_mfcc", "config": {}, "artifact": path.name, "artifact_sha256": "invalid"}
    with pytest.raises(ValueError, match="checksum"):
        load_model(tmp_path, meta)


def test_source_zip_contains_data_and_model_code(tmp_path):
    import subprocess
    import sys
    from zipfile import ZipFile
    project_root = Path(__file__).resolve().parents[1]
    tracked_manifest = project_root / "MANIFEST.sha256.json"
    manifest_before = tracked_manifest.read_bytes()
    output_dir = tmp_path / "package"
    subprocess.run(
        [sys.executable, "scripts/package.py", "--output-dir", str(output_dir)],
        check=True,
        capture_output=True,
    )
    with ZipFile(output_dir / "AudioGuard-source.zip") as archive:
        names = set(archive.namelist())
        assert "AudioGuard/src/data/protocol.py" in names
        assert "AudioGuard/src/models/cnn.py" in names
        assert "AudioGuard/src/features/acoustic.py" in names
        assert not any("/.venv/" in name or "/feature_cache/" in name for name in names)
        assert not any(name.startswith("AudioGuard/results/") and not name.endswith(".gitkeep") for name in names)
        manifest = json.loads(archive.read("AudioGuard/MANIFEST.sha256.json"))
        import hashlib
        for path, expected in manifest.items():
            assert hashlib.sha256(archive.read("AudioGuard/" + path)).hexdigest() == expected
    assert tracked_manifest.read_bytes() == manifest_before
