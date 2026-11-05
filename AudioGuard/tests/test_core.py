"""Small invariant tests prevent label reversal, leakage and metric convention errors."""
import numpy as np
import pytest
from src.data.protocol import parse_protocol, verify_disjoint
from src.evaluation.metrics import empirical_eer, metrics, attack_metrics
from src.features.spectrum import extract
from src.models.logistic import fit, score, select_device
from src.robustness.transforms import noise
from src.common import load_config


def test_score_orientation_and_threshold():
    y, s = [0, 0, 1, 1], [.1, .2, .8, .9]
    selection = empirical_eer(y, s)
    assert selection["eer"] == 0
    result = metrics(y, s, selection["threshold"])
    assert result["fp"] == result["fn"] == 0
    assert metrics(y, s, .95)["fn"] == 2


def test_ties_and_reversed_scores():
    assert empirical_eer([0, 1], [.5, .5])["eer"] == .5
    assert metrics([0, 1], [.9, .1], .5)["accuracy"] == 0
    with pytest.raises(ValueError):
        metrics([0, 1], [np.nan, .2], .5)


def test_leakage_fails():
    data = {p: [{"speaker": p, "utterance": p}] for p in ("train", "dev", "eval")}
    verify_disjoint(data)
    data["eval"][0]["speaker"] = "train"
    with pytest.raises(ValueError):
        verify_disjoint(data)


def test_protocol_rejects_unknown_format(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("speaker trial spoof\n")
    with pytest.raises(ValueError):
        parse_protocol(path, "train")


def test_protocol_labels(tmp_path):
    path = tmp_path / "ok.txt"
    path.write_text("s u0 - - bonafide\ns u1 - A01 spoof\n")
    assert [r["label"] for r in parse_protocol(path, "train")] == [0, 1]


def test_features_short_clip():
    cfg = load_config("configs/baseline.yaml")["features"]
    x = extract(np.zeros(20), cfg)
    assert x.shape == (2 * cfg["bands"],)
    assert np.isfinite(x).all()


def test_noise_reproducible_and_snr():
    x = np.ones(1000, dtype=np.float32) * .1
    a = noise(x, 10, 2026, "trial")
    assert np.array_equal(a, noise(x, 10, 2026, "trial"))
    measured = 10 * np.log10(np.mean(x ** 2) / np.mean((a-x) ** 2))
    assert abs(measured - 10) < .001
    with pytest.raises(ValueError):
        noise(np.zeros(10), 10, 1, "silent")


def test_model_seed_and_training_only_scaler():
    cfg = load_config("configs/smoke.yaml")["model"]
    cfg["device"] = "cpu"
    x, y = np.array([[-2.], [-1.], [1.], [2.]]), np.array([0, 0, 1, 1])
    a, _, _ = fit(x, y, cfg, 2)
    b, _, _ = fit(x, y, cfg, 2)
    assert np.array_equal(a["weights"], b["weights"])
    assert a["mean"][0] == x.mean()
    assert score(x, a)[-1] > score(x, a)[0]
    assert select_device("cpu") == "cpu"


def test_attack_coverage():
    rows = [{"label": 0, "attack": "-"}, {"label": 1, "attack": "A99"}]
    result = attack_metrics(rows, np.array([.1, .2]), .5, ["A01"])
    assert result["A99"] == {"n": 1, "frr": 1.0, "seen_in_train": False}


def test_frozen_threshold_not_eval_eer():
    # Changing eval labels changes diagnostics but cannot change the supplied threshold.
    threshold = empirical_eer([0, 0, 1, 1], [.1, .3, .7, .9])["threshold"]
    a = metrics([0, 0, 1, 1], [.2, .6, .4, .8], threshold)
    b = metrics([1, 1, 0, 0], [.2, .6, .4, .8], threshold)
    assert a["threshold"] == b["threshold"] == threshold


def test_audio_validation(tmp_path):
    import soundfile as sf
    from src.features.spectrum import read_audio
    path = tmp_path / "wrong.wav"
    sf.write(path, np.zeros(100), 8000)
    with pytest.raises(ValueError, match="sample rate"):
        read_audio(path, 16000)
    sf.write(path, np.zeros((100, 2)), 16000)
    with pytest.raises(ValueError, match="mono"):
        read_audio(path, 16000)


def test_mp3_roundtrip():
    from src.robustness.transforms import mp3
    t = np.arange(8000) / 16000
    decoded = mp3((.2 * np.sin(2 * np.pi * 500 * t)).astype(np.float32), 16000, 32)
    assert decoded.ndim == 1 and len(decoded) > 0
    assert np.isfinite(decoded).all()
