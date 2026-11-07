from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from demo.inference import (
    MAX_DURATION_SECONDS,
    MAX_UPLOAD_BYTES,
    DemoInputError,
    cleanup_upload,
    load_waveform,
)
from src.features.spectrum import read_audio


def test_16khz_mono_uses_the_training_decode_path(tmp_path):
    source = tmp_path / "native.wav"
    t = np.arange(16000, dtype=np.float64) / 16000
    original = (.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(source, original, 16000, subtype="FLOAT")

    audio, warning = load_waveform(source, temp_root=tmp_path)

    assert warning is None
    assert np.array_equal(audio, read_audio(source, 16000))


def test_non_native_stereo_audio_is_downmixed_and_resampled(tmp_path):
    source = tmp_path / "stereo_48000.wav"
    rate = 48000
    t = np.arange(rate, dtype=np.float64) / rate
    stereo = np.column_stack((.2 * np.sin(2 * np.pi * 440 * t),
                              .1 * np.sin(2 * np.pi * 18000 * t))).astype(np.float32)
    sf.write(source, stereo, rate, subtype="FLOAT")

    audio, warning = load_waveform(source, temp_root=tmp_path)

    assert warning is None
    assert audio.dtype == np.float32 and audio.ndim == 1
    assert len(audio) == 16000
    naive_stride_downmix = stereo[::3].mean(axis=1)
    assert np.max(np.abs(audio - naive_stride_downmix)) > 1e-3


def test_short_audio_warns_without_rejection(tmp_path):
    source = tmp_path / "short.wav"
    sf.write(source, np.full(3200, .1, dtype=np.float32), 16000, subtype="FLOAT")

    audio, warning = load_waveform(source, temp_root=tmp_path)

    assert len(audio) == 3200
    assert warning == "Audio is shorter than 1 second; this prediction may be unreliable."


@pytest.mark.parametrize("filename,contents,message", [
    ("empty.wav", b"", "Audio file is empty."),
    ("text.wav", b"this is not audio", "Audio could not be decoded. Check the file and try again."),
    ("audio.txt", b"x", "Unsupported audio type. Use WAV, FLAC, MP3, WebM, or Opus."),
])
def test_empty_corrupt_and_unsupported_files_have_safe_messages(tmp_path, filename, contents, message):
    source = tmp_path / filename
    source.write_bytes(contents)

    with pytest.raises(DemoInputError, match=message.replace(".", r"\.")):
        load_waveform(source, temp_root=tmp_path)


def test_silent_audio_is_rejected(tmp_path):
    source = tmp_path / "silence.wav"
    sf.write(source, np.zeros(16000, dtype=np.float32), 16000, subtype="FLOAT")

    with pytest.raises(DemoInputError, match="Audio is silent"):
        load_waveform(source, temp_root=tmp_path)


def test_upload_size_limit_is_checked_before_decode(tmp_path):
    source = tmp_path / "too_large.wav"
    with source.open("wb") as handle:
        handle.truncate(MAX_UPLOAD_BYTES + 1)

    with pytest.raises(DemoInputError, match="25 MiB upload limit"):
        load_waveform(source, temp_root=tmp_path)


def test_duration_limit_is_checked_from_audio_metadata(tmp_path):
    source = tmp_path / "too_long.wav"
    samples = int((MAX_DURATION_SECONDS + .1) * 16000)
    sf.write(source, np.full(samples, .01, dtype=np.float32), 16000, subtype="FLOAT")

    with pytest.raises(DemoInputError, match="30-second limit"):
        load_waveform(source, temp_root=tmp_path)


def test_cleanup_only_removes_app_owned_temporary_files(tmp_path):
    app_temp = tmp_path / "app-temp"
    app_temp.mkdir()
    upload = app_temp / "upload.wav"
    upload.write_bytes(b"temporary upload")
    external = tmp_path / "user-owned.wav"
    external.write_bytes(b"outside app temp")

    assert cleanup_upload(upload, app_temp)
    assert not upload.exists()
    assert not cleanup_upload(external, app_temp)
    assert external.exists()


def test_examples_allow_only_the_four_selected_eval_audio_files(tmp_path, monkeypatch):
    from demo import app as demo_app

    audio_root = tmp_path / "external" / "ASVspoof2019_LA_eval" / "flac"
    audio_root.mkdir(parents=True)
    rows = [
        {"utterance": "bf_one", "label": 0, "attack": "-"},
        {"utterance": "bf_two", "label": 0, "attack": "-"},
        {"utterance": "sp_a01", "label": 1, "attack": "A01"},
        {"utterance": "sp_a02", "label": 1, "attack": "A02"},
        {"utterance": "sp_a03", "label": 1, "attack": "A01"},
    ]
    for row in rows:
        (audio_root / f"{row['utterance']}.flac").write_bytes(b"fixture")

    monkeypatch.setattr(
        demo_app,
        "load_config",
        lambda _: {"dataset": {"protocols": {"eval": "protocol.txt"},
                               "audio_roots": {"eval": str(audio_root)}}},
    )
    monkeypatch.setattr(demo_app, "parse_protocol", lambda *_: rows)

    examples, labels, allowed_paths = demo_app._examples("unused.local.yaml")

    expected = [
        str((audio_root / "bf_one.flac").resolve()),
        str((audio_root / "bf_two.flac").resolve()),
        str((audio_root / "sp_a01.flac").resolve()),
        str((audio_root / "sp_a02.flac").resolve()),
    ]
    assert allowed_paths == expected
    assert examples == [[path] for path in expected]
    assert len(set(allowed_paths)) == 4
    assert len(labels) == 4


def test_gradio_example_cache_allows_only_byte_identical_selected_files(tmp_path):
    from demo import app as demo_app

    source_root = tmp_path / "dataset" / "eval"
    cache_root = tmp_path / "app-temp"
    source_root.mkdir(parents=True)
    cache_root.mkdir()
    source_paths = []
    processed = {}
    for index, utterance in enumerate(("bf_one", "bf_two", "sp_a01", "sp_a02")):
        source = source_root / f"{utterance}.flac"
        source.write_bytes(f"audio-{index}".encode())
        cached = cache_root / f"hash-{index}" / source.name
        cached.parent.mkdir()
        cached.write_bytes(source.read_bytes())
        source_paths.append(str(source.resolve()))
        processed[index] = [{"path": str(cached)}]

    class Examples:
        non_none_processed_examples = processed

    allowed_copies = demo_app._gradio_example_cache_paths(
        Examples(), source_paths, cache_root
    )

    assert allowed_copies == [
        str((cache_root / f"hash-{index}" / Path(source).name).resolve())
        for index, source in enumerate(source_paths)
    ]
    assert all(Path(path).is_file() for path in allowed_copies)
    assert all(Path(path).is_relative_to(cache_root.resolve()) for path in allowed_copies)
