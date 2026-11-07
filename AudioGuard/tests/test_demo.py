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
