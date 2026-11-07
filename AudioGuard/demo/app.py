"""Private, local-only Gradio interface for the frozen Phase 3 Log-Mel CNN."""
from __future__ import annotations

import argparse
import atexit
import filecmp
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Gradio stores incoming upload and microphone files here. This process-specific
# directory is app-owned and is removed on callback or process shutdown.
APP_TEMP_ROOT = Path(tempfile.mkdtemp(prefix="audioguard-phase5-gradio-")).resolve()
os.environ["GRADIO_TEMP_DIR"] = str(APP_TEMP_ROOT)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from demo.inference import (
    MAX_DURATION_SECONDS,
    MAX_UPLOAD_BYTES,
    DemoInputError,
    FrozenCNN,
    FrozenSVM,
    cleanup_upload,
    load_waveform,
)
from src.common import load_config
from src.data.protocol import parse_protocol

CNN_RUN_DEFAULT = ROOT / "results/real_cnn_logmel/cnn_logmel"
SVM_META_DEFAULT = ROOT / "results/real_svm/phase2_original/svm_mfcc"
SVM_ARTIFACT_DEFAULT = ROOT / "results/phase4a/artifact_inputs/svm_mfcc"
# Some Windows/Chromium builds label audio-only .webm files as video/webm,
# which Gradio's audio-only uploader rejects before AudioGuard sees the file.
# Re-tag only that browser File metadata; the original bytes and path stay intact.
WEBM_UPLOAD_MIME_COMPAT = """<script>
document.addEventListener("change", (event) => {
  const input = event.target;
  if (!(input instanceof HTMLInputElement) || input.type !== "file" || !input.files || input.files.length !== 1) return;
  const file = input.files[0];
  if (!file.name.toLowerCase().endsWith(".webm") || file.type !== "video/webm") return;
  const transfer = new DataTransfer();
  transfer.items.add(new File([file], file.name, {type: "audio/webm", lastModified: file.lastModified}));
  input.files = transfer.files;
}, true);
</script>"""
LOG_DIR = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "AudioGuard" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOGGER = logging.getLogger("audioguard.demo")
if not LOGGER.handlers:
    handler = logging.FileHandler(LOG_DIR / "audioguard-demo.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


def _cleanup_app_temp():
    # APP_TEMP_ROOT was freshly created by this process; never clean any other path.
    shutil.rmtree(APP_TEMP_ROOT, ignore_errors=True)


atexit.register(_cleanup_app_temp)


def _limitations_text(run):
    phase3_path = Path(run).resolve().parent / "phase3_recomputed_metrics.json"
    phase4_path = ROOT / "results/phase4b/optimized_results/phase4b_metrics.json"
    try:
        phase3 = json.loads(phase3_path.read_text(encoding="utf-8"))
        phase4 = json.loads(phase4_path.read_text(encoding="utf-8"))
        clean_eer = float(phase3["cnn"]["eval"]["nearest_point_eer"])

        def pct(numerator, denominator):
            return 100.0 * numerator / denominator if denominator else float("nan")

        conditions = phase4["conditions"]
        noise = conditions["noise_20dB"]["CNN"]
        mp3_64 = conditions["mp3_64kbps"]["CNN"]
        mp3_128 = conditions["mp3_128kbps"]["CNN"]
        predicted_bona_fide = pct(noise["tn"] + noise["fn"], noise["n"])
        mp3_64_bf_spoof = pct(mp3_64["fp"], mp3_64["tn"] + mp3_64["fp"])
        mp3_128_bf_spoof = pct(mp3_128["fp"], mp3_128["tn"] + mp3_128["fp"])
        evidence = (
            f"Saved Phase 3 clean EVAL nearest-point EER: {clean_eer:.2%}. "
            f"On the saved Phase 4B subset at the frozen threshold, the CNN labeled "
            f"{predicted_bona_fide:.1f}% of clips bona fide under 20 dB added noise. "
            f"Under MP3, it labeled {mp3_64_bf_spoof:.1f}% (64 kbps) and "
            f"{mp3_128_bf_spoof:.1f}% (128 kbps) of bona fide clips spoof."
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        LOGGER.exception("Saved limitation metrics could not be loaded")
        evidence = "Saved Phase 3/4B robustness metrics are unavailable; no numerical summary is shown."
    return (
        "### Scope and limitations\n"
        "Trained on ASVspoof 2019 LA only. This is not a general deepfake detector "
        "and is not proof of identity or authenticity. Microphone results are "
        "exploratory demonstrations, not benchmark results; room noise and "
        "compression can change predictions.\n\n" + evidence
    )


def _examples(dataset_config):
    cfg_path = Path(dataset_config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    if not cfg_path.is_file():
        return [], [], [], (
            "Official EVAL examples are unavailable because the local dataset config "
            "was not found. Upload or record audio to use the demo."
        )
    try:
        dataset = load_config(cfg_path)["dataset"]
        rows = parse_protocol(dataset["protocols"]["eval"], "eval")
    except FileNotFoundError:
        return [], [], [], (
            "Official EVAL examples are unavailable because the local EVAL protocol "
            "was not found. Upload or record audio to use the demo."
        )
    bona_fide = [r for r in rows if r["label"] == 0][:2]
    attacks = sorted({r["attack"] for r in rows if r["label"] == 1})
    selected_spoof = []
    for attack in attacks:
        row = next(r for r in rows if r["label"] == 1 and r["attack"] == attack)
        selected_spoof.append(row)
        if len(selected_spoof) == 2:
            break
    selected = bona_fide + selected_spoof
    if len(bona_fide) != 2 or len(selected_spoof) != 2 or len({r["attack"] for r in selected_spoof}) != 2:
        return [], [], [], (
            "Official EVAL examples are unavailable because the EVAL protocol did "
            "not contain enough examples. Upload or record audio to use the demo."
        )
    audio_root = Path(dataset["audio_roots"]["eval"])
    examples = []
    labels = []
    allowed_paths = []
    for row in selected:
        path = (audio_root / f"{row['utterance']}.flac").resolve()
        if not path.is_file():
            return [], [], [], (
                "Official EVAL examples are unavailable because selected EVAL audio "
                "is missing. Upload or record audio to use the demo."
            )
        examples.append([str(path)])
        allowed_paths.append(str(path))
        attack_text = f", {row['attack']}" if row["label"] == 1 else ""
        labels.append(f"{row['utterance']} — {'spoof' if row['label'] else 'bona fide'}{attack_text}")
    return examples, labels, allowed_paths, None


def _gradio_example_cache_paths(example_component, source_paths, temp_root=APP_TEMP_ROOT):
    """Return only Gradio's exact temporary copies of the four selected examples.

    Gradio 6 postprocesses ``gr.Examples`` audio into per-file copies in its
    temporary directory. Those files are not covered by ``allowed_paths`` when
    that argument is an explicit allowlist, so the browser cannot load them.
    Add the copies individually after confirming their names and bytes match the
    selected EVAL files; never expose the temporary directory itself.
    """
    temp_root = Path(temp_root).resolve()
    source_by_name = {Path(path).name: Path(path).resolve() for path in source_paths}
    cache_paths = []
    for processed in getattr(example_component, "non_none_processed_examples", {}).values():
        for value in processed:
            if not isinstance(value, dict) or not value.get("path"):
                continue
            cached = Path(value["path"]).resolve()
            if cached in source_by_name.values():
                continue
            if not cached.is_relative_to(temp_root):
                continue
            source = source_by_name.get(cached.name)
            if source is None or not cached.is_file() or not filecmp.cmp(cached, source, shallow=False):
                raise RuntimeError(f"Unexpected Gradio EVAL example cache file: {cached}")
            cache_paths.append(str(cached))

    if cache_paths and {Path(path).name for path in cache_paths} != set(source_by_name):
        raise RuntimeError("Gradio did not create one verified cache copy per EVAL example")
    return list(dict.fromkeys(cache_paths))


def _figure_image(fig):
    from io import BytesIO
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    image.load()
    buffer.close()
    return image


def _plots(audio, logmel):
    times = np.arange(len(audio), dtype=np.float64) / 16000.0
    wave_fig, wave_ax = plt.subplots(figsize=(9, 2.4))
    wave_ax.plot(times, audio, linewidth=0.55)
    wave_ax.set(xlabel="Time (seconds)", ylabel="Amplitude", title="Resampled mono waveform")
    wave_ax.grid(alpha=0.2)

    mel_fig, mel_ax = plt.subplots(figsize=(9, 3.2))
    duration = len(audio) / 16000.0
    mel_ax.imshow(logmel, origin="lower", aspect="auto", extent=(0, duration, 0, logmel.shape[0]),
                  interpolation="nearest", cmap="magma")
    mel_ax.set(xlabel="Time (seconds)", ylabel="Mel band", title="Log-Mel frontend input")
    return _figure_image(wave_fig), _figure_image(mel_fig)


def _build_with_allowed_paths(run_directory, dataset_config="configs/baseline.local.yaml",
                              svm_metadata_directory=SVM_META_DEFAULT,
                              svm_artifact_directory=SVM_ARTIFACT_DEFAULT):
    run = Path(run_directory)
    if not run.is_absolute():
        run = ROOT / run
    cnn = FrozenCNN.from_run(run)
    try:
        svm = FrozenSVM.from_artifacts(svm_metadata_directory, svm_artifact_directory)
    except Exception:
        LOGGER.exception("Optional SVM baseline could not be loaded")
        svm = None
    example_rows, example_labels, allowed_paths, examples_notice = _examples(dataset_config)

    def analyse(path):
        try:
            audio, warning = load_waveform(path, temp_root=APP_TEMP_ROOT)
            logit, logmel = cnn.infer(audio)
            label = "spoof" if logit >= cnn.threshold else "bona fide"
            margin = logit - cnn.threshold
            output = {
                "predicted_label": label,
                "raw_logit": logit,
                "frozen_dev_threshold": cnn.threshold,
                "margin_from_threshold": margin,
                "score_note": "Raw model logit; this score is not a calibrated probability.",
                "warning": warning,
            }
            if svm is not None:
                svm_score = svm.infer(audio)
                output["SVM baseline"] = {
                    "predicted_label": "spoof" if svm_score >= svm.threshold else "bona fide",
                    "raw_score": svm_score,
                    "frozen_dev_threshold": svm.threshold,
                    "margin_from_threshold": svm_score - svm.threshold,
                }
            waveform_plot, logmel_plot = _plots(audio, logmel)
            return output, waveform_plot, logmel_plot, None
        except DemoInputError as exc:
            LOGGER.info("Rejected upload: %s", exc)
            return {"error": str(exc)}, None, None, None
        except Exception:
            LOGGER.exception("Unexpected local inference failure")
            return {"error": "Audio could not be analyzed. Check the file and try again."}, None, None, None
        finally:
            try:
                cleanup_upload(path, APP_TEMP_ROOT)
            except OSError:
                LOGGER.exception("Could not remove app-owned upload temporary file")

    size_mib = MAX_UPLOAD_BYTES / (1024 * 1024)
    notice = (
        "**Privacy:** when run locally, audio stays on this machine. If hosted "
        "elsewhere, audio is sent to that server. Audio is removed from this app's "
        "temporary storage after analysis.\n\n"
        f"Upload limit: {size_mib:.0f} MiB. Maximum duration: "
        f"{MAX_DURATION_SECONDS:.0f} seconds. Audio shorter than 1 second is "
        "accepted with a warning. Silent, empty, corrupt, and unsupported inputs are rejected."
    )
    preprocessing = (
        "**Preprocessing parity:** 16 kHz mono float32 input; the training "
        "`soundfile` reader is used directly when the input already has that "
        "format. Other supported inputs are decoded, downmixed to mono, and "
        "resampled to 16 kHz with FFmpeg. No peak normalization is applied. "
        "The saved 64-band Log-Mel frontend, saved TRAIN mean/std, and the "
        "400-frame center-crop or right zero-pad are used."
    )
    with gr.Blocks(title="AudioGuard local research demo") as app:
        gr.Markdown("# AudioGuard — local research demo")
        gr.Markdown(notice)
        gr.Markdown(_limitations_text(run))
        gr.Markdown(preprocessing)
        if svm is None:
            gr.Markdown("**SVM baseline:** unavailable (both local model and scaler files are required).")
        else:
            gr.Markdown("**SVM baseline:** Phase 2 MFCC/SVM comparator; shown separately from the Phase 3 CNN.")
        audio = gr.Audio(type="filepath", label="Upload a file or record with the browser microphone",
                         sources=["upload", "microphone"], format=None)
        gr.Markdown("Use the microphone's start/stop controls, then select **Analyse audio**.")
        button = gr.Button("Analyse audio")
        result = gr.JSON(label="Prediction (raw scores, not probabilities)")
        with gr.Row():
            wave_output = gr.Image(label="Waveform", type="pil")
            mel_output = gr.Image(label="Log-Mel spectrogram", type="pil")
        button.click(analyse, inputs=audio,
                     outputs=[result, wave_output, mel_output, audio])
        if example_rows:
            gr.Markdown("### Official ASVspoof 2019 LA EVAL examples\n" + "\n".join(f"- {label}" for label in example_labels))
            example_component = gr.Examples(
                examples=example_rows, inputs=audio, label="Load an EVAL example"
            )
            allowed_paths.extend(
                _gradio_example_cache_paths(example_component, allowed_paths)
            )
        else:
            gr.Markdown(f"### Official ASVspoof 2019 LA EVAL examples\n{examples_notice}")
    return app, allowed_paths


def build(run_directory, dataset_config="configs/baseline.local.yaml",
          svm_metadata_directory=SVM_META_DEFAULT,
          svm_artifact_directory=SVM_ARTIFACT_DEFAULT):
    """Build the demo while preserving the public Blocks-returning API."""
    app, _ = _build_with_allowed_paths(
        run_directory, dataset_config, svm_metadata_directory, svm_artifact_directory
    )
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Frozen Phase 3 CNN run directory")
    parser.add_argument("--dataset-config", default="configs/baseline.local.yaml")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    app, allowed_paths = _build_with_allowed_paths(args.run, args.dataset_config)
    app.launch(server_name="127.0.0.1", server_port=args.port, share=False,
               max_file_size=MAX_UPLOAD_BYTES, allowed_paths=allowed_paths,
               head=WEBM_UPLOAD_MIME_COMPAT)


if __name__ == "__main__":
    main()
