# AudioGuard local demo

Run from the Python project root with the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe demo\app.py --run results/real_cnn_logmel/cnn_logmel --dataset-config configs/baseline.local.yaml --port 7860
```

The app explicitly binds to `127.0.0.1` and disables Gradio sharing. It supports
WAV, FLAC, MP3, WebM and Opus file uploads and the browser microphone's
start/stop recording controls. Select **Analyse audio** after recording. Uploads
are limited to 25 MiB and 30 seconds; clips shorter than one second are accepted
with a warning. Empty, corrupt, unsupported, silent and over-limit inputs return
a safe user-facing message. Details go to the local
`%LOCALAPPDATA%/AudioGuard/logs/audioguard-demo.log` file.

## Frozen Phase 3 preprocessing

The demo uses `scripts.experiment.load_model` to validate and load the saved
`cnn_logmel` checkpoint and TRAIN-fitted normalization arrays. It reads the
frozen DEV threshold from the saved `eval_clean.json` and checks that it exactly
matches `training.json`.

The input path is shared by uploads and microphone recordings:

1. Check the file type and byte limit. If the decoded input is already mono at
   16 kHz, read it with the same `soundfile` float32 reader used by the dataset
   pipeline. No resampling occurs on that path.
2. Otherwise, FFmpeg decodes the audio, downmixes to mono and genuinely resamples
   it to 16 kHz. The temporary converted WAV uses float32 samples. The original
   waveform is never peak-normalized.
3. Compute the saved 64-band HTK-unit-peak Log-Mel representation with the
   frozen 512-sample Hann STFT window, 160-sample hop, 512-point FFT,
   0–8 kHz range and `log(power + 1e-8)` frontend.
4. Apply the saved TRAIN mean and standard deviation to each Mel band. Center
   crop sequences longer than 400 frames; right-pad shorter sequences with
   zeros after normalization. These are the same crop/pad semantics as the
   Phase 3 `tensor_input` function.
5. Run the frozen CNN and display its raw logit, loaded threshold, label and
   signed margin (logit minus threshold). The logit is not a calibrated
   probability.

The waveform and Log-Mel view are diagnostic plots from the same converted
waveform and frontend input used for prediction. A local SVM panel is shown only
when both local `model.joblib` and `scaler.joblib` artifacts are available; it is
labelled as a baseline and its files are not part of the source commit.

## Privacy and scope

Gradio's upload directory is process-specific and app-owned. The analyzed upload
is deleted in a `finally` block; unprocessed uploads in that app-owned folder are
removed when the app process exits. FFmpeg conversion files are created in a
temporary directory and removed automatically. EVAL example audio is read from
the dataset path in the local config and is not copied into the repository.

When run locally, audio stays on this machine. If hosted elsewhere, audio is sent
to that server. The detector was trained on ASVspoof 2019 LA only; it is not a
general deepfake detector and cannot prove identity or authenticity. Microphone
outputs are exploratory demonstrations, not benchmark results. Noise,
compression, recording devices and room acoustics can change predictions. The
limitations panel computes its reported values from the saved Phase 3 metrics
and Phase 4B metrics JSON files at runtime.

If automated browser microphone testing is unavailable, manually open the local
URL, click **Load an EVAL example** to confirm file analysis, then select the
microphone source, allow microphone permission, record and stop, and click
**Analyse audio**. Confirm a prediction and plots appear and the audio control
clears after analysis.
