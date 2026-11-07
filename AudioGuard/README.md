# AudioGuard

A reproducible ASVspoof Logical Access research workspace: can detectors trained on
known synthesis attacks recognize genuine and spoofed speech, including attacks
absent from training, and withstand noise and lossy compression?

**Research status:** the official dataset is not present here. No real speech
benchmark performance or recruitment-project conclusion is claimed. Synthetic tone
fixtures validate software wiring only. Read the saved scope on every result.

This folder contains actual files, not patch instructions. The existing Ising-chain
project outside this folder is untouched.

## Download the folder as a ZIP

Use the separately presented **AudioGuard-source.zip** file, not Arena's “Download
all changes as a patch file” control. That control exports Git diffs and cannot be
changed by project code. The ZIP contains the complete `AudioGuard/` source folder,
README, glossary, configs, notebooks, tests and phase notes. It excludes datasets,
caches, virtual environments and trained weights. `MANIFEST.sha256.json` provides
file-integrity checks. Rebuild it with `python scripts/package.py`.

## Install

Use a supported Python interpreter (the executed version is in saved provenance).
All direct dependencies are pinned in `requirements.txt`. For the original lightweight
NumPy baseline alone, install `requirements-core.txt` instead.

Windows PowerShell:

```powershell
cd AudioGuard
py -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python scripts/research_smoke.py
```

Linux / Colab:

```bash
cd AudioGuard
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/research_smoke.py
```

In Windows commands below, use the virtual environment's interpreter unless activated.
`notebooks/colab_quickstart.ipynb` is also available. Windows and Colab instructions
are provided but were not executed in this Linux sandbox.

## Inspect and map the real official partitions

Obtain the official ASVspoof LA audio and labelled train/dev/eval protocols yourself;
respect its terms. Do not use PA protocols or randomly re-split trials. Dataset
provenance cannot be authenticated by trial-format checks alone.

- Set `dataset.root` in `configs/baseline.yaml` to the extracted real dataset.
- Run `python scripts/audioguard.py inspect` to print actual files, protocol preview
  lines and field counts. No directory layout is prescribed.
- Fill `protocols` and `audio_roots` for each official partition using that output.
  YAML paths are relative to this workspace or absolute; forward slashes work on Windows.
- Run `python scripts/audioguard.py audit`. The unchanged loader validates protocol
  fields, rejects missing/ambiguous audio and prints speaker and utterance overlaps.
  It fails on leakage before model fitting. It logs full and used sample counts,
  protocol hashes and explicit subset scope. A sorted-prefix subset is debugging
  only, never a representative full-dataset result.

## Research experiments

Inspect `configs/features.yaml` and `configs/research.yaml` before running:

```bash
python scripts/experiment.py train --config configs/research.yaml
python scripts/experiment.py evaluate --config configs/research.yaml
python scripts/experiment.py report --config configs/research.yaml
python scripts/verify_research.py --config configs/research.yaml
```

Alternatively, `suite` performs these model phases in order: train every model,
then evaluate every model, then generate reports. It does not fit after seeing eval.
Choose individual models using `--model svm_mfcc`, `--model cnn_logmel`,
`--model cnn_mfcc` or `--model shortcut`. The YAML model list selects the CNN-input
ablation; the frontend and all optimizer settings are frozen into training metadata.
Use a fresh YAML output path for each experiment. Completed models and condition
results refuse overwrite; failures are retained, not silently replaced.

### What is implemented

- Cached log-Mel matrices, MFCCs with regression delta and delta-delta, and pooled
  MFCC mean/std vectors. Exact frontend parameters are in the versioned feature YAML.
- Standardized RBF SVM with train-only preprocessing and spoof-positive decision scores.
- A compact convolutional network: Conv/BatchNorm/ReLU/MaxPool blocks, global average
  pooling, dropout and a single logit. Weighted binary cross-entropy, AdamW and
  dev-loss early stopping. The train-only memorization sanity check must pass before
  a freshly initialized model is trained normally. Actual subset IDs are recorded.
- Precision, recall, F-score, ROC-AUC, empirical EER, confusion matrices, ROC curves,
  training curves, per-attack EER and per-speaker error rates.
- A duration/leading-silence/RMS-only classifier, fixed silence-trimming counterfactual,
  additive waveform-SNR noise and actual MP3 encode/decode.
- Ranked false-positive/false-negative examples with IDs, paths, scores and margins.
- Local Gradio demo displaying only saved result-file metrics, plus a limitations notice.

**Selection policy:** bonafide is the negative class; spoof is positive. Higher
scores mean spoof. Operating thresholds are selected on clean dev only and frozen
for every eval condition. Eval EER is a diagnostic ranking statistic, not a new
threshold. Model checkpoint selection uses dev loss, never eval loss. No future
hyperparameter experiment may be tuned using these eval results.

**Attack claims:** per-attack EER uses that attack plus the shared eval bonafide pool;
spoof-only ROC is undefined. Attack-ID presence/absence is checked against full
protocols. Protocol files do not describe synthesis algorithms, so they cannot prove
that a new ID is a new method or a reuse of an old one. Requested reuse mappings
are clearly unverified unless referenced metadata is supplied. See
`docs/research_guide.md` and each run's `attack_verification.json`.

## Results and reproducibility

Each model's run folder holds its complete config/provenance, dataset audit,
train-audio hashes, model and scaler checksums, dev selection, raw-score CSVs,
eval JSON, analysis, figures and generated report. Reports and plots read saved JSON;
verification recomputes metrics from saved CSVs. There are no manually entered
benchmark values in this README. Generated synthetic validation status follows below.

Preserve ignored experiment outputs separately. Never commit datasets, feature caches
or weights. Package pins and seeds aid replay but do not guarantee bitwise equality
across hardware. CUDA is selected automatically when usable; otherwise CPU. CUDA
execution and full-dataset resource usage remain untested here. Only fitting/inference
of the CNN use the selected Torch device; handcrafted features run on CPU.

```powershell
.\.venv\Scripts\python.exe demo\app.py --run results/real_cnn_logmel/cnn_logmel --dataset-config configs/baseline.local.yaml --port 7860
```

The Phase 3 demo binds only to `127.0.0.1` with Gradio sharing disabled. It accepts
WAV, FLAC, MP3, WebM and Opus uploads or browser-microphone recordings, enforces
25 MiB and 30-second limits, and removes app-managed audio after analysis. See
[`demo/README.md`](demo/README.md) for the frozen CNN preprocessing path, privacy
notes, validation steps and error behavior.

Do not load untrusted pickle/joblib checkpoints. The demo is not a forensic or
production security tool. Native FFmpeg is preferred, with a pinned executable-wheel
fallback; errors are saved if a platform lacks a compatible codec.

## Folder map and technical review

`configs/`, `src/{data,features,models,evaluation,robustness}/`, `scripts/`, `tests/`,
`notebooks/`, `demo/`, `docs/`, `data/`, `models/`, `results/` and `reports/figures/`
are real workspace folders. Read `docs/glossary.md`, `docs/research_guide.md` and the
phase notes before defending the project. Existing baseline commands remain available
through `scripts/audioguard.py`; its original leakage checks and threshold selection
have not been replaced.

<!-- VALIDATION:START -->
Scope: SYNTHETIC TONE SOFTWARE TEST ONLY; NOT ASVSPOOF

- `-m pytest -q`: PASS.

```text
.......................                                                  [100%]
23 passed in 3.02s
```

- `scripts/experiment.py suite --config configs/research_smoke.yaml`: PASS.
- `scripts/verify_research.py --config configs/research_smoke.yaml`: PASS.
Evidence source: `docs/research_validation.json`. No official-data metrics are available.
<!-- VALIDATION:END -->
